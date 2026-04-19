"""Decode saved Lyra-2 latents to video without re-running diffusion.

Loads the VAE only (~5 sec) and decodes latents saved with --save_latents_dir.
Supports speed, reverse, loop, and frame-range options for real-time exploration.

Usage:
  python scripts/decode_latents.py --latent results/latents/00_zoom_in.pt --out results/decoded/zoom_in.mp4
  python scripts/decode_latents.py --latent results/latents/00_zoom_in.pt --out results/decoded/slow.mp4 --speed 0.5
  python scripts/decode_latents.py --latent results/latents/00_zoom_in.pt --out results/decoded/reverse.mp4 --reverse
  python scripts/decode_latents.py --latent results/latents/00_zoom_in.pt --out results/decoded/loop.mp4 --loop
"""
import argparse
import os
import sys
from contextlib import nullcontext

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_vae(vae_pth="checkpoints/vae/vae.pth", device="cuda", dtype=torch.bfloat16):
    from lyra_2._src.tokenizers.wan2pt1 import WanVAE
    print(f"Loading VAE from {vae_pth}...")
    vae = WanVAE(vae_pth=vae_pth, device=device, dtype=dtype, is_amp=False)
    print("VAE loaded.")
    return vae


@torch.no_grad()
def decode_latents(latents, start_index, vae_stats, vae_core, device, dtype):
    """Stream-decode all latent frames, return pixel video (B, C, T, H, W) in [-1, 1]."""
    img_mean = vae_stats["img_mean"].to(device)
    img_std = vae_stats["img_std"].to(device)
    video_mean = vae_stats["video_mean"].to(device)
    video_std = vae_stats["video_std"].to(device)
    scale_mean = vae_stats["scale_mean"]
    scale_inv_std = vae_stats["scale_inv_std"]
    if torch.is_tensor(scale_mean):
        scale_mean = scale_mean.to(device)
    if torch.is_tensor(scale_inv_std):
        scale_inv_std = scale_inv_std.to(device)
    z_dim = int(vae_stats["z_dim"])

    ctx = torch.amp.autocast("cuda", dtype=dtype) if torch.cuda.is_available() else nullcontext()
    dec_feat_cache = [None] * vae_core._conv_num
    frames_out = []

    T_lat = latents.shape[2]
    for t in range(T_lat):
        chunk = latents[:, :, t:t+1].to(device=device, dtype=dtype)

        if t == 0 and start_index == 0:
            mu = chunk * img_std.type_as(chunk) + img_mean.type_as(chunk)
        else:
            mu = chunk * video_std[:, :, :1].type_as(chunk) + video_mean[:, :, :1].type_as(chunk)

        if torch.is_tensor(scale_inv_std):
            z = mu / scale_inv_std.view(1, z_dim, 1, 1, 1).type_as(mu) + scale_mean.view(1, z_dim, 1, 1, 1).type_as(mu)
        else:
            z = mu / scale_inv_std + scale_mean

        with ctx:
            x = vae_core.conv2(z)
            out_t = vae_core.decoder(x, feat_cache=dec_feat_cache, feat_idx=[0])
        frames_out.append(out_t.float().cpu())

    video = torch.cat(frames_out, dim=2)  # (B, C, T_lat*fpl, H, W)
    return video[:, :, start_index:]      # trim init frames


def apply_playback_options(video, speed=1.0, reverse=False, loop=False, frame_range=None):
    """Post-process decoded pixel video tensor (B, C, T, H, W) in [-1, 1]."""
    T = video.shape[2]

    if frame_range is not None:
        s, e = frame_range
        video = video[:, :, s:e]

    if speed != 1.0:
        if speed < 1.0:
            factor = round(1.0 / speed)
            video = video.repeat_interleave(factor, dim=2)
        else:
            step = round(speed)
            video = video[:, :, ::step]

    if reverse:
        video = video.flip(dims=[2])

    if loop:
        video = torch.cat([video, video.flip(dims=[2])], dim=2)

    return video


def save_video(video, path, fps=16):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames_uint8 = ((video[0].clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)
    frames_np = frames_uint8.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    writer = imageio.get_writer(path, fps=fps, quality=5, macro_block_size=1,
                                output_params=["-f", "mp4"])
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()
    print(f"Saved {frames_np.shape[0]} frames → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", required=True, help="Path to saved .pt latent file")
    parser.add_argument("--out", required=True, help="Output video path (.mp4)")
    parser.add_argument("--vae_pth", default="checkpoints/vae/vae.pth")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier. 0.5=half speed, 2=double speed.")
    parser.add_argument("--reverse", action="store_true", help="Reverse the video")
    parser.add_argument("--loop", action="store_true", help="Ping-pong loop (forward then reverse)")
    parser.add_argument("--frames", type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="Decode only frames START..END")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading latents from {args.latent}...")
    data = torch.load(args.latent, map_location="cpu", weights_only=False)
    latents = data["history_latents"]
    start_index = int(data["start_index"])
    frames_per_latent = int(data["frames_per_latent"])
    vae_stats = data["vae_stats"]
    T_lat = latents.shape[2]
    T_video = (T_lat - start_index) * frames_per_latent
    print(f"  Latents: {list(latents.shape)}, start_index={start_index}, "
          f"frames_per_latent={frames_per_latent} → {T_video} video frames")

    vae = load_vae(args.vae_pth, device=args.device, dtype=torch.bfloat16)
    vae_core = vae.model

    print("Decoding...")
    video = decode_latents(latents, start_index, vae_stats, vae_core,
                           device=args.device, dtype=torch.bfloat16)
    print(f"Decoded: {list(video.shape)}")

    video = apply_playback_options(video, speed=args.speed, reverse=args.reverse,
                                   loop=args.loop, frame_range=args.frames)

    save_video(video, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
