"""Interactive latent explorer for Lyra-2 zoom videos.

Decodes latents once (~30 sec), then navigate in real-time with keyboard and mouse.

Controls:
  Space          Play / Pause
  Left / Right   Step one frame
  Up / Down      Speed x2 / /2
  R              Reverse direction
  L              Toggle ping-pong loop
  Home / End     Jump to first / last frame
  1-9            Jump to 10%-90% of video
  S              Save current frame as PNG
  Mouse drag     Scrub timeline (left-click + drag horizontally)
  Scroll wheel   Step frames forward/backward
  Q / Esc        Quit

Usage:
  python scripts/explore_latents.py --latent results_skyfall_336/videos/00/latents/zoom_in.pt
  python scripts/explore_latents.py --latent results_skyfall_336/videos/00/latents/zoom_in.pt --also results_skyfall_336/videos/00/latents/zoom_out.pt
  python scripts/explore_latents.py --latent results_skyfall_336/videos/00/latents/zoom_in.pt --no-cache  # force re-decode, skip cache
"""
import argparse
import os
import sys
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Decode ─────────────────────────────────────────────────────────────────────

def load_vae(vae_pth, device, dtype):
    from lyra_2._src.tokenizers.wan2pt1 import WanVAE
    print(f"Loading VAE from {vae_pth} ...")
    vae = WanVAE(vae_pth=vae_pth, device=device, dtype=dtype, is_amp=False)
    print("VAE loaded.")
    return vae


def _cache_path(latent_path):
    return os.path.splitext(latent_path)[0] + "_decoded.npy"


def load_cached(latent_path):
    p = _cache_path(latent_path)
    if os.path.exists(p):
        print(f"Loading cached frames from {p} ...")
        return np.load(p)
    return None


def save_cached(latent_path, video_np):
    p = _cache_path(latent_path)
    np.save(p, video_np)
    print(f"Cached decoded frames → {p}")


@torch.no_grad()
def decode_to_numpy(latent_path, vae_core, device, dtype, vae_stats_override=None):
    """Decode latent .pt → numpy uint8 (T, H, W, 3) array in memory."""
    data = torch.load(latent_path, map_location="cpu", weights_only=False)
    latents = data["history_latents"]
    start_index = int(data["start_index"])
    vae_stats = vae_stats_override or data["vae_stats"]

    img_mean   = vae_stats["img_mean"].to(device)
    img_std    = vae_stats["img_std"].to(device)
    video_mean = vae_stats["video_mean"].to(device)
    video_std  = vae_stats["video_std"].to(device)
    scale_mean    = vae_stats["scale_mean"]
    scale_inv_std = vae_stats["scale_inv_std"]
    if torch.is_tensor(scale_mean):
        scale_mean = scale_mean.to(device)
    if torch.is_tensor(scale_inv_std):
        scale_inv_std = scale_inv_std.to(device)
    z_dim = int(vae_stats["z_dim"])

    ctx = torch.amp.autocast("cuda", dtype=dtype) if torch.cuda.is_available() else nullcontext()
    vae_core.clear_cache()
    dec_feat_cache = [None] * vae_core._conv_num
    frames = []
    T_lat = latents.shape[2]

    print(f"Decoding {latent_path} ({T_lat} latent frames) ...")
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
            out = vae_core.decoder(x, feat_cache=dec_feat_cache, feat_idx=[0])
        pixel = out[0].float().cpu()  # (C, frames_per_lat, H, W)
        frames.append(pixel)

    video = torch.cat(frames, dim=1)[:, start_index:]  # (C, T, H, W)
    video_np = ((video.clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)
    video_np = video_np.permute(1, 2, 3, 0).numpy()    # (T, H, W, C) RGB
    print(f"  → {video_np.shape[0]} frames, {video_np.shape[2]}×{video_np.shape[1]}")
    return video_np


# ── HUD ────────────────────────────────────────────────────────────────────────

def draw_hud(frame, idx, total, playing, speed, reverse, loop, label=""):
    h, w = frame.shape[:2]
    out = frame.copy()

    # Timeline bar
    bar_y, bar_h = h - 18, 6
    cv2.rectangle(out, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)
    fill = int(w * idx / max(total - 1, 1))
    cv2.rectangle(out, (0, bar_y), (fill, bar_y + bar_h), (0, 200, 100), -1)

    # Status text
    icon  = ">" if playing else "||"
    flags = ("R" if reverse else "") + ("L" if loop else "")
    text  = f"{icon} {idx+1}/{total}  {speed:.2f}x  {flags}  {label}"
    cv2.putText(out, text, (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── Mouse scrub ────────────────────────────────────────────────────────────────

_mouse_state = {"dragging": False, "x": 0}

def make_mouse_cb(state, frames_ref):
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["x"] = x
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["x"] = x
    return cb


# ── Main explorer ──────────────────────────────────────────────────────────────

def run_explorer(all_videos, labels, fps=16, save_dir="."):
    """all_videos: list of (T, H, W, 3) uint8 numpy arrays."""
    # Concatenate multiple videos side by side if needed
    if len(all_videos) > 1:
        max_t = max(v.shape[0] for v in all_videos)
        padded = []
        for v in all_videos:
            if v.shape[0] < max_t:
                pad = np.zeros((max_t - v.shape[0], *v.shape[1:]), dtype=np.uint8)
                v = np.concatenate([v, pad], axis=0)
            padded.append(v)
        frames = np.concatenate(padded, axis=2)  # side by side
        label = " | ".join(labels)
    else:
        frames = all_videos[0]
        label = labels[0]

    T, H, W, _ = frames.shape
    win = "Lyra-2 Latent Explorer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scale = min(1440 / W, 900 / H, 2.0)
    cv2.resizeWindow(win, int(W * scale), int(H * scale))

    mouse_state = {"dragging": False, "x": 0}
    cv2.setMouseCallback(win, make_mouse_cb(mouse_state, frames))

    idx = 0
    playing = True
    speed = 1.0
    reverse = False
    loop = False
    frame_delay = 1.0 / fps
    last_time = time.time()

    print(f"\nExplorer ready — {T} frames  ({W}×{H})")
    print("Space=play/pause  ←/→=step  ↑/↓=speed  R=reverse  L=loop  1-9=jump  S=save  Q=quit")

    while True:
        # Mouse scrub
        if mouse_state["dragging"]:
            idx = int(mouse_state["x"] / max(W - 1, 1) * (T - 1))
            idx = max(0, min(T - 1, idx))

        # Draw
        frame_rgb = frames[idx]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = draw_hud(frame_bgr, idx, T, playing, speed, reverse, loop, label)
        cv2.imshow(win, frame_bgr)

        # Keyboard (1ms poll)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # Q or Esc
            break
        elif key == ord(' '):
            playing = not playing
        elif key == 81 or key == 2424832 % 256:  # Left arrow
            playing = False
            idx = max(0, idx - 1)
        elif key == 83 or key == 2555904 % 256:  # Right arrow
            playing = False
            idx = min(T - 1, idx + 1)
        elif key == 82 or key == 2490368 % 256:  # Up arrow
            speed = min(speed * 2, 16.0)
        elif key == 84 or key == 2621440 % 256:  # Down arrow
            speed = max(speed / 2, 0.0625)
        elif key == ord('r'):
            reverse = not reverse
        elif key == ord('l'):
            loop = not loop
        elif key in (ord('s'), ord('S')):
            out_path = os.path.join(save_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(out_path, frame_bgr)
            print(f"Saved {out_path}")
        elif ord('1') <= key <= ord('9'):
            idx = int((key - ord('1')) / 8 * (T - 1))
        elif key == 0:  # Home (some terminals)
            idx = 0
        elif key == 255:  # End
            idx = T - 1

        # Scroll wheel (OpenCV encodes as flags in some builds)

        # Auto-advance when playing
        if playing and not mouse_state["dragging"]:
            now = time.time()
            if now - last_time >= frame_delay / speed:
                last_time = now
                step = -1 if reverse else 1
                idx += step
                if idx >= T:
                    idx = T - 1 - (idx - T) if loop else T - 1
                    if not loop:
                        playing = False
                elif idx < 0:
                    idx = 1 - idx if loop else 0
                    if not loop:
                        playing = False

    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", required=True, help="Primary latent .pt file")
    parser.add_argument("--also", nargs="*", default=[], help="Additional latent .pt files (shown side-by-side)")
    parser.add_argument("--vae_pth", default="checkpoints/vae/vae.pth")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--save_dir", default=".", help="Directory for saved frames (S key)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="Disable decoded frame cache")
    parser.set_defaults(cache=True)
    args = parser.parse_args()

    dtype = torch.bfloat16
    all_paths = [args.latent] + (args.also or [])

    # Check which paths still need decoding
    need_decode = [p for p in all_paths if not (args.cache and load_cached(p) is not None)]
    vae_core = None
    if need_decode:
        vae = load_vae(args.vae_pth, args.device, dtype)
        vae_core = vae.model

    all_videos, labels = [], []
    for path in all_paths:
        if args.cache:
            arr = load_cached(path)
            if arr is not None:
                all_videos.append(arr)
                labels.append(os.path.basename(path).replace(".pt", ""))
                continue
        arr = decode_to_numpy(path, vae_core, args.device, dtype)
        if args.cache:
            save_cached(path, arr)
        all_videos.append(arr)
        labels.append(os.path.basename(path).replace(".pt", ""))

    run_explorer(all_videos, labels, fps=args.fps, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
