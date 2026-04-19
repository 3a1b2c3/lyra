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
  Q / Esc        Quit

With --depth:
  A / D          Rotate camera left / right (yaw)
  W / S          Tilt camera up / down (pitch)
  C              Reset camera rotation

Usage:
  python scripts/explore_latents.py --latent results_skyfall_004/videos/00/latents/zoom_in.pt
  python scripts/explore_latents.py --latent results_skyfall_004/videos/00/latents/zoom_in.pt --also results_skyfall_004/videos/00/latents/zoom_out.pt
  python scripts/explore_latents.py --latent results_skyfall_004/videos/00/latents/zoom_in.pt --depth assets/skyfall_input_004/00_depth.npz
  python scripts/explore_latents.py --latent results_skyfall_004/videos/00/latents/zoom_in.pt --no-cache
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
    print(f"Cached decoded frames -> {p}")


@torch.no_grad()
def decode_to_numpy(latent_path, vae_core, device, dtype, vae_stats_override=None):
    """Decode latent .pt -> numpy uint8 (T, H, W, 3) array in memory."""
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
    print(f"  -> {video_np.shape[0]} frames, {video_np.shape[2]}x{video_np.shape[1]}")
    return video_np


# ── Depth warp ─────────────────────────────────────────────────────────────────

def load_depth(depth_npz, vid_H, vid_W):
    """Load depth npz and scale intrinsics to match video resolution."""
    d = np.load(depth_npz)
    depth = d["depth_hw"].astype(np.float32)
    K = d["K_33"].astype(np.float64)
    H_d, W_d = depth.shape
    depth_resized = cv2.resize(depth, (vid_W, vid_H), interpolation=cv2.INTER_LINEAR)
    K_vid = K.copy()
    K_vid[0, 0] *= vid_W / W_d   # fx
    K_vid[0, 2] *= vid_W / W_d   # cx
    K_vid[1, 1] *= vid_H / H_d   # fy
    K_vid[1, 2] *= vid_H / H_d   # cy
    print(f"Depth loaded: {W_d}x{H_d} -> {vid_W}x{vid_H}, depth range [{depth.min():.1f}, {depth.max():.1f}]")
    return depth_resized, K_vid


def make_warp_maps(depth, K, yaw, pitch):
    """Compute backward-warp source maps for cv2.remap given yaw/pitch rotation (radians)."""
    H, W = depth.shape
    cy_r, sy_r = np.cos(yaw), np.sin(yaw)
    cp_r, sp_r = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy_r, 0, sy_r], [0, 1, 0], [-sy_r, 0, cy_r]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp_r, -sp_r], [0, sp_r, cp_r]], dtype=np.float64)
    R = Rx @ Ry
    Kinv = np.linalg.inv(K)
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.stack([xs.ravel().astype(np.float64),
                    ys.ravel().astype(np.float64),
                    np.ones(H * W, dtype=np.float64)], axis=0)  # (3, N)
    # Unproject output pixels to 3D using source depth as proxy
    rays = Kinv @ pts                             # (3, N)
    pts3d = rays * depth.ravel().astype(np.float64)  # (3, N)
    # Backward: rotate source to find where output pixel came from
    pts3d_src = R.T @ pts3d                       # (3, N)
    proj = K @ pts3d_src                          # (3, N)
    u_src = (proj[0] / proj[2]).reshape(H, W).astype(np.float32)
    v_src = (proj[1] / proj[2]).reshape(H, W).astype(np.float32)
    return u_src, v_src


# ── HUD ────────────────────────────────────────────────────────────────────────

def draw_hud(frame, idx, total, playing, speed, reverse, loop, label="", last_key=255):
    h, w = frame.shape[:2]
    out = frame.copy()

    bar_y, bar_h = h - 18, 6
    cv2.rectangle(out, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)
    fill = int(w * idx / max(total - 1, 1))
    cv2.rectangle(out, (0, bar_y), (fill, bar_y + bar_h), (0, 200, 100), -1)

    icon  = ">" if playing else "||"
    flags = ("R" if reverse else "") + ("L" if loop else "")
    text  = f"{icon} {idx+1}/{total}  {speed:.2f}x  {flags}  {label}"
    cv2.putText(out, text, (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    key_char = chr(last_key) if 32 <= last_key < 127 else f"#{last_key}"
    key_text = f"key:{key_char}"
    cv2.putText(out, key_text, (w - 80, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, key_text, (w - 80, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1, cv2.LINE_AA)
    return out


# ── Mouse scrub ────────────────────────────────────────────────────────────────

def make_mouse_cb(state):
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

def run_explorer(all_videos, labels, fps=16, save_dir=".", depth=None, K=None):
    """all_videos: list of (T, H, W, 3) uint8 numpy arrays."""
    T = max(v.shape[0] for v in all_videos)
    vid_H, vid_W = all_videos[0].shape[1:3]

    # Pad shorter videos to same length
    padded = []
    for v in all_videos:
        if v.shape[0] < T:
            pad = np.zeros((T - v.shape[0], *v.shape[1:]), dtype=np.uint8)
            v = np.concatenate([v, pad], axis=0)
        padded.append(v)

    label = " | ".join(labels)
    W_disp = vid_W * len(padded)
    H_disp = vid_H

    win = "Lyra-2 Latent Explorer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scale = min(1440 / W_disp, 900 / H_disp, 2.0)
    cv2.resizeWindow(win, int(W_disp * scale), int(H_disp * scale))

    mouse_state = {"dragging": False, "x": 0}
    cv2.setMouseCallback(win, make_mouse_cb(mouse_state))

    has_depth = depth is not None and K is not None
    idx = 0
    playing = True
    speed = 1.0
    reverse = False
    loop = False
    yaw = 0.0
    pitch = 0.0
    last_key = 255
    step_rad = np.radians(3.0)
    warp_cache = {"yaw": None, "pitch": None, "map_x": None, "map_y": None}
    frame_delay = 1.0 / fps
    last_time = time.time()

    print(f"\nExplorer ready -- {T} frames  ({W_disp}x{H_disp})")
    if has_depth:
        print("Space=play/pause  </>=step  ^/v=speed  A/D=yaw  Z/X=pitch  C=reset view  R=reverse  L=loop  S=save  Q=quit")
    else:
        print("Space=play/pause  </>=step  ^/v=speed  R=reverse  L=loop  1-9=jump  S=save  Q=quit")
        print("Tip: pass --depth assets/skyfall_input_004/00_depth.npz to enable camera rotation")

    while True:
        # Mouse scrub
        if mouse_state["dragging"]:
            idx = int(mouse_state["x"] / max(W_disp - 1, 1) * (T - 1))
            idx = max(0, min(T - 1, idx))

        # Recompute warp maps only when angle changes
        if has_depth and (yaw != warp_cache["yaw"] or pitch != warp_cache["pitch"]):
            warp_cache["map_x"], warp_cache["map_y"] = make_warp_maps(depth, K, yaw, pitch)
            warp_cache["yaw"] = yaw
            warp_cache["pitch"] = pitch

        # Warp each panel independently, then concatenate
        panels = []
        for v in padded:
            frame_rgb = v[idx]
            if has_depth and (yaw != 0.0 or pitch != 0.0):
                frame_rgb = cv2.remap(frame_rgb, warp_cache["map_x"], warp_cache["map_y"],
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            panels.append(frame_rgb)
        frame_rgb = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        angle_str = f"  yaw={np.degrees(yaw):.0f} pit={np.degrees(pitch):.0f}" if has_depth else ""
        frame_bgr = draw_hud(frame_bgr, idx, T, playing, speed, reverse, loop, label + angle_str, last_key)
        cv2.imshow(win, frame_bgr)

        raw = cv2.waitKey(1)
        key = raw & 0xFF
        if raw != -1:
            last_key = key

        if key in (ord('q'), 27):
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
        elif key == 0:
            idx = 0
        elif key == 255:
            idx = T - 1
        elif has_depth:
            if key == ord('a'):
                yaw -= step_rad
            elif key == ord('d'):
                yaw += step_rad
            elif key == ord('z'):
                pitch -= step_rad
            elif key == ord('x'):
                pitch += step_rad
            elif key == ord('c'):
                yaw = 0.0
                pitch = 0.0

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
    parser.add_argument("--depth", default=None, help="Path to _depth.npz for camera rotation (e.g. assets/skyfall_input_004/00_depth.npz)")
    parser.add_argument("--vae_pth", default="checkpoints/vae/vae.pth")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--save_dir", default=".", help="Directory for saved frames (S key)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="Disable decoded frame cache")
    parser.set_defaults(cache=True)
    args = parser.parse_args()

    dtype = torch.bfloat16
    all_paths = [args.latent] + (args.also or [])

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

    depth, K = None, None
    if args.depth:
        vid_H, vid_W = all_videos[0].shape[1:3]
        depth, K = load_depth(args.depth, vid_H, vid_W)

    run_explorer(all_videos, labels, fps=args.fps, save_dir=args.save_dir, depth=depth, K=K)


if __name__ == "__main__":
    main()
