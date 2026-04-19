"""Build a multi-frame spatial cache from Skyfall-GS scenes for Lyra-2.

For each scene, selects N evenly-spaced keyframes, projects points3D.ply into
each camera, and saves the result as a multiframe_cache npz.

Output per scene:  {out_dir}/{scene}_multiframe.npz
  video       (T, H, W, 3)  uint8 RGB
  depth_hw    (T, H, W)     float32
  camera_w2c  (T, 4, 4)     float32
  intrinsics  (T, 3, 3)     float32
  frame_ids   (T,)          int32   original frame indices

This npz is consumed by lyra2_zoomgs_inference.py via --multiframe_cache_dir.

Usage:
  python scripts/prepare_skyfall_multiframe.py
  python scripts/prepare_skyfall_multiframe.py --num_frames 8 --target_hw 320 576
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

SKYFALL_DIR = "assets/skyfall/datasets_NYC"
OUT_DIR = "assets/skyfall_input"


def read_ply_xyz(ply_path: str) -> np.ndarray:
    with open(ply_path, "rb") as f:
        while f.readline().strip() != b"end_header":
            pass
        dt = np.dtype([
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ])
        data = np.frombuffer(f.read(), dtype=dt)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    R, t = c2w[:3, :3], c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = (-R.T @ t).astype(np.float32)
    return w2c


def project_depth(xyz: np.ndarray, c2w: np.ndarray, fl_x, fl_y, cx, cy, W, H) -> np.ndarray:
    """Project PLY into camera and return dense depth map (H, W) float32."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (xyz - t) @ R
    Z = cam[:, 2]
    u = cam[:, 0] / Z * fl_x + cx
    v = cam[:, 1] / Z * fl_y + cy
    vis = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not vis.any():
        return np.zeros((H, W), dtype=np.float32)

    u_v, v_v, Z_v = u[vis], v[vis], Z[vis]
    order = np.argsort(-Z_v)
    ui = np.round(u_v[order]).astype(int).clip(0, W - 1)
    vi = np.round(v_v[order]).astype(int).clip(0, H - 1)
    Z_s = Z_v[order].astype(np.float32)

    interp = NearestNDInterpolator(list(zip(vi, ui)), Z_s)
    ys, xs = np.mgrid[0:H, 0:W]
    return interp(ys, xs).astype(np.float32)


def load_image(scene_dir: str, file_path: str) -> np.ndarray:
    """Load image from scene, trying common extensions. Returns RGB uint8 (H,W,3)."""
    src = os.path.join(scene_dir, file_path)
    if not os.path.exists(src):
        base = os.path.splitext(src)[0]
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            if os.path.exists(base + ext):
                src = base + ext
                break
    bgr = cv2.imread(src)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {src}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prepare_scene_multiframe(scene_dir: str, out_dir: str, num_frames: int,
                              target_h: int, target_w: int, idx: int = 0,
                              ref_frame_idx: int = 0):
    scene_name = os.path.basename(scene_dir)
    ply_path = os.path.join(scene_dir, "points3D.ply")
    if not os.path.exists(ply_path):
        print(f"  {scene_name}: no points3D.ply — skipping")
        return

    for name in ("transforms.json", "transforms_train.json"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                meta = json.load(f)
            break
    else:
        raise FileNotFoundError(f"No transforms.json in {scene_dir}")

    all_frames = meta["frames"]
    N_avail = len(all_frames)
    # Pick frames adjacent to the reference frame so their viewpoints overlap
    # with the zoom trajectory. Start from ref+1 going forward; wrap if needed.
    start = ref_frame_idx + 1
    indices = [min(start + i, N_avail - 1) for i in range(num_frames)]
    selected = [all_frames[i] for i in indices]

    fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    src_W, src_H = int(meta["w"]), int(meta["h"])

    # Scale intrinsics to target resolution
    sx = target_w / src_W
    sy = target_h / src_H
    K = np.array([
        [fl_x * sx, 0,         cx * sx],
        [0,         fl_y * sy, cy * sy],
        [0,         0,         1.0],
    ], dtype=np.float32)

    xyz = read_ply_xyz(ply_path)
    print(f"  {scene_name}: {len(selected)} frames, {len(xyz)} PLY points")

    videos, depths, w2cs, intrinsics, frame_ids = [], [], [], [], []

    for frame_idx, frame in zip(indices, selected):
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = c2w_to_w2c(c2w)

        # Image
        rgb = load_image(scene_dir, frame["file_path"])
        rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Depth at source resolution, then resize
        depth_src = project_depth(xyz, c2w, fl_x, fl_y, cx, cy, src_W, src_H)
        depth_resized = cv2.resize(depth_src, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        videos.append(rgb_resized)
        depths.append(depth_resized)
        w2cs.append(w2c)
        intrinsics.append(K)
        frame_ids.append(frame_idx)

    out_path = os.path.join(out_dir, f"{idx:02d}_multiframe.npz")
    np.savez(
        out_path,
        video=np.stack(videos).astype(np.uint8),        # (T, H, W, 3)
        depth_hw=np.stack(depths).astype(np.float32),   # (T, H, W)
        camera_w2c=np.stack(w2cs).astype(np.float32),   # (T, 4, 4)
        intrinsics=np.stack(intrinsics).astype(np.float32),  # (T, 3, 3)
        frame_ids=np.array(frame_ids, dtype=np.int32),
    )
    print(f"         -> {out_path}  ({len(selected)} frames @ {target_h}x{target_w})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyfall_dir", default=SKYFALL_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Number of evenly-spaced keyframes per scene (default: 4)")
    parser.add_argument("--target_hw", type=int, nargs=2, default=[320, 576],
                        metavar=("H", "W"),
                        help="Target resolution matching inference --resolution (default: 320 576)")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Reference frame index (default: 0); multiframe picks the next N frames")
    parser.add_argument("--scenes", nargs="*", help="Specific scene names; default: all")
    args = parser.parse_args()

    scenes = sorted(
        d for d in os.listdir(args.skyfall_dir)
        if os.path.isdir(os.path.join(args.skyfall_dir, d))
    )
    if args.scenes:
        scenes = [s for s in scenes if s in args.scenes]

    target_h, target_w = args.target_hw
    print(f"Building multiframe caches for {len(scenes)} scene(s), "
          f"{args.num_frames} frames each @ {target_h}x{target_w}")

    for i, scene in enumerate(scenes):
        prepare_scene_multiframe(
            scene_dir=os.path.join(args.skyfall_dir, scene),
            out_dir=args.out_dir,
            num_frames=args.num_frames,
            target_h=target_h,
            target_w=target_w,
            idx=i,
            ref_frame_idx=args.frame_idx,
        )
    print("Done.")


if __name__ == "__main__":
    main()
