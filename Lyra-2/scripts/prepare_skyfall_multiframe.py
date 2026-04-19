"""Build a multi-frame spatial cache from Skyfall-GS scenes for Lyra-2.

For each scene, selects N frames adjacent to the reference frame, loads their
images, and reuses the pre-computed reference depth (from prepare_skyfall_depth.py)
as a proxy depth for all keyframes.  No point-cloud projection is performed.

Output per scene:  {out_dir}/{idx:02d}_multiframe.npz
  video       (T, H, W, 3)  uint8 RGB
  depth_hw    (T, H, W)     float32  (reference depth reused for all frames)
  camera_w2c  (T, 4, 4)     float32  relative to reference camera
  intrinsics  (T, 3, 3)     float32
  frame_ids   (T,)          int32    original frame indices

Usage:
  python scripts/prepare_skyfall_multiframe.py
  python scripts/prepare_skyfall_multiframe.py --num_frames 8 --target_hw 320 576
"""
import argparse
import json
import os

import cv2
import numpy as np

SKYFALL_DIR = "assets/skyfall/datasets_NYC"
OUT_DIR = "assets/skyfall_input"


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    R, t = c2w[:3, :3], c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = (-R.T @ t).astype(np.float32)
    return w2c


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

    # Pick N frames immediately after the reference frame
    start = ref_frame_idx + 1
    indices = [min(start + i, N_avail - 1) for i in range(num_frames)]
    selected = [all_frames[i] for i in indices]

    fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    src_W, src_H = int(meta["w"]), int(meta["h"])

    sx = target_w / src_W
    sy = target_h / src_H
    K = np.array([
        [fl_x * sx, 0,         cx * sx],
        [0,         fl_y * sy, cy * sy],
        [0,         0,         1.0],
    ], dtype=np.float32)

    # Reference c2w: keyframe poses are stored relative to reference camera
    c2w_ref = np.array(all_frames[ref_frame_idx]["transform_matrix"], dtype=np.float64)

    # Load pre-computed reference depth (from prepare_skyfall_depth.py)
    ref_depth_path = os.path.join(out_dir, f"{idx:02d}_depth.npz")
    if os.path.exists(ref_depth_path):
        ref_npz = np.load(ref_depth_path)
        ref_depth = cv2.resize(ref_npz["depth_hw"].astype(np.float32),
                               (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        print(f"  [{idx:02d}] {scene_name}: loaded reference depth from {ref_depth_path}")
    else:
        ref_depth = np.ones((target_h, target_w), dtype=np.float32)
        print(f"  [{idx:02d}] {scene_name}: no reference depth npz found, using constant depth=1")

    print(f"  [{idx:02d}] {scene_name}: {N_avail} frames available, "
          f"selecting {len(selected)} (ref={ref_frame_idx}, next {num_frames}):")

    videos, depths, w2cs, intrinsics, frame_ids = [], [], [], [], []

    for frame_idx, frame in zip(indices, selected):
        file_path = frame["file_path"]
        print(f"         frame {frame_idx:3d}  {file_path}")

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c_world = c2w_to_w2c(c2w)
        # Express pose relative to reference camera (Lyra-2 uses ref as world origin)
        w2c_rel = (w2c_world @ c2w_ref).astype(np.float32)

        rgb = load_image(scene_dir, file_path)
        rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        videos.append(rgb_resized)
        depths.append(ref_depth)
        w2cs.append(w2c_rel)
        intrinsics.append(K)
        frame_ids.append(frame_idx)

    out_path = os.path.join(out_dir, f"{idx:02d}_multiframe.npz")
    np.savez(
        out_path,
        video=np.stack(videos).astype(np.uint8),
        depth_hw=np.stack(depths).astype(np.float32),
        camera_w2c=np.stack(w2cs).astype(np.float32),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        frame_ids=np.array(frame_ids, dtype=np.int32),
    )
    print(f"         -> {out_path}  ({len(selected)} frames @ {target_h}x{target_w})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyfall_dir", default=SKYFALL_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--num_frames", type=int, default=6,
                        help="Number of frames adjacent to reference per scene (default: 6)")
    parser.add_argument("--target_hw", type=int, nargs=2, default=[320, 576],
                        metavar=("H", "W"))
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Reference frame index (default: 0)")
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
