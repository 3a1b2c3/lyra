"""Project Skyfall-GS point clouds into reference camera frames to produce
dense depth maps, bypassing DA3 single-image depth estimation.

Reads  assets/skyfall/datasets_NYC/<scene>/points3D.ply  and the matching
transforms.json, projects the SfM point cloud into the reference camera, and
densifies the resulting sparse depth map via nearest-neighbour interpolation.

Outputs alongside prepare_skyfall.py results:
  {idx:02d}_depth.npz  --  keys: depth_hw (H,W float32), K_33 (3,3 float32),
                                  mask_hw (H,W float32, 1=valid)

Pass --depth_dir to lyra2_zoomgs_inference.py to load these instead of DA3.

Usage:
  python scripts/prepare_skyfall_depth.py
  python scripts/prepare_skyfall_depth.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input --frame_idx 0
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.interpolate import NearestNDInterpolator

SKYFALL_DIR = "assets/skyfall/datasets_NYC"
OUT_DIR = "assets/skyfall_input"


def read_ply_xyz(ply_path: str) -> np.ndarray:
    """Read XYZ from a binary PLY with x,y,z,nx,ny,nz,r,g,b layout."""
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


def project(xyz: np.ndarray, c2w: np.ndarray, fl_x, fl_y, cx, cy, W, H):
    """Project world points into image plane. Returns (u, v, Z) for all points."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (xyz - t) @ R  # world -> camera (R is orthonormal, so R^-1 = R^T but applied as matmul)
    # NeRF c2w: cam_pt = R^T (world_pt - t)
    cam = (xyz - t) @ R
    Z = cam[:, 2]
    u = cam[:, 0] / Z * fl_x + cx
    v = cam[:, 1] / Z * fl_y + cy
    return u, v, Z


def densify(u_vis, v_vis, Z_vis, H, W) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour fill from sparse projected depth to full (H,W) depth map."""
    sparse = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    ui = np.round(u_vis).astype(int).clip(0, W - 1)
    vi = np.round(v_vis).astype(int).clip(0, H - 1)
    # Keep nearest point per pixel (sort far→near so near overwrites)
    order = np.argsort(-Z_vis)
    sparse[vi[order], ui[order]] = Z_vis[order].astype(np.float32)
    mask[vi[order], ui[order]] = True

    # Nearest-neighbour interpolation over the whole image grid
    ys, xs = np.mgrid[0:H, 0:W]
    interp = NearestNDInterpolator(list(zip(vi[order], ui[order])), Z_vis[order].astype(np.float32))
    dense = interp(ys, xs).astype(np.float32)

    return dense, mask.astype(np.float32)


def prepare_scene_depth(scene_dir: str, out_dir: str, idx: int, frame_idx: int = 0,
                        crop_aspect: tuple[float, float] | None = None):
    ply_path = os.path.join(scene_dir, "points3D.ply")
    if not os.path.exists(ply_path):
        print(f"  [{idx:02d}] No points3D.ply - skipping")
        return

    for name in ("transforms.json", "transforms_train.json"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                meta = json.load(f)
            break
    else:
        raise FileNotFoundError(f"No transforms.json in {scene_dir}")

    frames = meta["frames"]
    frame = frames[min(frame_idx, len(frames) - 1)]
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    W, H = int(meta["w"]), int(meta["h"])

    xyz = read_ply_xyz(ply_path)
    u, v, Z = project(xyz, c2w, fl_x, fl_y, cx, cy, W, H)

    vis = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    print(f"  [{idx:02d}] {os.path.basename(scene_dir)}: "
          f"{vis.sum()}/{len(xyz)} pts visible, "
          f"depth {Z[vis].min():.1f}-{Z[vis].max():.1f}")

    depth_dense, mask_hw = densify(u[vis], v[vis], Z[vis], H, W)

    # Center-crop depth map and adjust K to match cropped image
    if crop_aspect is not None:
        crop_w = W
        crop_h = round(W * crop_aspect[1] / crop_aspect[0])
        if crop_h > H:
            crop_h = H
            crop_w = round(H * crop_aspect[0] / crop_aspect[1])
        left = (W - crop_w) // 2
        top  = (H - crop_h) // 2
        depth_dense = depth_dense[top:top + crop_h, left:left + crop_w]
        mask_hw     = mask_hw[top:top + crop_h, left:left + crop_w]
        cx -= left
        cy -= top
        W, H = crop_w, crop_h

    K_33 = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)

    tag = f"{idx:02d}"
    out_path = os.path.join(out_dir, f"{tag}_depth.npz")
    np.savez(out_path, depth_hw=depth_dense, K_33=K_33, mask_hw=mask_hw)
    print(f"         -> {out_path}  ({H}x{W})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyfall_dir", default=SKYFALL_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--scenes", nargs="*", help="Specific scene names; default: all")
    parser.add_argument("--crop_aspect", type=str, default="16:9",
                        help="Center-crop to W:H aspect ratio after projection. Pass 'none' to disable.")
    args = parser.parse_args()

    crop_aspect = None
    if args.crop_aspect.lower() != "none":
        aw, ah = args.crop_aspect.split(":")
        crop_aspect = (float(aw), float(ah))

    scenes = sorted(
        d for d in os.listdir(args.skyfall_dir)
        if os.path.isdir(os.path.join(args.skyfall_dir, d))
    )
    if args.scenes:
        scenes = [s for s in scenes if s in args.scenes]

    print(f"Projecting point clouds for {len(scenes)} scene(s) -> {args.out_dir}")
    for i, scene in enumerate(scenes):
        prepare_scene_depth(
            scene_dir=os.path.join(args.skyfall_dir, scene),
            out_dir=args.out_dir,
            idx=i,
            frame_idx=args.frame_idx,
            crop_aspect=crop_aspect,
        )
    print("Done.")


if __name__ == "__main__":
    main()
