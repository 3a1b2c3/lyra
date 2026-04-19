"""Visualize Skyfall scene: PLY point cloud + training cameras + zoom trajectory.

Usage:
  python scripts/visualize_scene.py --scene assets/skyfall/datasets_NYC/NYC_336
  python scripts/visualize_scene.py --scene assets/skyfall/datasets_NYC/NYC_336 --ref_frame 0 --zoom_strength 0.15
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def load_point_cloud(scene_dir, max_points=30000):
    ply_path = os.path.join(scene_dir, "points3D.ply")
    pc = trimesh.load(ply_path)
    pts = np.array(pc.vertices)
    colors = np.array(pc.colors)[:, :3] / 255.0 if pc.colors is not None else None
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]
    return pts, colors


def load_cameras(scene_dir):
    for name in ("transforms_train.json", "transforms.json"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                meta = json.load(f)
            break
    else:
        raise FileNotFoundError(f"No transforms.json in {scene_dir}")

    c2ws = []
    for frame in meta["frames"]:
        c2ws.append(np.array(frame["transform_matrix"], dtype=np.float64))
    return np.array(c2ws), meta


def draw_frustum(ax, c2w, size=0.3, color="blue", alpha=0.6):
    """Draw a simple camera frustum as 4 lines from optical centre to corners."""
    origin = c2w[:3, 3]
    right = c2w[:3, 0] * size
    up = -c2w[:3, 1] * size  # flip y (OpenCV convention)
    fwd = c2w[:3, 2] * size

    corners = [
        origin + fwd + right + up,
        origin + fwd - right + up,
        origin + fwd - right - up,
        origin + fwd + right - up,
    ]
    # Lines from origin to each corner
    segs = [[origin, c] for c in corners]
    # Frame rectangle
    segs += [[corners[i], corners[(i+1) % 4]] for i in range(4)]
    lc = Line3DCollection(segs, colors=color, linewidths=0.8, alpha=alpha)
    ax.add_collection3d(lc)
    return origin


def zoom_trajectory(c2w_ref, n_frames, strength, direction="forward"):
    """Simple linear zoom: move along -z axis of reference camera."""
    positions = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1) * strength
        if direction == "forward":
            offset = -c2w_ref[:3, 2] * t  # move forward
        else:
            offset = c2w_ref[:3, 2] * t   # move backward
        pos = c2w_ref[:3, 3] + offset
        positions.append(pos)
    return np.array(positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="assets/skyfall/datasets_NYC/NYC_336")
    parser.add_argument("--ref_frame", type=int, default=0)
    parser.add_argument("--zoom_strength", type=float, default=0.15,
                        help="Zoom-in strength (same as --zoom_in_strength)")
    parser.add_argument("--zoom_frames", type=int, default=81)
    parser.add_argument("--max_cams", type=int, default=50,
                        help="Max training cameras to draw frustums for")
    parser.add_argument("--max_points", type=int, default=30000)
    args = parser.parse_args()

    print(f"Loading point cloud...")
    pts, colors = load_point_cloud(args.scene, args.max_points)
    print(f"  {len(pts)} points")

    print(f"Loading cameras...")
    c2ws, meta = load_cameras(args.scene)
    print(f"  {len(c2ws)} training cameras")

    c2w_ref = c2ws[args.ref_frame]

    # Subsample training cameras for display
    step = max(1, len(c2ws) // args.max_cams)
    cam_subset = c2ws[::step]

    # Zoom trajectories
    zoom_in_pos = zoom_trajectory(c2w_ref, args.zoom_frames, strength=args.zoom_strength, direction="forward")
    zoom_out_pos = zoom_trajectory(c2w_ref, args.zoom_frames, strength=args.zoom_strength * 3, direction="backward")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Point cloud
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors if colors is not None else "gray",
               s=0.5, alpha=0.4, linewidths=0)

    # Training cameras
    cam_origins = []
    for i, c2w in enumerate(cam_subset):
        color = "steelblue" if i > 0 else "red"
        alpha = 0.3 if i > 0 else 1.0
        o = draw_frustum(ax, c2w, size=0.15, color=color, alpha=alpha)
        cam_origins.append(o)
    cam_origins = np.array(cam_origins)
    ax.plot(cam_origins[:, 0], cam_origins[:, 1], cam_origins[:, 2],
            "b-", linewidth=0.5, alpha=0.4, label="Training cameras")

    # Reference camera
    ax.scatter(*c2w_ref[:3, 3], c="red", s=80, zorder=5, label="Reference camera")

    # Zoom-in trajectory
    ax.plot(zoom_in_pos[:, 0], zoom_in_pos[:, 1], zoom_in_pos[:, 2],
            "g-", linewidth=2.5, label=f"Zoom-in (str={args.zoom_strength})")
    ax.scatter(*zoom_in_pos[-1], c="green", s=60, zorder=5)

    # Zoom-out trajectory
    ax.plot(zoom_out_pos[:, 0], zoom_out_pos[:, 1], zoom_out_pos[:, 2],
            "orange", linewidth=2.5, linestyle="--", label="Zoom-out")
    ax.scatter(*zoom_out_pos[-1], c="orange", s=60, zorder=5)

    # Axis labels and style
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    scene_name = os.path.basename(args.scene)
    ax.set_title(f"{scene_name} — point cloud + cameras + zoom trajectory")
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect ratio approximation
    extents = np.array([pts[:, i].ptp() for i in range(3)])
    max_extent = extents.max()
    mid = np.array([pts[:, i].mean() for i in range(3)])
    ax.set_xlim(mid[0] - max_extent/2, mid[0] + max_extent/2)
    ax.set_ylim(mid[1] - max_extent/2, mid[1] + max_extent/2)
    ax.set_zlim(mid[2] - max_extent/2, mid[2] + max_extent/2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
