"""Linear frame interpolation post-process for Lyra-2 zoom videos.

Reads zoom_in.mp4 and zoom_out.mp4 from a per-image output directory,
inserts (factor-1) interpolated frames between each pair, overwrites the
files, and re-creates combined.mp4.

Usage:
  python scripts/interpolate_videos.py --per_image_dir results_skyfall_336/videos/00 --factor 4 --fps 16
"""
import argparse
import os

import cv2
import imageio
import numpy as np


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def write_video_frames(frames, path, fps):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, quality=5, macro_block_size=1,
                                output_params=["-f", "mp4"])
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def interpolate(frames, factor):
    """Insert (factor-1) linearly blended frames between each consecutive pair."""
    if factor <= 1 or len(frames) < 2:
        return frames
    out = []
    for i in range(len(frames) - 1):
        f0 = frames[i].astype(np.float32)
        f1 = frames[i + 1].astype(np.float32)
        out.append(frames[i])
        for j in range(1, factor):
            t = j / factor
            out.append((f0 * (1.0 - t) + f1 * t).astype(np.uint8))
    out.append(frames[-1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_image_dir", required=True,
                        help="Per-image output dir (e.g. results_skyfall_336/videos/00)")
    parser.add_argument("--factor", type=int, default=4,
                        help="Interpolation factor: 4 → 21 frames become 81 (default: 4)")
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    interp_frames = {}
    for tag in ("zoom_in", "zoom_out"):
        path = os.path.join(args.per_image_dir, f"{tag}.mp4")
        if not os.path.exists(path):
            print(f"  [{tag}] not found, skipping")
            continue
        frames = read_video_frames(path)
        out_frames = interpolate(frames, args.factor)
        n_in, n_out = len(frames), len(out_frames)
        print(f"  [{tag}] {n_in} → {n_out} frames ({args.factor}× interp)")
        write_video_frames(out_frames, path, args.fps)
        interp_frames[tag] = out_frames

    # Re-create combined: reversed zoom_out + zoom_in
    if "zoom_in" in interp_frames and "zoom_out" in interp_frames:
        combined = list(reversed(interp_frames["zoom_out"])) + interp_frames["zoom_in"]
        combined_path = os.path.join(args.per_image_dir, "combined.mp4")
        print(f"  [combined] {len(combined)} frames")
        write_video_frames(combined, combined_path, args.fps)

    print("Done.")


if __name__ == "__main__":
    main()
