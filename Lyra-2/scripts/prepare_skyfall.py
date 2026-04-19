"""
Prepare Skyfall-GS scenes for Lyra-2 inference.

Reads datasets_NYC/<scene>/transforms.json and extracts a keyframe image
(+ camera pose) into assets/skyfall_input/ in Lyra-2's expected format:
  00.png, 00.txt   <- scene 0 reference frame + prompt
  01.png, 01.txt   <- scene 1 ...

Usage:
  python scripts/prepare_skyfall.py --skyfall_dir assets/skyfall/datasets_NYC
  python scripts/prepare_skyfall.py --skyfall_dir assets/skyfall/datasets_NYC --frame_idx 30
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


DEFAULT_PROMPT = (
    "A static urban street scene. The camera slowly pushes forward. "
    "Buildings, sidewalks and parked vehicles remain perfectly still."
)


def load_transforms(scene_dir: str):
    for name in ("transforms.json", "transforms_train.json"):
        path = os.path.join(scene_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"No transforms.json in {scene_dir}")


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Invert a 4x4 camera-to-world matrix to world-to-camera."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


def build_K(meta: dict) -> np.ndarray:
    return np.array([
        [meta["fl_x"], 0,            meta["cx"]],
        [0,            meta["fl_y"], meta["cy"]],
        [0,            0,            1.0],
    ], dtype=np.float32)


def center_crop_to_aspect(img: Image.Image, K: np.ndarray, aspect_w: float, aspect_h: float):
    """Center-crop image and adjust K to match the given aspect ratio."""
    W, H = img.size
    target_w = W
    target_h = round(W * aspect_h / aspect_w)
    if target_h > H:
        target_h = H
        target_w = round(H * aspect_w / aspect_h)
    left = (W - target_w) // 2
    top  = (H - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    K = K.copy()
    K[0, 2] -= left   # cx
    K[1, 2] -= top    # cy
    return img, K, target_w, target_h


def prepare_scene(scene_dir: str, out_dir: str, idx: int, frame_idx: int, prompt: str,
                  crop_aspect: tuple[float, float] | None = None):
    meta = load_transforms(scene_dir)
    frames = meta["frames"]
    frame = frames[min(frame_idx, len(frames) - 1)]

    src_img = os.path.join(scene_dir, frame["file_path"])
    if not os.path.exists(src_img):
        base = os.path.splitext(src_img)[0]
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            if os.path.exists(base + ext):
                src_img = base + ext
                break

    tag = f"{idx:02d}"
    dst_img = os.path.join(out_dir, f"{tag}.png")
    dst_txt = os.path.join(out_dir, f"{tag}.txt")
    dst_pose = os.path.join(out_dir, f"{tag}_pose.npz")

    img = Image.open(src_img).convert("RGB")
    K = build_K(meta)
    img_w, img_h = meta.get("w", img.size[0]), meta.get("h", img.size[1])

    if crop_aspect is not None:
        img, K, img_w, img_h = center_crop_to_aspect(img, K, crop_aspect[0], crop_aspect[1])

    img.save(dst_img)

    with open(dst_txt, "w") as f:
        f.write(prompt)

    c2w = np.array(frame["transform_matrix"], dtype=np.float32)
    w2c = c2w_to_w2c(c2w)
    np.savez(dst_pose, w2c=w2c, K=K, c2w=c2w,
             image_wh=np.array([img_w, img_h]))

    scene_name = os.path.basename(scene_dir)
    crop_str = f" (cropped {img.size[0]}×{img.size[1]})" if crop_aspect else ""
    print(f"  [{tag}] {scene_name} frame {frame_idx} -> {dst_img}{crop_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyfall_dir", default="assets/skyfall/datasets_NYC",
                        help="Path to the datasets_NYC (or datasets_Jacksonville) folder")
    parser.add_argument("--out_dir", default="assets/skyfall_input",
                        help="Output directory for Lyra-2 inference inputs")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Which frame index to use as reference (default: 0)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--scenes", nargs="*",
                        help="Specific scene names to process (default: all)")
    parser.add_argument("--crop_aspect", type=str, default="16:9",
                        help="Center-crop to W:H aspect ratio, e.g. 16:9. Pass 'none' to disable.")
    args = parser.parse_args()

    crop_aspect = None
    if args.crop_aspect.lower() != "none":
        aw, ah = args.crop_aspect.split(":")
        crop_aspect = (float(aw), float(ah))

    os.makedirs(args.out_dir, exist_ok=True)

    scenes = sorted([
        d for d in os.listdir(args.skyfall_dir)
        if os.path.isdir(os.path.join(args.skyfall_dir, d))
    ])
    if args.scenes:
        scenes = [s for s in scenes if s in args.scenes]

    print(f"Preparing {len(scenes)} scene(s) -> {args.out_dir}")
    for i, scene in enumerate(scenes):
        prepare_scene(
            scene_dir=os.path.join(args.skyfall_dir, scene),
            out_dir=args.out_dir,
            idx=i,
            frame_idx=args.frame_idx,
            prompt=args.prompt,
            crop_aspect=crop_aspect,
        )

    print(f"\nDone. Run inference with:")
    print(f"  run_example.bat  (update --input_image_path {args.out_dir} --prompt_dir {args.out_dir})")


if __name__ == "__main__":
    main()
