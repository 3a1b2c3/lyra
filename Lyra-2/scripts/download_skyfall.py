"""Download the Skyfall-GS dataset from HuggingFace.

Structure:
  datasets_NYC/  (474 MB) — NYC_004, NYC_010, NYC_219, NYC_336
  datasets_Jacksonville/  — Jacksonville scenes

Each scene: RGB images + masks + camera poses (transforms_train.json) + point clouds.
"""
import argparse
import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "jayinnn/Skyfall-GS-datasets"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "skyfall")


def list_repo_files(repo_id, prefix=None, max_retries=10, wait=10):
    api = HfApi()
    for attempt in range(1, max_retries + 1):
        try:
            files = list(api.list_repo_files(repo_id, repo_type="dataset"))
            if prefix:
                files = [f for f in files if f.startswith(prefix)]
            return files
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"\nList attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}. Retrying in {wait}s...")
            time.sleep(wait)


def download_file_with_retry(repo_id, filename, local_dir, max_retries=10, wait=10):
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=local_dir,
            )
            return
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"\n  Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}. Retrying in {wait}s...")
            time.sleep(wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=["NYC", "Jacksonville", "all"], default="NYC",
                        help="Which city subset to download (default: NYC)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()

    prefix = None if args.city == "all" else f"datasets_{args.city}/"

    print(f"Listing files in {REPO_ID} (city={args.city})...")
    files = list_repo_files(REPO_ID, prefix=prefix, max_retries=args.retries)
    print(f"Found {len(files)} files. Downloading to {args.out}")

    for i, filename in enumerate(files, 1):
        dest = os.path.join(args.out, filename)
        if os.path.exists(dest):
            print(f"[{i}/{len(files)}] Skip (exists): {filename}")
            continue
        print(f"[{i}/{len(files)}] {filename}", flush=True)
        download_file_with_retry(REPO_ID, filename, args.out, max_retries=args.retries)

    print(f"Done: {args.out}")
