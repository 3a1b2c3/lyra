"""Download nvidia/Lyra-2.0 from HuggingFace using curl."""
import os
import shutil
import subprocess

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"

from huggingface_hub import list_repo_files, hf_hub_url
from huggingface_hub.utils import get_token

REPO_ID = "nvidia/Lyra-2.0"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

token = get_token()
if not token:
    raise RuntimeError("No HuggingFace token found. Run: huggingface-cli login")

def free_gb():
    return shutil.disk_usage(LOCAL_DIR).free / 1024**3

print(f"Free space: {free_gb():.1f} GB")

files = list(list_repo_files(REPO_ID))
print(f"Repo has {len(files)} files total.")

for repo_path in files:
    dest = os.path.join(LOCAL_DIR, repo_path.replace("/", os.sep))
    size = os.path.getsize(dest) if os.path.exists(dest) else -1

    if size >= 0:  # 0-byte files are valid (intentional empty shards)
        print(f"  skip  {repo_path}")
        continue

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = hf_hub_url(REPO_ID, repo_path)
    print(f"  fetch {repo_path} ({free_gb():.1f} GB free) ...", flush=True)

    result = subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5",
         "-H", f"Authorization: Bearer {token}",
         "-w", "\nHTTP %{http_code}  final_url: %{url_effective}  size: %{size_download}\n",
         "--output", dest, url],
        capture_output=False,
    )

    actual = os.path.getsize(dest) if os.path.exists(dest) else 0

    if result.returncode != 0:
        if 0 < actual < 2000:
            with open(dest, "rb") as f:
                print(f"    file content ({actual} bytes):", f.read().decode("utf-8", errors="replace"))
        raise RuntimeError(
            f"curl failed (exit {result.returncode}) for {repo_path} — "
            f"{actual} bytes written, {free_gb():.1f} GB free"
        )

    mb = actual // 1024**2
    print(f"        done ({mb} MB, {free_gb():.1f} GB free)")

print("All done.")
