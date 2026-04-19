"""Download Eigen 3.4.0 headers into the torch include path (needed for VIPE CUDA build)."""
import io, os, urllib.request, zipfile
import torch

dest = os.path.join(os.path.dirname(torch.__file__), "include", "eigen3")
if os.path.exists(os.path.join(dest, "Eigen")):
    print("[eigen3] already installed, skipping.")
else:
    print("[eigen3] downloading Eigen 3.4.0...")
    url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    with urllib.request.urlopen(url) as r:
        data = r.read()
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            parts = name.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith("Eigen"):
                continue
            out = os.path.join(dest, parts[1].replace("/", os.sep))
            if name.endswith("/"):
                os.makedirs(out, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(name) as src, open(out, "wb") as dst:
                    dst.write(src.read())
    print(f"[eigen3] installed to {dest}")
