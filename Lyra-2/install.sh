#!/usr/bin/env bash
set -euo pipefail

# Activate venv (create if missing)
VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3.10 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Set CUDA_HOME to system CUDA installation
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Build environment variables
SITE="$VENV_DIR/lib/python3.10/site-packages"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LD_LIBRARY_PATH="$VENV_DIR/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Install PyTorch
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# Install Flash Attention
MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

# Build vendored CUDA extensions
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

# Verify installation
export LD_LIBRARY_PATH="$VENV_DIR/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help
