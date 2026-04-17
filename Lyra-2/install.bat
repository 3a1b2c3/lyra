@echo off
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat

pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

:: Main requirements (megatron-core excluded — installed separately below)
pip install --no-deps -r requirements_windows_base.txt

:: megatron-core: cannot be built on Windows; a single-GPU stub lives at
:: Lyra-2/megatron/ and is importable when PYTHONPATH includes the Lyra-2 dir
echo [INFO] Skipping megatron-core (using built-in single-GPU stub)

:: MoGe: use uv to avoid the Windows pip git-subprocess thread bug
uv pip install "git+https://github.com/microsoft/MoGe.git"

:: transformer_engine: optional — Linux only; wan2pt1.py falls back to torch SDPA if absent
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%
set CUDA_PATH=%CUDA_HOME%
pip install --no-build-isolation "transformer_engine[pytorch]" || echo [WARN] transformer_engine not installed - using PyTorch SDPA fallback

:: flash-attn: use pre-built Windows wheel (cu128, torch 2.7, cp312)
pip install --no-deps -r requirements_windows.txt

:: Activate MSVC (cl.exe) so CUDA extensions can compile
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set USE_SYSTEM_EIGEN=1
pip install --no-build-isolation -e lyra_2/_src/inference/vipe
pip install --no-build-isolation -e "lyra_2/_src/inference/depth_anything_3[gs]"

set PYTHONPATH=%~dp0
python -c "import torch, flash_attn, vipe_ext, depth_anything_3.api, moge.model.v1; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available()); print('all imports OK')"
