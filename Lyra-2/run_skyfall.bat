@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PYTHONPATH=%~dp0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTHONUNBUFFERED=1

if not exist ".ffmpeg_path_cache.bat" (
    python -c "import imageio_ffmpeg,shutil,os; src=imageio_ffmpeg.get_ffmpeg_exe(); d=os.path.dirname(src); dst=os.path.join(d,'ffmpeg.exe'); shutil.copy2(src,dst) if not os.path.exists(dst) else None; open('.ffmpeg_path_cache.bat','w').write('@set FFMPEG_DIR='+d+'\n')"
)
call .ffmpeg_path_cache.bat
set PATH=%FFMPEG_DIR%;%PATH%

:: Step 0 — download Skyfall-GS dataset (skips already-downloaded files)
python scripts/download_skyfall.py --city NYC
if errorlevel 1 goto :error

:: Step 1 — prepare inputs from Skyfall-GS
python scripts/prepare_skyfall.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input --frame_idx 0
if errorlevel 1 goto :error

:: Step 2 — run Lyra-2 inference
python -m lyra_2._src.inference.lyra2_zoomgs_inference --input_image_path assets/skyfall_input --prompt_dir assets/skyfall_input --num_samples 4 --experiment lyra2 --checkpoint_dir checkpoints/model --output_path results_skyfall/videos --num_frames_zoom_in 81 --num_frames_zoom_out 81 --resolution 320,576 --offload_when_prompt --warp_chunk_size 4 --use_dmd --torch_compile --log_file results_skyfall\run.log
if errorlevel 1 goto :error

echo.
echo All steps complete. Results in results_skyfall\videos\
goto :eof

:error
echo.
echo ERROR: step failed (exit code %errorlevel%)
exit /b 1
