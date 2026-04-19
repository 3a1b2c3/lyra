@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PYTHONPATH=%~dp0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTHONUNBUFFERED=1

:: ── Frame interpolation settings ────────────────────────────────────────────
:: RIFE_FACTOR=4 → 21 input frames → 81 output frames (~4x faster inference)
:: RIFE_FACTOR=2 → 41 input frames → 81 output frames (~2x faster inference)
set RIFE_FACTOR=4
set /a ZOOM_FRAMES=(81-1)/RIFE_FACTOR+1
:: ────────────────────────────────────────────────────────────────────────────

if not exist ".ffmpeg_path_cache.bat" (
    python -c "import imageio_ffmpeg,shutil,os; src=imageio_ffmpeg.get_ffmpeg_exe(); d=os.path.dirname(src); dst=os.path.join(d,'ffmpeg.exe'); shutil.copy2(src,dst) if not os.path.exists(dst) else None; open('.ffmpeg_path_cache.bat','w').write('@set FFMPEG_DIR='+d+'\n')"
)
call .ffmpeg_path_cache.bat
set PATH=%FFMPEG_DIR%;%PATH%

:: Step 0 — download Skyfall-GS dataset (skips already-downloaded files)
python scripts/download_skyfall.py --city NYC
if errorlevel 1 goto :error

:: Step 1 — extract reference frames (NYC_336 only)
python scripts/prepare_skyfall.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input_336 --frame_idx 0 --scenes NYC_336
if errorlevel 1 goto :error

:: Step 2 — project PLY point clouds into reference cameras (NYC_336 only)
python scripts/prepare_skyfall_depth.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input_336 --frame_idx 0 --scenes NYC_336
if errorlevel 1 goto :error

:: Step 3 — build multi-frame spatial cache (optional; uncomment to enable)
:: python scripts/prepare_skyfall_multiframe.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input_336 --frame_idx 0 --target_hw 320 576 --scenes NYC_336
:: if errorlevel 1 goto :error

:: Step 4 — run Lyra-2 inference with reduced frame count
python -m lyra_2._src.inference.lyra2_zoomgs_inference --input_image_path assets/skyfall_input_336/00.png --prompt_dir assets/skyfall_input_336 --depth_dir assets/skyfall_input_336 --num_samples 1 --experiment lyra2 --checkpoint_dir checkpoints/model --output_path results_skyfall_336_rife/videos --num_frames_zoom_in !ZOOM_FRAMES! --num_frames_zoom_out !ZOOM_FRAMES! --resolution 320,576 --offload_when_prompt --warp_chunk_size 4 --use_dmd --torch_compile --zoom_in_strength 0.15 --zoom_out_strength 0.5 --log_file results_skyfall_336_rife\run.log
if errorlevel 1 goto :error

:: Step 5 — interpolate frames back to 81
echo.
echo Interpolating frames !ZOOM_FRAMES! ^→ 81 ^(factor %RIFE_FACTOR%^)...
python scripts/interpolate_videos.py --per_image_dir results_skyfall_336_rife/videos/00 --factor %RIFE_FACTOR% --fps 16
if errorlevel 1 goto :error

echo.
echo All steps complete. Results in results_skyfall_336_rife\videos\
goto :eof

:error
echo.
echo ERROR: step failed (exit code %errorlevel%)
exit /b 1
