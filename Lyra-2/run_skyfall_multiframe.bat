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

:: Step 1 — extract reference frames (1 per scene)
python scripts/prepare_skyfall.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input --frame_idx 0
if errorlevel 1 goto :error

:: Step 2 — project PLY point clouds into reference cameras (replaces DA3 depth)
python scripts/prepare_skyfall_depth.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input --frame_idx 0
if errorlevel 1 goto :error

:: Step 3 — build multi-frame spatial cache (4 frames adjacent to reference frame)
python scripts/prepare_skyfall_multiframe.py --skyfall_dir assets/skyfall/datasets_NYC --out_dir assets/skyfall_input --num_frames 4 --frame_idx 0 --target_hw 320 576
if errorlevel 1 goto :error

:: Step 4 — run Lyra-2 inference with PLY depth + multiframe spatial cache
python -m lyra_2._src.inference.lyra2_zoomgs_inference --input_image_path assets/skyfall_input --prompt_dir assets/skyfall_input --depth_dir assets/skyfall_input --multiframe_cache_dir assets/skyfall_input --num_retrieval_views 3 --num_samples 4 --experiment lyra2 --checkpoint_dir checkpoints/model --output_path results_skyfall_multiframe/videos --num_frames_zoom_in 81 --num_frames_zoom_out 81 --resolution 320,576 --offload_when_prompt --warp_chunk_size 4 --use_dmd --torch_compile --log_file results_skyfall_multiframe\run.log
if errorlevel 1 goto :error

echo.
echo All steps complete. Results in results_skyfall_multiframe\videos\
goto :eof

:error
echo.
echo ERROR: step failed (exit code %errorlevel%)
exit /b 1
