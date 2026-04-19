@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PYTHONPATH=%~dp0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTHONUNBUFFERED=1

:: ffmpeg
if not exist ".ffmpeg_path_cache.bat" (
    python -c "import imageio_ffmpeg,shutil,os; src=imageio_ffmpeg.get_ffmpeg_exe(); d=os.path.dirname(src); dst=os.path.join(d,'ffmpeg.exe'); shutil.copy2(src,dst) if not os.path.exists(dst) else None; open('.ffmpeg_path_cache.bat','w').write('@set FFMPEG_DIR='+d+'\n')"
)
call .ffmpeg_path_cache.bat
set PATH=%FFMPEG_DIR%;%PATH%

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set START_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set START_TICKS=%%t

:: ── Step 1: Download ────────────────────────────────────────────────────────
if exist "assets\skyfall\datasets_NYC" (
    echo [1/3] Skyfall data already present, skipping download.
) else (
    echo [1/3] Downloading Skyfall-GS NYC dataset...
    python scripts\download_skyfall.py --city NYC --out assets\skyfall
    if errorlevel 1 ( echo ERROR: download failed & pause & exit /b 1 )
)

:: ── Step 2: Prepare inputs ──────────────────────────────────────────────────
echo [2/3] Preparing inputs from Skyfall scenes...
python scripts\prepare_skyfall.py --skyfall_dir assets\skyfall\datasets_NYC --out_dir assets\skyfall_input --frame_idx 0
if errorlevel 1 ( echo ERROR: prepare failed & pause & exit /b 1 )

:: ── Step 3: Run Lyra-2 inference ────────────────────────────────────────────
echo [3/3] Running Lyra-2 inference...
python -m lyra_2._src.inference.lyra2_zoomgs_inference --input_image_path assets\skyfall_input --prompt_dir assets\skyfall_input --num_samples 4 --experiment lyra2 --checkpoint_dir checkpoints\model --output_path results_skyfall\videos --num_frames_zoom_in 81 --num_frames_zoom_out 81 --resolution 320,576 --offload_when_prompt --warp_chunk_size 4 --use_dmd --torch_compile --log_file results_skyfall\run.log
if errorlevel 1 ( echo ERROR: inference failed & pause & exit /b 1 )

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set END_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set END_TICKS=%%t

powershell -NoProfile -Command "$elapsed = [timespan]::FromTicks(%END_TICKS% - %START_TICKS%); Write-Host ''; Write-Host ('Start   : %START_TIME%'); Write-Host ('End     : %END_TIME%'); Write-Host ('Elapsed : ' + ('{0}h {1}m {2}s' -f [int]$elapsed.TotalHours, $elapsed.Minutes, $elapsed.Seconds))"
echo Results: results_skyfall\videos\
pause
