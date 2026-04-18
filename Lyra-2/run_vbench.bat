@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PYTHONPATH=%~dp0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTHONUNBUFFERED=1

:: ffmpeg path — Python runs once to build cache, instant on subsequent runs
if not exist ".ffmpeg_path_cache.bat" (
    python -c "import imageio_ffmpeg,shutil,os; src=imageio_ffmpeg.get_ffmpeg_exe(); d=os.path.dirname(src); dst=os.path.join(d,'ffmpeg.exe'); shutil.copy2(src,dst) if not os.path.exists(dst) else None; open('.ffmpeg_path_cache.bat','w').write('@set FFMPEG_DIR='+d+'\n')"
)
call .ffmpeg_path_cache.bat
set PATH=%FFMPEG_DIR%;%PATH%

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set START_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set START_TICKS=%%t

python -m lyra_2._src.inference.lyra2_zoomgs_inference --input_image_path assets/samples --prompt_dir assets/samples --num_samples 5 --experiment lyra2 --checkpoint_dir checkpoints/model --output_path results_vbench/videos --num_frames_zoom_in 81 --num_frames_zoom_out 241 --resolution 320,576 --offload_when_prompt --warp_chunk_size 4 --use_dmd --worldcache 2>&1 | powershell -NoProfile -Command "& { $input | Tee-Object -FilePath 'results_vbench\run.log' }"

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set END_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set END_TICKS=%%t

powershell -NoProfile -Command "$elapsed = [timespan]::FromTicks(%END_TICKS% - %START_TICKS%); Write-Host ''; Write-Host ('Start   : %START_TIME%'); Write-Host ('End     : %END_TIME%'); Write-Host ('Elapsed : ' + ('{0}h {1}m {2}s' -f [int]$elapsed.TotalHours, $elapsed.Minutes, $elapsed.Seconds))"
