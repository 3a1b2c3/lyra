@echo off
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat

set PYTHONPATH=%~dp0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: Put imageio-ffmpeg's bundled ffmpeg on PATH so ffmpegcv can find it
for /f "tokens=*" %%p in ('python -c "import imageio_ffmpeg, os; print(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))"') do set PATH=%%p;%PATH%

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set START_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set START_TICKS=%%t

python -m lyra_2._src.inference.lyra2_zoomgs_inference ^
    --input_image_path assets/samples ^
    --prompt_dir assets/samples ^
    --sample_id 0 ^
    --experiment lyra2 ^
    --checkpoint_dir checkpoints/model ^
    --output_path results_example/videos ^
    --num_frames_zoom_in 81 ^
    --num_frames_zoom_out 81 ^
    --resolution 480,832 ^
    --offload ^
    --warp_chunk_size 4

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set END_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set END_TICKS=%%t

powershell -NoProfile -Command "$elapsed = [timespan]::FromTicks(%END_TICKS% - %START_TICKS%); Write-Host ''; Write-Host ('Start   : %START_TIME%'); Write-Host ('End     : %END_TIME%'); Write-Host ('Elapsed : ' + ('{0}h {1}m {2}s' -f [int]$elapsed.TotalHours, $elapsed.Minutes, $elapsed.Seconds))"
