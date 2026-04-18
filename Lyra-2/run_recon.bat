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

:: MSVC required for VIPE CUDA JIT extension
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%

:: Usage: run_recon.bat <video_path> [output_dir]
:: Example: run_recon.bat my_video.mp4
:: Example: run_recon.bat my_video.mp4 outputs/my_scene

if "%~1"=="" (
    echo Usage: run_recon.bat ^<video_path^> [output_dir]
    exit /b 1
)

set VIDEO=%~1
set OUTPUT=%~2
if "%OUTPUT%"=="" set OUTPUT=outputs\recon

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set START_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set START_TICKS=%%t

python -m lyra_2._src.inference.vipe_da3_gs_recon ^
    --input_video_path "%VIDEO%" ^
    --output_dir "%OUTPUT%" ^
    --da3_model_path_custom checkpoints/recon/model.pt ^
    --da3_max_frames 96 ^
    --max_resolution 1080

for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set END_TIME=%%t
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set END_TICKS=%%t

powershell -NoProfile -Command "$elapsed = [timespan]::FromTicks(%END_TICKS% - %START_TICKS%); Write-Host ''; Write-Host ('Start   : %START_TIME%'); Write-Host ('End     : %END_TIME%'); Write-Host ('Elapsed : ' + ('{0}h {1}m {2}s' -f [int]$elapsed.TotalHours, $elapsed.Minutes, $elapsed.Seconds))"
