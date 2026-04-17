@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%
C:\workspace\world\lyra\.venv\Scripts\python.exe -m pip install --no-build-isolation -e C:\workspace\world\lyra\Lyra-2\lyra_2\_src\inference\vipe > C:\workspace\world\lyra\Lyra-2\build_vipe.log 2>&1
echo Exit code: %ERRORLEVEL% >> C:\workspace\world\lyra\Lyra-2\build_vipe.log
