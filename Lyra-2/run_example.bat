@echo off
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat

set PYTHONPATH=%~dp0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set START_TIME=%TIME%

python -m lyra_2._src.inference.lyra2_zoomgs_inference ^
    --input_image_path assets/samples ^
    --prompt_dir assets/samples ^
    --sample_id 0 ^
    --experiment lyra2 ^
    --checkpoint_dir checkpoints/model ^
    --output_path results_example/videos ^
    --num_frames_zoom_in 81 ^
    --num_frames_zoom_out 81 ^
    --resolution 480,832

echo.
echo Start : %START_TIME%
echo End   : %TIME%
