@echo off
cd /d %~dp0
echo ??????????????????...
echo.
python prepare_combined_dataset.py --base datasets/anpr-model-1 --extra datasets --output datasets/combined
python train_plate_model.py --data datasets/combined --epochs 12 --batch 64
pause
