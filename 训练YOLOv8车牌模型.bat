@echo off
cd /d %~dp0
python train_yolov8_plate.py --data datasets/combined/data.yaml --epochs 80 --imgsz 640 --batch 16
pause
