@echo off
chcp 65001 >nul
cd /d %~dp0
echo [1/3] Converting CCPD images to YOLO format...
python prepare_ccpd_dataset.py --source datasets/CCPD --output datasets/ccpd_yolo --limit 50000 --clean
echo.
echo [2/3] Training YOLOv8 plate detector on Chinese license plates...
echo If you have NVIDIA GPU, edit this file and change --device cpu to --device 0.
python train_yolov8_plate.py --data datasets/ccpd_yolo/data.yaml --epochs 80 --imgsz 640 --batch 16 --device cpu --output models/yolov8_plate.pt
echo.
echo [3/3] Done. Restart the Flask system to load models\yolov8_plate.pt.
pause
