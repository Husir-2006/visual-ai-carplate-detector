@echo off
chcp 65001 >nul
cd /d %~dp0
echo [1/4] Installing basic AI dependencies if needed...
python -m pip install requests opencv-python ultralytics
echo.
echo [2/4] Downloading CCPD2020 from Zenodo. This file is about 0.85GB and may take a while.
python download_ccpd_dataset.py --contains CCPD2020 --target datasets/CCPD
echo.
echo [3/4] Extract datasets\CCPD\CCPD2020.zip manually if Windows has not extracted it.
echo Recommended extraction folder: datasets\CCPD\CCPD2020
echo.
echo [4/4] After extraction, run:
echo python prepare_ccpd_dataset.py --source datasets/CCPD --output datasets/ccpd_yolo --limit 50000 --clean
pause
