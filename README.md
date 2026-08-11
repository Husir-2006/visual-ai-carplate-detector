# 企业车牌识别与车辆管理系统

本项目是 P402019B 创新应用综合实训课程大作业，面向校园/企业门岗车辆通行管理场景。系统支持上传车辆图片，自动检测车辆与车牌区域，识别车牌号，并根据车牌号查询车辆档案、通行记录和黑白名单信息。

## 重要说明：为什么国内车牌识别会不稳定

早期演示模型主要使用国外/印尼车牌数据训练，车牌颜色、字符结构、拍摄距离和中文省份简称都与中国蓝牌差异很大。因此在真实国内车牌上容易出现这些问题：

- 省份汉字丢失，例如 `粤A78T3R` 被识别成 `A78T3R` 或 `478T3R`。
- 相似字符混淆，例如 `A/4`、`B/8`、`T/1`、`S/5`。
- 车牌检测框偏移，导致 OCR 读到保险杠、车标或背景文字。

当前版本已经增加了国内蓝牌的后处理规则和车辆档案模糊匹配，但更根本的改进方式是使用国内车牌数据集重新训练检测模型。

## 一键运行系统

双击：

```text
运行非静态版.bat
```

然后打开：

```text
http://127.0.0.1:5000
```

首次运行如果缺少基础依赖：

```bash
pip install -r requirements.txt
```

如需 PaddleOCR / EasyOCR / YOLOv8 训练能力：

```bash
pip install -r requirements-ai.txt
```

## 推荐国内数据集

优先使用 CCPD 中国车牌数据集：

- CCPD2019：规模大，适合蓝牌检测和复杂场景训练。
- CCPD2020：体积较小，包含更多国内新场景，适合课程项目先快速训练。
- 数据源：Zenodo CCPD archive，记录号 `15647076`。
- 官方项目：`https://github.com/detectRecog/CCPD`

原始数据集较大，不放进最终提交包，只在报告中注明来源。

## 下载并转换 CCPD

推荐先下载 CCPD2020，约 0.85GB：

```bash
python download_ccpd_dataset.py --list
python download_ccpd_dataset.py --contains CCPD2020 --target datasets/CCPD
```

下载完成后，把 `datasets/CCPD/CCPD2020.zip` 解压到：

```text
datasets/CCPD/CCPD2020
```

转换为 YOLOv8 格式：

```bash
python prepare_ccpd_dataset.py --source datasets/CCPD --output datasets/ccpd_yolo --limit 50000 --clean
```

输出结构：

```text
datasets/ccpd_yolo/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
  data.yaml
  ocr_labels.csv
```

其中 `ocr_labels.csv` 保存了从 CCPD 文件名解析出来的真实车牌号，后续如果要继续训练字符识别模型，可以直接复用。

## 重新训练国内车牌检测模型

CPU 训练：

```bash
python train_yolov8_plate.py --data datasets/ccpd_yolo/data.yaml --epochs 80 --imgsz 640 --batch 16 --device cpu --output models/yolov8_plate.pt
```

有 NVIDIA 显卡时建议改成：

```bash
python train_yolov8_plate.py --data datasets/ccpd_yolo/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0 --output models/yolov8_plate.pt
```

也可以直接双击：

```text
训练国内车牌YOLOv8模型.bat
```

训练完成后重启 Flask 系统，程序会优先加载：

```text
models/yolov8_plate.pt
```

## 系统模块

非静态版前端包含 5 个模块：

- 识别工作台：上传图片，返回识别图、车辆图、车牌裁剪图和目标明细。
- 车辆档案：查看登记车辆、车主/部门、权限和最近通行信息。
- 通行记录：查看门岗通行流水。
- 黑白名单：查看自动放行车辆和需要人工复核车辆。
- 系统设置：查看当前模型优先级、OCR 引擎和识别阈值。

后端接口：

- `/detect`：图片识别接口。
- `/api/fleet`：车辆档案 JSON。
- `/api/pass-records`：通行记录 JSON。
- `/api/access-lists`：黑白名单 JSON。
- `/api/settings`：系统设置 JSON。

## AI 推理策略

检测模型优先级：

```text
models/yolov8_plate.pt 或 models/best.pt
> models/yolov5s.onnx
> models/tiny_plate_detector.pt
> OpenCV 兜底检测
```

OCR 优先级：

```text
PaddleOCR > EasyOCR > Tesseract > 数据集文件名标注兜底
```

车辆档案校正：

```text
OCR 原始结果
> 清洗符号和噪声
> 中国蓝牌结构校验
> 近似字符纠错
> 与车辆档案库模糊匹配
```

## 最终材料

课程提交材料位于：

```text
deliverables/
```

最终压缩包：

```text
24281098.zip
```

压缩包按组长学号命名，已排除原始数据集、运行缓存、旧包和临时预览文件。
