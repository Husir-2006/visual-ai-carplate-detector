# 企业车牌识别与车辆管理系统

本项目是 P402019B 创新应用综合实训课程大作业，面向校园/企业门岗车辆通行管理场景。系统支持上传车辆图片，自动检测车辆与车牌区域，识别车牌号，并根据车牌查询车辆档案、通行记录、黑白名单和系统设置。

项目当前包含两个版本：

- 非静态版：Flask 后端 + 前端页面 + YOLOv8/ OCR 推理，推荐用于演示。
- 静态版：`static-demo.html`，用于无后端环境下快速展示页面形态。

## 核心功能

- 车辆与车牌检测：优先加载 `models/yolov8_plate.pt` 国内蓝牌 YOLOv8 模型。
- 车牌 OCR：优先使用 RapidOCR，兼容 PaddleOCR、EasyOCR、Tesseract 等方案。
- 车辆档案查询：识别车牌后自动匹配 `data/fleet.json` 中的企业车辆档案。
- 通行记录写入：每次识别成功后自动新增一条 `data/pass_records.json` 通行流水。
- 管理模块：车辆档案、通行记录、黑白名单、系统设置均通过 JSON 接口驱动。
- 证据返回：接口返回标注图、车辆裁剪图、车牌裁剪图和目标明细，便于演示和复核。

## 一键运行

双击：

```text
运行非静态版.bat
```

然后打开：

```text
http://127.0.0.1:5000
```

基础依赖：

```bash
pip install -r requirements.txt
```

增强 OCR / YOLOv8 训练依赖：

```bash
pip install -r requirements-ai.txt
```

## 接口说明

- `/detect`：上传图片并返回识别结果，同时自动写入通行记录。
- `/api/fleet`：车辆档案数据。
- `/api/pass-records`：通行记录数据。
- `/api/access-lists`：黑白名单数据。
- `/api/settings`：系统设置数据。

## 数据集与训练

早期演示数据包含国外/印尼车牌，能证明检测与 OCR 流程，但不适合真实中国蓝牌。后续项目引入 CCPD2020 国内车牌数据集，转换后得到 11776 张有效样本，并训练 YOLOv8 车牌检测模型。

推荐数据集：

- CCPD 官方项目：`https://github.com/detectRecog/CCPD`
- Zenodo 数据归档：`https://zenodo.org/records/15647076`
- 已使用文件：`CCPD2020.zip`

下载：

```bash
python download_ccpd_dataset.py --contains CCPD2020 --target datasets/CCPD
```

转换：

```bash
python prepare_ccpd_dataset.py --source datasets/CCPD --output datasets/ccpd_yolo --limit 50000 --clean
```

训练：

```bash
python train_yolov8_plate.py --data datasets/ccpd_yolo/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0 --output models/yolov8_plate.pt
```

已完成训练结果：

- 有效样本：11776 张。
- 训练轮数：86 轮提前停止。
- 最佳模型：约第 66 轮。
- mAP50：0.995。
- mAP50-95：约 0.897。
- 输出模型：`models/yolov8_plate.pt`。

## 项目结构

```text
visual-ai-campus-detector/
  app.py                         Flask 后端入口
  detector.py                    检测模型调度与结果绘制
  ocr_engine.py                  OCR 调度与车牌文本清洗
  train_yolov8_plate.py          YOLOv8 训练脚本
  prepare_ccpd_dataset.py        CCPD 转 YOLOv8 数据脚本
  data/                          演示车辆档案、记录、黑白名单和设置
  models/                        课程模型文件
  static/                        前端 JS/CSS
  templates/                     Flask 页面模板
  docs/                          数据集和结构说明
  deliverables/                  报告、PPT、视频、日志、截图
  24281098.zip                   最终提交包
```

## 最终提交材料

最终课程材料位于：

```text
deliverables/
```

最终压缩包：

```text
24281098.zip
```

压缩包包含源码、模型、系统数据、课程报告、PPT、演示视频、讲稿、开发日志、沟通记录、个人实习实训日志和运行截图；不包含原始数据集、训练缓存、上传缓存和临时工作区。
