# 企业车牌识别与车辆管理系统

本项目是 P402019B 创新应用综合实训课程作业，面向校园/企业门岗车辆通行管理场景。系统支持上传车辆图片，自动检测车辆与车牌区域，识别车牌号，并根据车牌号查询车辆档案、通行记录和黑白名单信息。

## 一键运行

双击：

```text
运行非静态版.bat
```

然后打开：

```text
http://127.0.0.1:5000
```

首次运行如缺少依赖，先安装：

```bash
pip install -r requirements.txt
```

## 系统模块

非静态版前端包含 5 个模块：

- 识别工作台：上传图片，返回识别图、车辆图、车牌裁剪图和目标明细。
- 车辆档案：查看登记车辆、车主部门、权限和最近通行信息。
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

当前仓库保留了课程演示用模型：

```text
models/tiny_plate_detector.pt
```

如果训练出更高准确率模型，把 `best.pt` 放到 `models/best.pt` 或 `models/yolov8_plate.pt`，重启系统即可自动优先加载。

## 推荐训练方式

基础依赖：

```bash
pip install -r requirements.txt
```

增强训练和 OCR 依赖：

```bash
pip install -r requirements-ai.txt
```

合并数据集：

```bash
python prepare_combined_dataset.py --base datasets/anpr-model-1 --extra datasets --output datasets/combined
```

推荐训练 YOLOv8：

```bash
python train_yolov8_plate.py --data datasets/combined/data.yaml --epochs 80 --imgsz 640 --batch 16
```

训练完成后脚本会尝试把最佳模型复制到：

```text
models/yolov8_plate.pt
```

CPU 训练会比较慢；如果电脑没有 NVIDIA 显卡，可以在 Kaggle 或 Colab 训练，再把 `best.pt` 下载回来放进 `models/`。

## 数据集来源

原始数据集较大，按课程要求不放入最终提交包，只在报告中注明来源：

1. Zenodo / Roboflow Universe：Indonesian License Plate Detection Dataset，DOI：10.5281/zenodo.15605718
2. Roboflow Public：License Plates Dataset，YOLO v5 PyTorch 格式
3. Kaggle：Automatic License Plate Recognition (ALPR) Dataset

本地数据集目录：

```text
datasets/
```

最终提交包不会包含该目录。

## 项目结构

详细文件说明见：

```text
docs/project_structure.md
```

## 课程材料

最终材料位于：

```text
deliverables/
```

包含课程报告、陈述 PPT、功能演示视频、开发日志与沟通记录、三份个人实习实训日志、运行截图和提交清单。

最终压缩包：

```text
24281098.zip
```

该压缩包按组长学号命名，已排除原始数据集、运行缓存、旧包和临时预览文件。
