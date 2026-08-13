# 项目结构说明

本项目已按“可运行、可训练、可提交”的思路整理。根目录保留系统源码、训练脚本、运行脚本、模型、演示数据、课程材料和最终压缩包；原始数据集、训练缓存和运行缓存不进入最终提交包。

## 运行系统

- `app.py`：Flask 非静态版入口，提供页面、识别接口和 JSON 数据接口。
- `templates/index.html`：企业车辆管理系统页面结构。
- `static/app.js`：前端交互、模块切换、识别结果展示、车辆档案查询和通行记录刷新。
- `static/style.css`：后台系统样式。
- `data/*.json`：车辆档案、通行记录、黑白名单和系统设置数据。
- `运行非静态版.bat`：双击启动本地系统。
- `static-demo.html`：单文件静态演示版。

## AI 与训练

- `detector.py`：检测调度，优先使用 YOLOv8 权重，其次回退到其他检测方案。
- `ocr_engine.py`：OCR 调度，支持 RapidOCR、PaddleOCR、EasyOCR、Tesseract 和演示兜底。
- `train_yolov8_plate.py`：YOLOv8 车牌检测训练脚本。
- `prepare_ccpd_dataset.py`：CCPD 数据集转 YOLOv8 格式脚本。
- `download_ccpd_dataset.py`：CCPD 数据集下载脚本。
- `prepare_combined_dataset.py`：早期多数据集合并脚本。
- `train_plate_model.py`、`tiny_yolo.py`：早期轻量课程演示模型与训练脚本。
- `models/yolov8_plate.pt`：当前推荐使用的国内蓝牌检测模型。
- `models/tiny_plate_detector.pt`：早期轻量演示模型，作为备用材料保留。

## 本地数据与输出

- `datasets/`：本地原始数据集和转换后的训练集，保留给继续训练使用，不进入最终提交包。
- `runs/`：YOLOv8 训练输出、验证图、训练曲线和指标记录，保留给复核训练结果使用，不进入最终提交包。
- `outputs/charts/`：课程材料中使用的训练曲线。
- `outputs/samples/`：保留的测试样例图。
- `uploads/`：运行时上传缓存，可删除，程序启动后会自动创建。
- `outputs/*.jpg/png`：运行时生成的识别结果图，可删除；正式截图已整理到 `deliverables/screenshots/`。

## 课程提交材料

- `deliverables/24281098-课程报告-企业车牌识别与车辆管理系统.docx`
- `deliverables/24281098-课程报告-企业车牌识别与车辆管理系统.pdf`
- `deliverables/24281098-陈述PPT-企业车牌识别与车辆管理系统.pptx`
- `deliverables/24281098-功能演示视频-企业车牌识别与车辆管理系统.mp4`
- `deliverables/24281098-演示视频讲稿.md`
- `deliverables/24281098-开发日志与沟通交流记录.md`
- `deliverables/24281098-真实运行截图说明.md`
- `deliverables/24281098-提交清单.md`
- `deliverables/screenshots/`：运行截图、真实识别截图和开发阶段截图。
- `deliverables/demo_vehicle_images/`：演示用车辆图片。
- `deliverables/北京交通大学计算机与信息技术学院实习实训日志-*.docx/pdf`：三名成员个人日志。

## 最终提交包

- `24281098.zip`：按组长学号命名的最终提交包。

压缩包包含源码、模型、系统数据、报告、PPT、演示视频、讲稿、日志、截图和提交清单；不包含 `datasets/` 原始数据集、`runs/` 训练缓存、上传缓存、临时 PPT 工作区和运行时输出图。

## 整理原则

- 保留：源码、模型、系统数据、课程材料、精选截图、训练脚本和说明文档。
- 本地保留但不打包：`datasets/`、`runs/`，方便继续训练和复核。
- 可清理：`__pycache__/`、`uploads/`、`deliverables/ppt_work/`、临时识别输出图、`.inspect.ndjson`。
