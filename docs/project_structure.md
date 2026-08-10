# 项目结构说明

本项目已经按“可运行、可训练、可提交”的思路整理。根目录只保留系统源码、训练脚本、运行脚本、模型、数据配置、课程材料和最终压缩包。

## 运行系统

- `app.py`：Flask 非静态版入口。
- `templates/index.html`：企业车辆管理系统页面结构。
- `static/app.js`：前端交互、模块切换、识别结果展示、车辆档案查询。
- `static/style.css`：后台系统样式。
- `data/*.json`：车辆档案、通行记录、黑白名单和系统设置数据。
- `运行非静态版.bat`：双击启动本地系统。
- `static-demo.html`：单文件静态演示版。

## AI 与训练

- `detector.py`：检测调度，优先使用 YOLOv8 权重，其次 YOLOv5 ONNX、Tiny-YOLO、OpenCV。
- `ocr_engine.py`：OCR 调度，优先 PaddleOCR，其次 EasyOCR、Tesseract、数据集标注兜底。
- `tiny_yolo.py`：课程演示用轻量模型结构。
- `train_plate_model.py`：轻量模型训练脚本。
- `train_yolov8_plate.py`：推荐的 YOLOv8 车牌检测训练脚本。
- `prepare_combined_dataset.py`：合并多个 YOLO 格式数据集。
- `requirements.txt`：基础运行依赖。
- `requirements-ai.txt`：增强训练/OCR 依赖。
- `models/tiny_plate_detector.pt`：已训练的课程演示模型。

训练出更高准确率模型后，将 `best.pt` 放到 `models/best.pt` 或 `models/yolov8_plate.pt`，系统会自动优先加载。

## 本地数据与输出

- `datasets/`：本地原始数据集和合并数据集，保留用于训练，但不进入最终提交包。
- `outputs/charts/`：训练曲线。
- `outputs/samples/`：保留的测试样例图。
- `uploads/`：运行时上传缓存，已清理，程序启动后会自动创建。

## 课程提交材料

- `deliverables/24281098-课程报告-企业车牌识别与车辆管理系统.docx`
- `deliverables/24281098-课程报告-企业车牌识别与车辆管理系统.pdf`
- `deliverables/24281098-陈述PPT-企业车牌识别与车辆管理系统.pptx`
- `deliverables/24281098-功能演示视频-企业车牌识别与车辆管理系统.mp4`
- `deliverables/24281098-开发日志与沟通交流记录.md`
- `deliverables/24281098-演示视频讲稿.md`
- `deliverables/24281098-真实运行截图说明.md`
- `deliverables/24281098-提交清单.md`
- `deliverables/screenshots/`：运行截图和开发阶段截图。
- `deliverables/demo_vehicle_images/`：演示用车辆图片。
- `deliverables/北京交通大学计算机与信息技术学院实习实训日志-*.docx/pdf`：三名成员个人日志。

## 最终提交包

- `24281098.zip`：按组长学号命名的最终提交包。

压缩包包含源码、模型、系统数据、报告、PPT、演示视频、日志、截图和提交清单；不包含 `datasets/` 原始数据集和旧缓存。

## 已清理内容

- 旧版 PPT、旧版课程报告、重复命名材料。
- PPT/报告渲染预览目录。
- 临时日志、服务器启动残留、上传缓存、Python 缓存。
- 旧压缩包和临时打包目录。
- 只用于早期生成材料的草稿脚本。
