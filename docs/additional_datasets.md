# 国内车牌数据集与重新训练建议

## 当前问题判断

项目早期使用的演示数据主要来自国外/印尼车牌，适合证明“目标检测 + OCR + 车辆管理系统”的完整流程，但不适合作为中国蓝牌的主要训练数据。真实测试中出现的 `粤A78T3R`、`吉A10VE7`、`粤TM1314`、`京KBT355` 属于典型中国蓝牌，包含省份汉字、城市字母和蓝底白字结构，与原始训练域差异较大。

因此，正确的改进路线是：

1. 使用国内车牌数据集重新训练车牌检测模型。
2. 使用中文 OCR 引擎识别车牌字符。
3. 增加中国车牌格式校验和车辆档案模糊匹配。

当前代码已经完成第 2、3 步的工程接入，并新增了 CCPD 转换和训练脚本。

## 推荐数据集

### CCPD

推荐优先使用 CCPD 中国车牌数据集。

- 官方项目：`https://github.com/detectRecog/CCPD`
- Zenodo 数据归档：`https://zenodo.org/records/15647076`
- 可用文件：
  - `CCPD2020.zip`：约 0.85GB，建议先下载，适合课程项目快速训练。
  - `CCPD2019.tar.xz`：约 12.26GB，数据量更大，适合进一步提升泛化能力。

## 下载

查看可下载文件：

```bash
python download_ccpd_dataset.py --list
```

下载 CCPD2020：

```bash
python download_ccpd_dataset.py --contains CCPD2020 --target datasets/CCPD
```

如果网络较慢，可以让下载命令在晚上持续运行。

## 解压

把：

```text
datasets/CCPD/CCPD2020.zip
```

解压到：

```text
datasets/CCPD/CCPD2020
```

## 转换为 YOLOv8

```bash
python prepare_ccpd_dataset.py --source datasets/CCPD --output datasets/ccpd_yolo --limit 50000 --clean
```

说明：

- CCPD 的车牌框和车牌号写在文件名中，脚本会自动解析。
- 输出的 `labels/*.txt` 是 YOLO 检测标注。
- 输出的 `ocr_labels.csv` 保存真实车牌号，可用于后续 OCR 训练或测试。
- `--limit 50000` 是课程项目推荐设置，如果电脑性能较好可以改成 `--limit 0` 使用全部数据。

## 训练

CPU 训练：

```bash
python train_yolov8_plate.py --data datasets/ccpd_yolo/data.yaml --epochs 80 --imgsz 640 --batch 16 --device cpu --output models/yolov8_plate.pt
```

NVIDIA 显卡训练：

```bash
python train_yolov8_plate.py --data datasets/ccpd_yolo/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0 --output models/yolov8_plate.pt
```

训练完成后重启非静态版系统，后端会自动优先加载：

```text
models/yolov8_plate.pt
```

## 保留国外数据的用途

国外/印尼数据不建议作为国内蓝牌主训练集，但可以保留少量用于展示“系统具备跨域识别流程”。最终报告中应说明：

- 第一阶段使用公开国外车牌数据完成原型验证。
- 第二阶段发现真实国内蓝牌存在域偏移。
- 第三阶段引入 CCPD 国内数据集重新训练并增加车牌格式后处理。
