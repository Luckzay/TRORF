# TEM 图像分析代码块

本代码块完成 TEM 图像中微粒的自动检测、像素粒径换算为实际粒径、统计结果导出，以及粒径分布直方图生成。

## 目录结构

```text
2_TEM_Analysis/
├── tem_analysis.py              # 主程序
├── requirements.txt             # Python 依赖
├── model.pth                    # 示例模型/配置文件
├── test_tem.jpg                 # 示例 TEM 测试原图
├── README.md                    # 运行说明
└── demo_outputs/                # 示例运行输出
    ├── detected_particles.png
    ├── particle_size_histogram.png
    ├── particle_measurements.csv
    └── summary.json
```

## 环境安装

建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

## 快速运行

在本文件夹下执行：

```bash
python tem_analysis.py --input test_tem.jpg --model model.pth --output-dir outputs --scale-nm 100 --scale-px 100
```

含义：`--scale-nm 100 --scale-px 100` 表示比例尺为 `100 nm = 100 px`，因此 `1 px = 1 nm`。

## 输出文件

运行后会在 `outputs/` 中生成：

- `detected_particles.png`：带检测框与粒径标注的 TEM 原图。
- `particle_size_histogram.png`：粒径分布直方图。
- `particle_measurements.csv`：每个微粒的坐标、检测框尺寸、像素直径、实际直径等数据。
- `summary.json`：检测数量、均值、中位数、标准差、最大值、最小值等统计摘要。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--input` | `test_tem.jpg` | TEM 输入图像路径 |
| `--model` | `model.pth` | 模型或检测配置文件路径 |
| `--output-dir` | `outputs` | 输出文件夹 |
| `--scale-nm` | `100` | 比例尺实际长度，单位 nm |
| `--scale-px` | `100` | 比例尺像素长度，单位 px |
| `--min-area` | `30` | 最小微粒轮廓面积，过滤噪声 |
| `--max-area` | `10000` | 最大微粒轮廓面积，过滤聚团/伪影 |
| `--bright-particles` | 关闭 | 用于检测暗背景上的亮色微粒 |
| `--bins` | `20` | 直方图分箱数量 |

## 方法说明

当前交付版本采用 OpenCV 轮廓检测作为可直接运行的基线算法：

1. 读取 TEM 图像。
2. 灰度化、CLAHE 对比度增强、高斯滤波。
3. Otsu 阈值分割疑似微粒区域。
4. 轮廓检测并按面积、长宽比、圆度过滤噪声。
5. 对每个微粒计算检测框宽高均值作为等效像素直径：

   ```text
   diameter_px = (bbox_width_px + bbox_height_px) / 2
   diameter_nm = diameter_px × (scale_nm / scale_px)
   ```

6. 保存检测框图、粒径 CSV、统计 JSON 和粒径分布直方图。

## 接入真实模型的方式

如后续有标注数据和训练好的 YOLO/CNN 权重，可替换 `load_model()` 和 `detect_particles()` 中的基线检测逻辑：

- `load_model()`：加载真实模型权重。
- `detect_particles()`：调用模型推理，返回每个微粒的检测框 `(x, y, w, h)`。
- 后续粒径换算、统计和可视化部分无需改变。

## 注意事项

- `model.pth` 为示例基线配置文件，并非真实 TEM 数据训练得到的检测模型。
- 示例 `test_tem.jpg` 为合成 TEM 风格测试图，仅用于演示流程。
- 对真实 TEM 图像，建议根据图像分辨率、背景和微粒大小调整 `--min-area`、`--max-area`、`--scale-nm` 和 `--scale-px`。
