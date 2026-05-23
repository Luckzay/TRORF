# 3_Concentration · 溶液浓度检测

输入比色皿/样品管图像，**自动检测**溶液 ROI 并预测浓度。

采用两阶段 pipeline：
1. **YOLO 目标检测**（深度学习）—— 自动定位图像中每个比色皿，裁剪溶液 ROI、去除瓶壁干扰
2. **XGBoost 回归**（机器学习）—— 从 ROI 提取 107 维颜色特征，回归出连续浓度值

## 环境要求

- Python ≥ 3.10（推荐 3.12）
- 依赖见 `requirements.txt`

## 安装

```bash
pip install -r requirements.txt
```

⚠️ **必须用 PyPI 官方源安装**。国内常用的清华镜像（`pypi.tuna.tsinghua.edu.cn`）目前**缺 xgboost 3.x 的 Windows wheel**，pip 会回退装到 xgboost 2.x，**导致浓度回归输出错误**（恒定偏移约 +0.32）。

如果你的 pip 默认源是清华/阿里等国内镜像，请显式指定 PyPI 官方源：

```bash
pip install -r requirements.txt --index-url https://pypi.org/simple
```

验证 xgboost 装对了：

```bash
python -c "import xgboost; print(xgboost.__version__)"
# 期望输出：3.x.x（如 3.2.0）；如显示 2.x.x 则要重装
```

## 使用

```bash
# 基本用法：检测一张图，结果自动保存到 results/
python concentration_detect.py --image concentration_samples/20260120_161950_620.jpg

# 指定输出路径
python concentration_detect.py --image concentration_samples/20260120_161950_620.jpg --output my_result.jpg
```

## 输出

- **控制台**：每个检测到的比色皿打印一行 `浓度 top left bottom right`
  例如：
  ```
  0.011  133 87 350 222
  0.101  145 285 362 420
  ...
  ```
- **results/<原图名>.jpg**：在原图上叠加 YOLO 检测框 + 浓度数值的可视化图

## 性能

在 VOCFe 测试集（50 张图 / 273 个比色皿）上批量评估：

| 指标 | 数值 |
|---|---|
| MAE（平均绝对误差） | **0.0070** |
| RMSE（均方根误差） | **0.0141** |
| 检出率 | **100.4%**（274 检出 / 273 真值）|
| 单图推理时间 | GPU < 1s，CPU 约 5–15s |

浓度范围 0.002–0.338，MAE 0.0070 对应相对误差约 2–3%。

GPU 环境会自动启用 CUDA 加速（`torch.cuda.is_available()` 自动检测），无 GPU 时自动降级到 CPU，无需手动配置。

## 工作原理

```
输入图像
   │
   ▼ YOLO 检测（detection_model.pth）—— 自动定位每个比色皿，去除瓶壁干扰
检测框 + ROI
   │
   ▼ 107 维颜色特征
   │   （RGB / HSV / Lab / LUV / YCrCb 5 个颜色空间 × 3 通道 × 7 统计量 + 2 全局特征）
特征向量
   │
   ▼ XGBoost 回归（calibration_model.pkl，800 棵树）
连续浓度值
   │
   ▼ 在原图叠加检测框 + 浓度文字 → results/
```

## 项目结构

```
3_Concentration/
├── concentration_detect.py     # 主入口脚本
├── yolo.py                      # YOLO 推理引擎（检测 + 浓度回归整合）
├── model/                       # YOLO 网络结构
│   ├── yolo.py                  #   YoloBody
│   └── nets/                    #   backbone + 训练辅助
├── utils/                       # 特征提取、图像预处理、边界框解码
│   ├── feature_extractor.py     #   107 维颜色特征
│   ├── utils.py                 #   图像预处理工具
│   └── utils_bbox.py            #   YOLO 边界框解码 + NMS
├── model_data/
│   └── voc_classes_Fe-S.txt     # 类别定义（11 档浓度值）
├── detection_model.pth          # YOLO 检测模型权重（自动定位比色皿）
├── calibration_model.pkl        # XGBoost 浓度回归模型（107 维输入，800 树）
├── concentration_samples/       # 测试样本图
├── results/                     # 输出目录（运行后自动填充）
├── requirements.txt
└── README.md
```

## 限制

- 比色皿需在均匀光照下拍摄；强反光或阴影会降低准确率
- 模型针对铁离子（Fe）显色体系训练，其他显色体系需重新训练模型
- 浓度有效范围 0.002–0.338（训练数据范围），超出范围外推会有偏差
- 输入图像建议分辨率不低于 640×640
- 结果图标注使用 Pillow 内置字体
