# 浓度检测系统 (Concentration Detection System)

## 项目简介

本项目是一个基于**深度学习（YOLO）**和**机器学习**的比色皿溶液浓度检测系统。系统采用多任务学习策略，能够同时完成：
1. **比色皿目标检测** - 定位图像中的比色皿位置
2. **溶液浓度预测** - 预测比色皿中溶液的浓度值（支持离散分类和连续回归）

### 技术架构
- **深度学习框架**: PyTorch + Ultralytics YOLOv8
- **机器学习库**: scikit-learn, XGBoost
- **计算机视觉**: OpenCV, PIL
- **数据处理**: NumPy, Pandas

---

## 项目结构说明

```
root/
├── config/                    # 配置文件目录
├── data/                      # VOC格式原始数据集
├── model/                     # YOLO深度学习模型定义
├── models/                    # 训练好的模型文件存储
├── predict/                   # 预测推理模块
├── train/                     # 训练脚本目录
├── utils/                     # 工具函数库
├── checkpoints/               # 训练检查点
├── docs/                      # 文档
└── process_data/              # 数据处理中间文件
```

---

## 各模块详细说明

###  **config/** - 配置管理模块

**用途**: 集中管理系统的所有配置参数

**主要文件**:
- `config.json` - 主配置文件（JSON格式）
  - 数据路径配置
  - 模型参数（类别数、浓度值等）
  - 训练超参数（epochs、batch_size、学习率等）
  - GPU配置
  
- `config_manager.py` - 配置管理器
  - 提供统一的配置读取接口
  - 支持配置验证和默认值处理
  
- `data_config.yaml` - 数据配置文件（YAML格式）
  - YOLO训练所需的数据集配置

**可修改内容**: 
- 类别名称和浓度值映射
- 训练参数（epochs、batch_size、imgsz等）
- 数据路径和模型保存路径

---

###  **data/** - 数据集目录（VOC格式）

**用途**: 存放VOC格式的原始标注数据和图片

**目录结构**:
```
data/
├── Annotations/           # XML标注文件
│   ├── 000001.xml
│   └── ...
├── JPEGImages/            # 原始图片
│   ├── 000001.jpg
│   └── ...
└── ImageSets/Main/        # 数据集划分索引
    ├── train.txt          # 训练集图片列表
    ├── val.txt            # 验证集图片列表
    ├── test.txt           # 测试集图片列表
    └── trainval.txt       # 训练+验证集列表
```

**标注格式**: VOC XML格式，包含边界框坐标和类别标签

---

###  **model/** - YOLO深度学习模型定义 

**用途**: 定义YOLO目标检测网络结构和训练逻辑

**主要文件**:
- `yolo.py` - YOLO主模型类
  - 模型加载和初始化
  - 前向推理接口
  
- `yolo_model.py` - YOLO模型封装
  - 高层API接口
  - 预测结果后处理
  
- `nets/` - 网络组件
  - `backbone.py` - 特征提取主干网络（CSPDarknet）
  - `yolo_training.py` - YOLO训练相关功能
    - 损失函数计算（VFL、CIOU、BCE）
    - 训练循环逻辑
    - 锚框生成

**技术特点**:
- Anchor-Free检测头
- CSPDarknet53主干网络
- PANet特征融合
- 多尺度预测

---

###  **models/** - 预训练模型仓库 

**用途**: 存储所有训练好的模型文件

**包含两类模型**:

#### A. YOLO深度学习模型 (.pt, .pth)
- `yolov8n.pt` / `yolov8_s.pth` - YOLOv8基础检测模型
- `yolo26n.pt` - YOLOv26检测模型
- `yolo_weights.pth` - YOLO权重文件
- `ulmodel.pt` - 自定义UL模型
- `best_model_xgboost_*.pkl` - 最佳XGBoost模型快照

#### B. 机器学习回归模型 (.pkl)
**浓度预测模型**（基于颜色特征）:
- `color_concentration_svr.pkl` - SVR支持向量回归模型
- `color_concentration_rf.pkl` - 随机森林回归模型
- `color_concentration_xgb.pkl` - XGBoost回归模型
- `color_concentration_scaler.pkl` - 特征标准化器

**传统机器学习对比模型**:
- `Linear_Regression_comparison.pkl` - 线性回归
- `Ridge_Regression_comparison.pkl` - 岭回归
- `Lasso_Regression_comparison.pkl` - Lasso回归
- `Elastic_Net_comparison.pkl` - Elastic Net
- `Random_Forest_comparison.pkl` - 随机森林
- `Gradient_Boosting_comparison.pkl` - 梯度提升
- `XGBoost_comparison.pkl` - XGBoost
- `SVR_(RBF)_comparison.pkl` - SVR(RBF核)

**特定元素分析模型**:
- `Fe.pth` / `Fe-S.pkl` / `Fe-S.pth` - 铁元素检测模型
- `Fe-phen_2.pth` / `Fe-phen_3.pth` - 铁-酚试剂模型
- `Fe_42features.pkl` - 42维特征铁元素模型
- `latestFe.pkl` - 最新铁元素模型

**其他模型**:
- `brightness_shift.pkl` - 亮度偏移校正模型
- 时间戳命名的实验模型（如 `1770881639.5110922.pkl`）

---

###  **predict/** - 预测推理模块 

**用途**: 提供模型推理和结果可视化功能

**主要文件**:
- `yolo.py` - YOLO推理引擎（34.6KB）
  - 图像预处理
  - 模型推理
  - 边界框解码
  - NMS后处理
  - 结果可视化绘制
  
- `predict.py` - 高级预测接口（9.1KB）
  - 单张图像预测
  - 批量预测
  - 浓度值输出（离散+连续）
  - 结果保存
  
- `single_infer.py` - 单次推理示例（3.6KB）
  - 快速测试接口
  - 演示用法
  
- `result_conc.jpg` - 示例预测结果图

**推理流程**:
```
输入图像 → YOLO检测比色皿 → 裁剪ROI区域 → 
提取颜色特征(HSV) → 机器学习模型预测浓度 → 输出结果
```

---

###  **train/** - 训练脚本目录 

**用途**: 包含所有训练相关的脚本和日志

#### A. YOLO深度学习训练
- `train.py` (26.2KB) - YOLO主训练脚本
  - 完整的YOLO模型训练流程
  - 支持迁移学习
  - 自动保存检查点
  - TensorBoard日志记录
  
- `yolo26n.pt` - YOLOv26预训练权重

**训练输出** (`runs/concentration_detection/`):
```
runs/concentration_detection/
├── weights/
│   ├── best.pt        # 最佳模型
│   ├── last.pt        # 最新模型
│   ├── epoch0.pt      # 各epoch检查点
│   └── ...
├── results.csv        # 训练指标CSV
├── results.png        # 训练曲线图
├── confusion_matrix.png  # 混淆矩阵
├── BoxF1_curve.png    # F1分数曲线
├── BoxPR_curve.png    # 精确率-召回率曲线
├── labels.jpg         # 标签分布图
└── train_batch*.jpg   # 训练批次可视化
```

#### B. 机器学习浓度回归训练
- `xgb_train.py` (18.9KB) - XGBoost浓度回归训练
  - 从XML标注提取ROI区域
  - 颜色特征提取（HSV空间）
  - 数据增强（旋转、翻转、色彩抖动）
  - XGBoost模型训练和评估
  - 交叉验证和超参数调优
  
- `concentration_regression_train.py` (8.1KB) - 多模型对比训练
  - 同时训练SVR、随机森林、XGBoost
  - 特征标准化处理
  - 模型性能对比（MSE、R²）
  - 自动保存最优模型
  
- `comprehensive_solution.py` (15.6KB) - 综合解决方案
  - 集成多种算法
  - 自动化训练流程
  - 结果分析和报告生成
  
- `targeted_hyperparameter_optimization.py` (15.6KB) - 超参数优化
  - 网格搜索/随机搜索
  - 贝叶斯优化
  - 自动化超参数调优

**训练输出** (`train/models/`):
- `color_concentration_svr.pkl` - 训练好的SVR模型
- `color_concentration_rf.pkl` - 训练好的随机森林模型
- `color_concentration_xgb.pkl` - 训练好的XGBoost模型
- `color_concentration_scaler.pkl` - 特征标准化器

**训练日志** (`train/logs/`):
- TensorBoard事件文件
- 损失曲线记录
- 训练指标监控

---

###  **utils/** - 工具函数库 

**用途**: 提供通用工具函数和辅助模块

**主要文件**:

#### 数据处理工具
- `dataloader.py` (18.3KB) - 数据加载器
  - VOC格式数据读取
  - YOLO格式转换
  - 数据批处理
  - 多线程数据加载
  
- `voc_to_yolo_converter.py` (10.0KB) - VOC转YOLO转换器
  - XML标注解析
  - 生成ImageSets索引文件
  - 创建训练/验证标注文件
  - 数据统计和可视化
  - **详细使用说明见文件顶部注释**

#### 特征工程
- `feature_extractor.py` (7.9KB) - 特征提取器
  - RGB颜色空间特征
  - HSV颜色空间特征（色调、饱和度、明度）
  - 统计特征（均值、标准差、直方图）
  - 纹理特征
  
- `augmentation.py` (11.7KB) - 数据增强模块
  - 几何变换（旋转、翻转、缩放）
  - 色彩增强（亮度、对比度、饱和度调整）
  - 噪声注入
  - 随机裁剪
  - **详见 `数据增强方法说明.md`**

#### YOLO工具
- `utils_bbox.py` (17.4KB) - 边界框工具
  - 边界框编解码
  - IoU计算（IoU、GIoU、CIoU）
  - NMS非极大值抑制
  - 坐标转换
  
- `utils_fit.py` (4.7KB) - 训练辅助工具
  - 损失计算
  - 指标统计
  - 训练进度显示
  
- `callbacks.py` (3.8KB) - 回调函数
  - 早停机制
  - 学习率调度
  - 模型保存触发
  
- `checkpoint_manager.py` (3.8KB) - 检查点管理
  - 模型保存和加载
  - 最佳模型跟踪
  - 检查点清理

#### 通用工具
- `utils.py` (3.5KB) - 通用工具函数
  - 类别文件读取
  - 颜色生成
  - 文件路径处理

#### 文档
- `数据增强方法说明.md` - 数据增强技术详细说明

---

###  **checkpoints/** - 训练检查点

**用途**: 临时存储训练过程中的模型检查点

---

###  **docs/** - 项目文档

**用途**: 存放项目相关文档和技术资料

---

###  **process_data/** - 数据处理中间文件

**用途**: 存放数据预处理过程中的临时文件

---

##  算法说明

### 深度学习部分（YOLO）

**目标**: 检测图像中的比色皿位置

**网络架构**:
```
输入图像(416×416×3)
    ↓
Backbone: CSPDarknet53 (特征提取)
    ↓
Neck: PANet (多尺度特征融合)
    ↓
Head: Decoupled Head (分类+回归)
    ↓
输出: 边界框 + 置信度 + 类别概率
```

**损失函数**:
- 分类损失: VFL (Varifocal Loss)
- 定位损失: CIOU Loss
- 置信度损失: BCE Loss

**输出类别**: 6个浓度等级
- 0ug, 40ug, 80ug, 120ug, 160ug, 200ug

---

### 机器学习部分（浓度回归）

**目标**: 基于颜色特征预测连续浓度值

**工作流程**:
```
YOLO检测比色皿 → 裁剪ROI区域 → 提取HSV颜色特征 → 
机器学习模型回归 → 连续浓度值
```

**特征提取**:
- H (Hue): 色调平均值
- S (Saturation): 饱和度平均值
- V (Value): 明度平均值
- 统计特征: 标准差、偏度、峰度
- 直方图特征: 颜色分布

**回归算法对比**:
1. **SVR (Support Vector Regression)**
   - 核函数: RBF
   - 优点: 高维空间表现好
   - 适用: 小样本数据

2. **Random Forest**
   - 树数量: 100
   - 优点: 抗过拟合，可解释性强
   - 适用: 中等规模数据

3. **XGBoost** ⭐推荐
   - 梯度提升树
   - 优点: 精度高，支持正则化
   - 适用: 各种规模数据

**数据增强策略**:
- 随机旋转 (±15°)
- 水平/垂直翻转
- 色彩抖动 (亮度、对比度、饱和度)
- 高斯噪声
- 随机裁剪

---

##  快速开始

### 环境要求

```bash
# Python版本: >= 3.7
# 依赖安装
pip install -r requirements.txt
```

**核心依赖**:
- torch >= 1.9.0
- torchvision >= 0.10.0
- ultralytics == 8.2.83
- opencv-python
- scikit-learn
- xgboost
- Pillow
- numpy
- matplotlib

---

### 使用流程

#### 1. 准备数据集

将VOC格式数据集放入 `data/` 目录：
```
data/
├── Annotations/*.xml
├── JPEGImages/*.jpg
```

运行转换器生成训练文件：
```bash
python utils/voc_to_yolo_converter.py
```

#### 2. 配置参数

编辑 `config/config.json`，设置：
- 类别名称和浓度值
- 训练超参数
- 数据路径

#### 3. 训练YOLO检测模型

```bash
python train/train.py
```

训练结果保存在 `train/runs/concentration_detection/`

#### 4. 训练浓度回归模型

**XGBoost模型**（推荐）:
```bash
python train/xgb_train.py
```

**多模型对比训练**:
```bash
python train/concentration_regression_train.py
```

#### 5. 预测推理

**单张图像预测**:
```bash
python predict/single_infer.py --image path/to/image.jpg
```

**批量预测**:
```bash
python predict/predict.py --dir path/to/images
```

---

## 训练模型基本步骤

### 1. 构建数据集

使用 LabelImg 工具对图像进行标注：
- 安装 LabelImg: `pip install labelimg`
- 运行 LabelImg: `labelimg`
- 选择 VOC 格式进行标注
- 标注完成后生成 XML 格式的标注文件

### 2. 组织数据目录结构

按照以下结构存放图像和标签文件：

```
VOCdevkit/
└── VOC2007/
    ├── Annotations/          # XML标注文件目录
    │   ├── 000001.xml
    │   ├── 000002.xml
    │   └── ...
    ├── JPEGImages/           # 图片文件目录
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── ImageSets/
        └── Main/             # 生成的索引文件目录
            ├── train.txt     # 训练集图片名称列表
            ├── val.txt       # 验证集图片名称列表
            ├── test.txt      # 测试集图片名称列表
            └── trainval.txt  # 训练+验证集图片名称列表
```

将标注好的数据放入 `data/` 目录下对应的位置。

### 3. 生成索引文件

运行 VOC 转 YOLO 转换器生成训练所需的索引文件：

```bash
python utils/voc_to_yolo_converter.py
```

此脚本会生成以下文件：
- `2007_train.txt`: 训练集标注文件
- `2007_val.txt`: 验证集标注文件

每行格式为：图片路径 x1,y1,x2,y2,类别id x1,y1,x2,y2,类别id ...

### 4. 配置类别标签

修改 `train/model_data/voc_classes.txt` 文件，设置您的类别标签名称，确保与标注时使用的类别一致。

### 5. 训练目标检测模型

使用 YOLO 进行目标检测模型训练：

```bash
python train/train.py
```

训练结果保存在 `train/runs/concentration_detection/` 目录下。

### 6. 训练浓度回归模型

使用机器学习方法训练浓度预测模型：

```bash
python train/concentration_regression_train.py
```

或者使用 XGBoost 专门训练：

```bash
python train/xgb_train.py
```

训练好的模型保存在 `train/models/` 目录下。

### 7. 模型预测应用

使用训练好的模型进行预测：

**单张图像预测:**
```bash
python predict/predict.py --image path/to/image.jpg
```

**批量预测:**
```bash
python predict/predict.py --dir path/to/images
```

预测结果将包含比色皿位置检测和浓度值预测。

---

##  模型性能指标

### YOLO检测模型
- mAP@0.5: 目标检测平均精度
- Precision: 精确率
- Recall: 召回率
- F1 Score: F1分数

### 浓度回归模型
- MSE (Mean Squared Error): 均方误差
- RMSE (Root MSE): 均方根误差
- MAE (Mean Absolute Error): 平均绝对误差
- R² Score: 决定系数

查看训练曲线和混淆矩阵：
```
train/runs/concentration_detection/results.png
train/runs/concentration_detection/confusion_matrix.png
```

---

##  配置说明

### 关键配置项 (`config/config.json`)

```json
{
  "model": {
    "num_classes": 6,
    "class_names": ["0ug", "40ug", "80ug", "120ug", "160ug", "200ug"],
    "concentration_values": [0.0, 40.0, 80.0, 120.0, 160.0, 200.0]
  },
  "training": {
    "epochs": 20,
    "imgsz": 416,
    "batch_size": 4,
    "learning_rate": 0.001
  },
  "gpu": {
    "use_gpu": true,
    "device_id": 0
  }
}
```

---

##  注意事项

1. **路径规范**: 确保所有路径不包含空格和中文字符
2. **类别一致性**: `classes_path` 必须与训练/预测时使用的类别文件一致
3. **数据量建议**: 训练集建议大于500张图片，否则需增加训练轮数
4. **GPU显存**: 根据显存大小调整 `batch_size`
5. **模型选择**: 
   - 离散浓度分类 → 使用YOLO模型
   - 连续浓度回归 → 使用XGBoost/SVR/RF模型


---

##  相关文档

- [算法详细介绍](ALGORITHM.md)
- [数据增强方法说明](utils/数据增强方法说明.md)
- [VOC转YOLO转换器说明](utils/voc_to_yolo_converter.py)（见文件顶部注释）

---

##  贡献指南

欢迎提交Issue和Pull Request！

---

##  许可证

本项目仅供学习和研究使用。

---

##  联系方式

如有问题，联系邮箱1617022583@qq.com。

---
