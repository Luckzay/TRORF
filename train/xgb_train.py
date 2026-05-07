"""
训练浓度预测模型的脚本
使用XGBoost回归算法预测溶液浓度
"""
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import sys
import os.path as osp
from utils.augmentation import DataAugmentation
from utils.feature_extractor import extract_color_features


def ensure_rgb_format(image):
    """
    确保图像为RGB格式，兼容PIL和OpenCV图像

    Args:
        image: PIL图像或numpy数组

    Returns:
        RGB格式的numpy数组
    """
    if isinstance(image, Image.Image):
        # PIL图像转numpy数组 (PIL使用RGB)
        return np.array(image)
    elif isinstance(image, np.ndarray):
        # OpenCV图像 (BGR) 转 RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 如果是灰度图，保持不变
            return image
    else:
        raise ValueError(f"不支持的图像类型: {type(image)}")





def apply_data_augmentation(roi, augmentation_prob=0.5):
    """
    应用数据增强到感兴趣区域

    Args:
        roi: 感兴趣区域
        augmentation_prob: 应用数据增强的概率

    Returns:
        增强后的ROI
    """
    if np.random.rand() > augmentation_prob:
        return roi  # 不应用增强，直接返回原图


    # 应用随机增强
    augmented_roi = DataAugmentation.apply_random_augmentation(roi)

    return augmented_roi


def parse_xml_annotation(xml_path, class_name_to_concentration):
    """
    解析XML标注文件

    Args:
        xml_path: XML文件路径
        class_name_to_concentration: 类别名称到浓度值的映射

    Returns:
        [{'bbox': [x1,y1,x2,y2], 'concentration': float}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    # 解析对象标注
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        # 获取浓度值
        concentration = class_name_to_concentration.get(class_name, 0)

        # 获取边界框坐标
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        annotations.append({
            'bbox': [xmin, ymin, xmax, ymax],
            'concentration': concentration
        })

    return annotations


def prepare_concentration_dataset(images_dir, annotations_dir, class_name_to_concentration, apply_augmentation=True):
    """
    准备浓度预测数据集

    Args:
        images_dir: 图像目录
        annotations_dir: 标注目录
        class_name_to_concentration: 类别名称到浓度值的映射
        apply_augmentation: 是否应用数据增强

    Returns:
        X: 特征矩阵
        y: 浓度标签向量
    """
    X = []
    y = []

    # 遵循项目信息内存中提到的数据路径配置
    print(f"从以下目录加载数据:")
    print(f"  图像目录: {images_dir}")
    print(f"  标注目录: {annotations_dir}")

    # 遍历标注文件
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            image_name = xml_file.replace('.xml', '.jpg')
            image_path = os.path.join(images_dir, image_name)

            # 检查图像文件是否存在，尝试其他扩展名
            if not os.path.exists(image_path):
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_image_path = os.path.join(images_dir, xml_file.replace('.xml', ext))
                    if os.path.exists(alt_image_path):
                        image_path = alt_image_path
                        break

            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                continue

            # 尝试用PIL加载图像（用于与预测时的Image.open兼容
            pil_image = Image.open(image_path)
            image = ensure_rgb_format(pil_image)

            # 解析标注
            try:
                annotations = parse_xml_annotation(xml_path, class_name_to_concentration)
            except ET.ParseError:
                print(f"警告: 无法解析XML文件: {xml_path}")
                continue
            # 提取每个标注区域的特征
            for ann in annotations:
                bbox = ann['bbox']
                x1, y1, x2, y2 = bbox

                # 确保坐标在图像范围内
                x1 = max(0, min(x1, image.shape[1]))
                y1 = max(0, min(y1, image.shape[0]))
                x2 = max(0, min(x2, image.shape[1]))
                y2 = max(0, min(y2, image.shape[0]))

                roi = image[y1:y2, x1:x2]

                # 确保ROI不为空
                if roi.size == 0:
                    continue

                # 提取特征
                features = extract_color_features(roi)

                # 添加到数据集
                X.append(features)
                y.append(ann['concentration'])

                # 如果启用数据增强，对ROI进行增强并提取特征
                if apply_augmentation:
                    # 应用数据增强
                    augmented_roi = apply_data_augmentation(roi)

                    # 提取增强后的特征
                    augmented_features = extract_color_features(augmented_roi)

                    # 添加到数据集
                    X.append(augmented_features)
                    y.append(ann['concentration'])  # 浓度值保持不变

    return np.array(X), np.array(y)


def train_concentration_regressor():
    """
    训练浓度预测模型
    """
    print("开始准备数据集...")

    # 设置数据路径 - 修复为正确的绝对路径
    images_dir = "../data/JPEGImages"  # 相对于train目录
    annotations_dir = "../data/Annotations"  # 相对于train目录

    # 确保路径存在
    if not os.path.exists(images_dir):
        images_dir = "../../data/JPEGImages"  # 如果在train子目录下运行
    if not os.path.exists(annotations_dir):
        annotations_dir = "../../data/Annotations"  # 如果在train子目录下运行

    # 检查路径是否存在
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    if not os.path.exists(annotations_dir):
        print(f"错误: 标注目录不存在: {annotations_dir}")
        return

    # 定义类别名称到浓度值的映射
    class_name_to_concentration = {
        "0": 0.000,
        "40ug": 0.074,
        "80ug": 0.155,
        "120ug": 0.241,
        "160ug": 0.306,
        "200ug": 0.395
    }

    # 准备数据集，启用数据增强
    X, y = prepare_concentration_dataset(images_dir, annotations_dir, class_name_to_concentration, apply_augmentation=True)

    if len(X) == 0:
        print("错误: 没有加载到任何训练数据")
        print("请确保:")
        print("  1. ../data/JPEGImages 目录包含图像文件")
        print("  2. ../data/Annotations 目录包含对应的XML标注文件")
        print("  3. XML文件中的类别名称与预定义的类别名称匹配")
        return

    print(f"数据增强后数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征, 浓度范围: [{y.min():.2f}, {y.max():.2f}]")

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 检查是否有GPU可用
    try:
        import xgboost as xgb
        print(f"XGBoost版本: {xgb.__version__}")

        # 检查XGBoost GPU支持
        try:
            # 尝试创建一个简单的GPU模型来验证GPU支持
            test_model = xgb.XGBRegressor(tree_method='gpu_hist', device='cuda', n_estimators=1)
            print("GPU加速可用，使用GPU进行训练...")
            tree_method = 'gpu_hist'
            predictor = 'gpu_predictor'
            device = 'cuda'
        except Exception as e:
            print(f"GPU加速不可用，使用CPU进行训练... 错误: {e}")
            tree_method = 'hist'
            predictor = None
            device = 'cpu'

    except ImportError:
        print("XGBoost未安装，无法训练模型")
        return

    # 定义优化的XGBoost模型
    print("开始训练优化的XGBoost回归模型...")

    # 使用优化的参数
    xgb_params = {
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 10,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': -1,
        'device': 'cuda',
        'tree_method': 'hist'
    }

    xgb_model = xgb.XGBRegressor(**xgb_params)

    class NonNegativeXGBRegressor:
        def __init__(self, base_regressor):
            self.base_regressor = base_regressor

        def fit(self, X, y):
            self.base_regressor.fit(X, y)
            return self

        def predict(self, X):
            raw_pred = self.base_regressor.predict(X)
            # 应用非负约束
            pred = np.maximum(raw_pred, 0)
            return pred

        def __getattr__(self, attr):
            # 代理其他属性和方法到基础回归器
            return getattr(self.base_regressor, attr)

    # 使用优化的XGBoost模型
    model = NonNegativeXGBRegressor(xgb_model)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("使用优化的XGBoost模型进行预测")

    # 计算评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n模型性能评估:")
    print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")

    # 计算泛化误差（训练MSE与测试MSE之差）
    generalization_error = test_mse - train_mse
    print(f"泛化误差(MSE): {generalization_error:.4f}")

    # 显示一些预测示例
    print("\n测试集前10个样本的预测结果:")
    print("真实值 -> 预测值")
    for i in range(min(10, len(y_test))):
        print(f"{y_test[i]:.3f} -> {y_test_pred[i]:.3f}")

    # 特别关注零浓度样本的预测效果
    zero_indices = np.where(y_test == 0)[0]
    if len(zero_indices) > 0:
        zero_predictions = y_test_pred[zero_indices]
        print(f"\n零浓度样本预测情况 (共{len(zero_indices)}个):")
        print(f"平均预测值: {np.mean(zero_predictions):.4f}, 标准差: {np.std(zero_predictions):.4f}")
        print(f"预测值范围: [{np.min(zero_predictions):.4f}, {np.max(zero_predictions):.4f}]")

        # 计算零浓度样本的MAE
        zero_mae = np.mean(np.abs(zero_predictions))
        print(f"零浓度样本MAE: {zero_mae:.4f}")

    # 计算非零浓度样本的误差
    non_zero_indices = np.where(y_test != 0)[0]
    if len(non_zero_indices) > 0:
        y_test_non_zero = y_test[non_zero_indices]
        y_test_pred_non_zero = y_test_pred[non_zero_indices]
        non_zero_mse = mean_squared_error(y_test_non_zero, y_test_pred_non_zero)
        non_zero_mae = mean_absolute_error(y_test_non_zero, y_test_pred_non_zero)
        non_zero_r2 = r2_score(y_test_non_zero, y_test_pred_non_zero)
        print(f"\n非零浓度样本性能:")
        print(f"MSE: {non_zero_mse:.4f}, R²: {non_zero_r2:.4f}, MAE: {non_zero_mae:.4f}")

    # 添加交叉验证评估
    from sklearn.model_selection import cross_val_score
    print("\n正在进行5折交叉验证评估...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"5折交叉验证R²分数: {cv_scores}")
    print(f"平均R²分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 保存模型 - 只保存基础模型以确保兼容性
    model_path = "../models/"+time.time().__str__()+".pkl"
    latest_path = "../models/latestFe.pkl"
    os.makedirs("../models", exist_ok=True)

    # 保存基础模型
    joblib.dump(model.base_regressor, model_path)
    joblib.dump(model.base_regressor, latest_path)

    print(f"\n模型已保存至: {model_path}and{latest_path}")

    # 特征重要性
    # 定义更详细的特征名称列表
    feature_names = [
        # RGB特征 (每通道7个统计量: 均值, 标准差, 中位数, 最大值, 最小值, 25%分位数, 75%分位数)
        'R_mean', 'R_std', 'R_median', 'R_max', 'R_min', 'R_p25', 'R_p75',
        'G_mean', 'G_std', 'G_median', 'G_max', 'G_min', 'G_p25', 'G_p75',
        'B_mean', 'B_std', 'B_median', 'B_max', 'B_min', 'B_p25', 'B_p75',

        # HSV特征
        'H_mean', 'H_std', 'H_median', 'H_max', 'H_min', 'H_p25', 'H_p75',
        'S_mean', 'S_std', 'S_median', 'S_max', 'S_min', 'S_p25', 'S_p75',
        'V_mean', 'V_std', 'V_median', 'V_max', 'V_min', 'V_p25', 'V_p75',

        # LAB特征
        'L_mean', 'L_std', 'L_median', 'L_max', 'L_min', 'L_p25', 'L_p75',
        'A_mean', 'A_std', 'A_median', 'A_max', 'A_min', 'A_p25', 'A_p75',
        'B_mean', 'B_std', 'B_median', 'B_max', 'B_min', 'B_p25', 'B_p75',

        # LUV特征
        'LUV_L_mean', 'LUV_L_std', 'LUV_L_median', 'LUV_L_max', 'LUV_L_min', 'LUV_L_p25', 'LUV_L_p75',
        'LUV_U_mean', 'LUV_U_std', 'LUV_U_median', 'LUV_U_max', 'LUV_U_min', 'LUV_U_p25', 'LUV_U_p75',
        'LUV_V_mean', 'LUV_V_std', 'LUV_V_median', 'LUV_V_max', 'LUV_V_min', 'LUV_V_p25', 'LUV_V_p75',

        # YCrCb特征
        'Y_mean', 'Y_std', 'Y_median', 'Y_max', 'Y_min', 'Y_p25', 'Y_p75',
        'Cr_mean', 'Cr_std', 'Cr_median', 'Cr_max', 'Cr_min', 'Cr_p25', 'Cr_p75',
        'Cb_mean', 'Cb_std', 'Cb_median', 'Cb_max', 'Cb_min', 'Cb_p25', 'Cb_p75',

        # 额外特征
        'Saturation', 'Color_Uniformity'
    ]

    # 获取特征重要性
    importances = model.feature_importances_

    print("\n特征重要性排序:")
    feature_importance = [(feature_names[i], importances[i]) for i in range(len(feature_names))]
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for name, importance in feature_importance[:6]:  # 只显示前6个最重要的特征
        print(f"{name}: {importance:.4f}")

    # 运行额外的特征分析
    print("\n开始进行详细特征重要性分析...")

    # 执行多种特征重要性分析
    try:
        # 1. 基于模型的特征重要性分析
        analyze_feature_importance(model, feature_names, top_n=15)

        # 2. 相关性分析
        correlation_analysis = calculate_correlation_analysis(X_train, y_train, feature_names, top_n=15)

        # 3. 排列重要性分析（使用较小的测试集以节省时间）
        if len(X_test) > 50:  # 如果测试集太大，取样分析
            sample_indices = np.random.choice(len(X_test), size=50, replace=False)
            X_test_sample = X_test[sample_indices]
            y_test_sample = y_test[sample_indices]
        else:
            X_test_sample = X_test
            y_test_sample = y_test

        permutation_analysis = permutation_importance_analysis(
            model, X_test_sample, y_test_sample, feature_names, n_repeats=3
        )

    except ImportError:
        print("注意: 缺少必要的可视化库 (matplotlib/seaborn)，跳过图形显示")


def load_and_predict_with_constraints(model_path):
    """
    加载模型并应用物理约束进行预测
    """
    loaded_model = joblib.load(model_path)

    def constrained_predict(X):
        raw_pred = loaded_model.predict(X)
        # 应用非负约束
        pred = np.maximum(raw_pred, 0)
        return pred

    return loaded_model, constrained_predict


def analyze_feature_importance(model, feature_names, top_n=20):
    """
    分析并可视化特征重要性

    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        top_n: 显示前N个最重要特征
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 获取特征重要性
    importances = model.feature_importances_

    # 创建特征重要性数据
    feature_importance = [(feature_names[i], importances[i]) for i in range(len(feature_names))]
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\n前 {top_n} 个最重要的特征:")
    print("=" * 40)
    for i, (name, importance) in enumerate(feature_importance[:top_n]):
        print(f"{i+1:2d}. {name:<20} : {importance:.6f}")

    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features, top_importances = zip(*feature_importance[:top_n])


    return feature_importance


def calculate_correlation_analysis(X, y, feature_names, top_n=10):
    """
    计算特征与目标变量的相关性分析

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称
        top_n: 显示前N个最相关的特征
    """
    from scipy.stats import pearsonr
    import numpy as np

    correlations = []
    for i in range(X.shape[1]):
        corr, _ = pearsonr(X[:, i], y)
        correlations.append(abs(corr))  # 使用绝对值表示相关强度

    # 创建相关性数据
    correlation_data = [(feature_names[i], correlations[i]) for i in range(len(feature_names))]
    correlation_data.sort(key=lambda x: x[1], reverse=True)

    print(f"\n前 {top_n} 个与目标变量相关性最高的特征 (Pearson 相关系数绝对值):")
    print("=" * 60)
    for i, (name, corr) in enumerate(correlation_data[:top_n]):
        print(f"{i+1:2d}. {name:<20} : {corr:.6f}")

    return correlation_data


def permutation_importance_analysis(model, X_test, y_test, feature_names, n_repeats=5):
    """
    使用排列重要性分析特征重要性

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        feature_names: 特征名称
        n_repeats: 重复次数
    """
    from sklearn.inspection import permutation_importance

    # 计算排列重要性
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)

    # 创建排列重要性数据
    perm_imp_data = [(feature_names[i], perm_importance.importances_mean[i])
                     for i in range(len(feature_names))]
    perm_imp_data.sort(key=lambda x: x[1], reverse=True)

    print(f"\n前10个排列重要性最高的特征:")
    print("=" * 40)
    for i, (name, imp) in enumerate(perm_imp_data[:10]):
        print(f"{i+1:2d}. {name:<20} : {imp:.6f}")


    return perm_imp_data


if __name__ == "__main__":
    # 训练模型
    model = train_concentration_regressor()