"""
综合性解决方案：解决浓度预测模型过拟合问题
结合数据增强、正则化、集成学习和超参数优化等多种技术
"""
import time
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
import joblib
import os
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from utils.augmentation import DataAugmentation
from utils.feature_extractor import extract_color_features
import xgboost as xgb


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


def apply_advanced_data_augmentation(roi, augmentation_prob=0.8):
    """
    应用高级数据增强到感兴趣区域

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


def prepare_enhanced_concentration_dataset(images_dir, annotations_dir, class_name_to_concentration):
    """
    准备增强的浓度预测数据集，使用更高级的数据增强技术

    Args:
        images_dir: 图像目录
        annotations_dir: 标注目录
        class_name_to_concentration: 类别名称到浓度值的映射

    Returns:
        X: 特征矩阵
        y: 浓度标签向量
    """
    X = []
    y = []

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

                # 应用多次数据增强以增加数据多样性
                for _ in range(2):  # 为每个样本创建2个增强版本
                    augmented_roi = apply_advanced_data_augmentation(roi)
                    augmented_features = extract_color_features(augmented_roi)
                    X.append(augmented_features)
                    y.append(ann['concentration'])  # 浓度值保持不变

    return np.array(X), np.array(y)


def create_regularized_models():
    """
    创建一组正则化良好的模型
    """
    # 高度正则化的XGBoost模型
    xgb_model = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1
    )
    
    # 限制复杂度的随机森林
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Ridge回归
    ridge_model = Ridge(alpha=15.0, random_state=42)
    
    # ElasticNet
    elastic_model = ElasticNet(alpha=0.2, l1_ratio=0.3, random_state=42)
    
    return xgb_model, rf_model, ridge_model, elastic_model


def train_comprehensive_solution():
    """
    训练综合性解决方案模型
    """
    print("开始准备增强数据集...")

    # 设置数据路径
    images_dir = "../data/JPEGImages"
    annotations_dir = "../data/Annotations"

    # 确保路径存在
    if not os.path.exists(images_dir):
        images_dir = "../../data/JPEGImages"
    if not os.path.exists(annotations_dir):
        annotations_dir = "../../data/Annotations"

    # 检查路径是否存在
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    if not os.path.exists(annotations_dir):
        print(f"错误: 标注目录不存在: {annotations_dir}")
        return

    # 定义类别名称到浓度值的映射
    class_name_to_concentration = {
        "0ug": 0,
        "40ug": 40,
        "80ug": 80,
        "120ug": 120,
        "160ug": 160,
        "200ug": 200
    }

    # 准备增强数据集
    X, y = prepare_enhanced_concentration_dataset(images_dir, annotations_dir, class_name_to_concentration)

    if len(X) == 0:
        print("错误: 没有加载到任何训练数据")
        print("请确保:")
        print("  1. ../data/JPEGImages 目录包含图像文件")
        print("  2. ../data/Annotations 目录包含对应的XML标注文件")
        print("  3. XML文件中的类别名称与预定义的类别名称匹配")
        return

    print(f"增强后数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征, 浓度范围: [{y.min():.2f}, {y.max():.2f}]")

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42  # 减少测试集大小以利用更多数据训练
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建正则化模型
    xgb_model, rf_model, ridge_model, elastic_model = create_regularized_models()

    print("开始训练正则化模型...")

    # 分别训练各个模型
    models = {
        'XGBoost': xgb_model,
        'RandomForest': rf_model,
        'Ridge': ridge_model,
        'ElasticNet': elastic_model
    }

    # 训练所有模型
    for name, model in models.items():
        print(f"训练 {name} 模型...")
        model.fit(X_train_scaled, y_train)

    # 创建集成模型
    ensemble = VotingRegressor([
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('ridge', ridge_model),
        ('elastic', elastic_model)
    ])

    print("训练集成模型...")
    ensemble.fit(X_train_scaled, y_train)

    # 对所有模型进行预测
    model_predictions = {}
    for name, model in models.items():
        model_predictions[name] = model.predict(X_test_scaled)

    # 集成模型预测
    ensemble_pred = ensemble.predict(X_test_scaled)

    print("使用综合性解决方案进行预测")

    # 评估各模型性能
    print("\n各模型性能评估:")
    for name, pred in model_predictions.items():
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

    # 评估集成模型性能
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

    print(f"集成模型 - MSE: {ensemble_mse:.4f}, R²: {ensemble_r2:.4f}, MAE: {ensemble_mae:.4f}")

    # 训练集上的性能（用于计算泛化误差）
    ensemble_train_pred = ensemble.predict(X_train_scaled)
    train_mse = mean_squared_error(y_train, ensemble_train_pred)
    train_r2 = r2_score(y_train, ensemble_train_pred)
    train_mae = mean_absolute_error(y_train, ensemble_train_pred)

    print(f"训练集 - MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")

    # 计算泛化误差（测试MSE - 训练MSE）
    generalization_error = ensemble_mse - train_mse
    print(f"泛化误差(MSE): {generalization_error:.4f}")

    # 进行交叉验证
    print("\n正在进行5折交叉验证评估...")
    cv_mse_scores = -cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f"5折交叉验证MSE分数: {cv_mse_scores}")
    print(f"平均MSE: {cv_mse_scores.mean():.4f} (+/- {cv_mse_scores.std() * 2:.4f})")
    print(f"5折交叉验证R²分数: {cv_r2_scores}")
    print(f"平均R²: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")

    # 显示一些预测示例
    print("\n测试集前10个样本的预测结果:")
    print("真实值 -> 集成预测值")
    for i in range(min(10, len(y_test))):
        print(f"{y_test[i]:.2f} -> {ensemble_pred[i]:.2f}")

    # 特别关注零浓度样本的预测效果
    zero_indices = np.where(y_test == 0)[0]
    if len(zero_indices) > 0:
        zero_predictions = ensemble_pred[zero_indices]
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
        y_test_pred_non_zero = ensemble_pred[non_zero_indices]
        non_zero_mse = mean_squared_error(y_test_non_zero, y_test_pred_non_zero)
        non_zero_mae = mean_absolute_error(y_test_non_zero, y_test_pred_non_zero)
        non_zero_r2 = r2_score(y_test_non_zero, y_test_pred_non_zero)
        print(f"\n非零浓度样本性能:")
        print(f"MSE: {non_zero_mse:.4f}, R²: {non_zero_r2:.4f}, MAE: {non_zero_mae:.4f}")

    # 保存最佳模型和标准化器
    model_path = "../models/comprehensive_solution_" + str(int(time.time())) + ".pkl"
    scaler_path = "../models/comprehensive_scaler_" + str(int(time.time())) + ".pkl"
    latest_model_path = "../models/latest_comprehensive.pkl"
    latest_scaler_path = "../models/latest_comprehensive_scaler.pkl"
    
    os.makedirs("../models", exist_ok=True)

    # 保存集成模型和标准化器
    joblib.dump(ensemble, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(ensemble, latest_model_path)
    joblib.dump(scaler, latest_scaler_path)

    print(f"\n模型已保存至: {model_path}")
    print(f"标准化器已保存至: {scaler_path}")
    print(f"最新模型已保存至: {latest_model_path}")
    print(f"最新标准化器已保存至: {latest_scaler_path}")

    # 与原始性能对比
    original_train_mse = 26.0587
    original_test_mse = 103.3948
    original_generalization_error = 77.3361

    print(f"\n性能改进对比:")
    print(f"训练MSE: 从 {original_train_mse:.4f} 到 {train_mse:.4f} (改进: {((original_train_mse-train_mse)/original_train_mse)*100:.2f}%)")
    print(f"测试MSE: 从 {original_test_mse:.4f} 到 {ensemble_mse:.4f} (改进: {((original_test_mse-ensemble_mse)/original_test_mse)*100:.2f}%)")
    print(f"泛化误差: 从 {original_generalization_error:.4f} 到 {generalization_error:.4f} (改进: {((original_generalization_error-abs(generalization_error))/original_generalization_error)*100:.2f}%)")

    # 检查是否显著改善了过拟合
    improvement_threshold = 0.5  # 期望至少改善50%
    mse_improvement = (original_test_mse - ensemble_mse) / original_test_mse
    generalization_improvement = (original_generalization_error - abs(generalization_error)) / original_generalization_error

    if mse_improvement > improvement_threshold and generalization_improvement > improvement_threshold:
        print(f"\n🎉 显著改进！MSE改善了 {mse_improvement*100:.2f}%，泛化误差改善了 {generalization_improvement*100:.2f}%")
    elif mse_improvement > 0.2:  # 至少改善20%
        print(f"\n✅ 改进成功！MSE改善了 {mse_improvement*100:.2f}%，泛化误差改善了 {generalization_improvement*100:.2f}%")
    else:
        print(f"\n⚠️  改进有限，可能需要进一步调整")

    return ensemble, scaler


if __name__ == "__main__":
    # 训练综合性解决方案
    model, scaler = train_comprehensive_solution()
    print("\n综合性解决方案训练完成!")
    print("\n该解决方案包含以下优化技术:")
    print("1. 高级数据增强 - 增加数据多样性")
    print("2. 模型正则化 - 防止过拟合")
    print("3. 集成学习 - 提高泛化能力")
    print("4. 特征标准化 - 改善模型收敛")
    print("5. 多模型融合 - 降低单一模型偏差")