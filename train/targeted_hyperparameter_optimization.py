"""
专门针对随机森林的超参数优化脚本
旨在找到最佳参数组合以实现MSE < 10的目标
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import os.path as osp
from utils.augmentation import DataAugmentation
from utils.feature_extractor import extract_color_features
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import randint, uniform


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
            
            # 尝试用PIL加载图像（用于与预测时的Image.open兼容）
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


def enforce_physical_constraints(predictions):
    """
    执行物理约束：浓度不能为负值
    
    Args:
        predictions: 预测值数组
        
    Returns:
        应用约束后的预测值数组
    """
    # 根据经验教训内存中的知识，浓度不能为负
    constrained_predictions = np.maximum(predictions, 0.0)
    return constrained_predictions


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)
    y_pred = enforce_physical_constraints(y_pred)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{model_name} - MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
    
    return mse, r2, mae


def detailed_hyperparameter_search():
    """
    详细的超参数搜索以找到最佳参数组合
    """
    print("开始准备数据集...")
    
    # 设置数据路径
    images_dir = "../data/JPEGImages"
    annotations_dir = "../data/Annotations"
    
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
    
    # 准备数据集，启用数据增强
    X, y = prepare_concentration_dataset(images_dir, annotations_dir, class_name_to_concentration, apply_augmentation=True)
    
    if len(X) == 0:
        print("错误: 没有加载到任何训练数据")
        return
    
    print(f"数据增强后数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征, 浓度范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 第一步：粗略网格搜索
    print("\n第一步：粗略网格搜索...")
    param_grid_coarse = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    grid_search_coarse = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_coarse,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_coarse.fit(X_train, y_train)
    
    print(f"粗略搜索最佳参数: {grid_search_coarse.best_params_}")
    print(f"粗略搜索最佳得分: {-grid_search_coarse.best_score_:.4f}")
    
    # 第二步：基于粗略搜索结果进行精细搜索
    print("\n第二步：精细网格搜索...")
    
    # 获取粗略搜索的最佳参数
    best_params = grid_search_coarse.best_params_
    
    # 在最佳参数周围定义更精细的搜索空间
    n_est_range = [max(100, best_params['n_estimators']-200), best_params['n_estimators'], min(1200, best_params['n_estimators']+200)]
    n_est_range = [x for x in n_est_range if x > 0]
    
    max_depth_range = [max(5, best_params['max_depth']-10) if best_params['max_depth'] is not None else 20, 
                       best_params['max_depth'], 
                       best_params['max_depth']+10 if best_params['max_depth'] is not None else 30]
    max_depth_range = [x if x is not None else None for x in max_depth_range]
    max_depth_range = [x for x in max_depth_range if x != 0]  # 移除0值
    
    # 确保max_depth_range不包含None和0
    refined_depth_range = []
    for d in max_depth_range:
        if d is not None and d > 0:
            refined_depth_range.append(d)
        elif d is None:
            refined_depth_range.append(None)
    if None not in refined_depth_range:
        refined_depth_range.extend([None])
    
    param_grid_fine = {
        'n_estimators': n_est_range,
        'max_depth': refined_depth_range,
        'min_samples_split': [max(1, best_params['min_samples_split']-2), 
                              best_params['min_samples_split'], 
                              best_params['min_samples_split']+2],
        'min_samples_leaf': [max(1, best_params['min_samples_leaf']-1), 
                             best_params['min_samples_leaf'], 
                             best_params['min_samples_leaf']+1],
        'max_features': ['sqrt', 'log2', 0.7]  # 添加max_features参数
    }
    
    # 清理参数范围
    param_grid_fine['n_estimators'] = [x for x in param_grid_fine['n_estimators'] if x >= 10]
    param_grid_fine['min_samples_split'] = [x for x in param_grid_fine['min_samples_split'] if x >= 1]
    param_grid_fine['min_samples_leaf'] = [x for x in param_grid_fine['min_samples_leaf'] if x >= 1]
    
    print(f"精细搜索参数范围: {param_grid_fine}")
    
    grid_search_fine = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid_fine,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_fine.fit(X_train, y_train)
    
    print(f"精细搜索最佳参数: {grid_search_fine.best_params_}")
    print(f"精细搜索最佳得分: {-grid_search_fine.best_score_:.4f}")
    
    # 第三步：使用随机搜索进一步优化
    print("\n第三步：随机搜索进一步优化...")
    
    # 基于前面的结果定义更大的搜索空间
    param_dist = {
        'n_estimators': randint(300, 1000),
        'max_depth': [None] + list(randint.rvs(10, 50, size=10)),
        'min_samples_split': randint(2, 15),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
        'bootstrap': [True, False],
        'min_impurity_decrease': uniform(0, 0.1)  # 添加这个参数
    }
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,  # 尝试50个随机组合
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"随机搜索最佳参数: {random_search.best_params_}")
    print(f"随机搜索最佳得分: {-random_search.best_score_:.4f}")
    
    # 选择最佳模型（可能是精细搜索或随机搜索的结果）
    if -grid_search_fine.best_score_ <= -random_search.best_score_:
        best_model = grid_search_fine.best_estimator_
        best_params_final = grid_search_fine.best_params_
        print(f"\n使用精细搜索的模型作为最佳模型")
    else:
        best_model = random_search.best_estimator_
        best_params_final = random_search.best_params_
        print(f"\n使用随机搜索的模型作为最佳模型")
    
    # 评估最佳模型
    print(f"\n评估最佳模型...")
    train_mse, train_r2, train_mae = evaluate_model(best_model, X_train, y_train, "最佳模型 - 训练集")
    test_mse, test_r2, test_mae = evaluate_model(best_model, X_test, y_test, "最佳模型 - 测试集")
    
    print(f"\n最终模型性能:")
    print(f"训练集 - MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"测试集 - MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
    
    # 检查是否达到了目标
    if test_mse < 10:
        print(f"\n🎉 恭喜！模型性能达到了目标，测试集MSE为 {test_mse:.4f} (< 10)")
    else:
        print(f"\n⚠️  模型性能尚未达到目标，测试集MSE为 {test_mse:.4f} (目标 < 10)")
        print(f"  改进程度: 从原始 ~197 降至 {test_mse:.4f}，改进了 {(197-test_mse)/197*100:.2f}%")
    
    # 显示测试集前10个样本的预测结果
    print(f"\n最佳模型在测试集上前10个样本的预测结果:")
    print("真实值 -> 预测值")
    y_test_pred = best_model.predict(X_test)
    y_test_pred = enforce_physical_constraints(y_test_pred)
    for i in range(min(10, len(y_test))):
        print(f"{y_test[i]:.2f} -> {y_test_pred[i]:.2f}")
    
    # 保存模型
    timestamp = int(time.time())
    model_path = f"../models/targeted_best_model_{timestamp}.pkl"
    latest_path = "../models/latest_targeted.pkl"
    
    os.makedirs("../models", exist_ok=True)
    joblib.dump(best_model, model_path)
    joblib.dump(best_model, latest_path)
    
    print(f"\n最佳模型已保存至: {model_path}")
    print(f"最新模型已保存至: {latest_path}")
    
    return best_model, best_params_final, test_mse


if __name__ == "__main__":
    print("开始针对性超参数优化...")
    best_model, best_params, final_mse = detailed_hyperparameter_search()
    
    print(f"\n优化完成！最终测试集MSE: {final_mse:.4f}")
    print(f"最佳参数: {best_params}")