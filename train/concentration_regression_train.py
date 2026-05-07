"""
浓度回归模型训练脚本
用于训练SVR、随机森林、XGBoost等机器学习模型
"""
import os
import cv2
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from pathlib import Path
from config.config_manager import config
import xml.etree.ElementTree as ET

from utils.feature_extractor import extract_color_features


def parse_xml_annotation(xml_path, class_names, concentration_values):
    """
    解析XML标注文件
    
    Args:
        xml_path: XML文件路径
        class_names: 类别名称列表
        concentration_values: 浓度值列表
        
    Returns:
        [{'bbox': [x1,y1,x2,y2], 'class_id': int, 'concentration': float}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    
    # 获取图像尺寸
    size_elem = root.find('size')
    if size_elem is not None:
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
    else:
        # 如果XML中没有尺寸信息，需要通过图像文件获取
        width, height = None, None
    
    # 获取类别名称到索引的映射
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # 解析对象标注
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in name_to_idx:
            continue  # 跳过未知类别
        
        class_id = name_to_idx[class_name]
        
        # 获取边界框坐标
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        # 获取对应浓度值
        concentration = concentration_values[class_id]
        
        annotations.append({
            'bbox': [xmin, ymin, xmax, ymax],
            'class_id': class_id,
            'class_name': class_name,
            'concentration': concentration
        })
    
    return annotations


def load_dataset():
    """
    加载数据集并提取特征和标签
    
    Returns:
        X: 特征矩阵
        y: 浓度标签向量
    """
    # 获取配置
    annotations_dir = config.annotations_dir
    images_dir = config.images_dir
    class_names = config.class_names
    concentration_values = config.concentration_values
    
    X = []
    y = []
    
    # 遍历标注文件
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            image_name = xml_file.replace('.xml', '.jpg')
            image_path = os.path.join(images_dir, image_name)
            
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                # 尝试其他可能的图像扩展名
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_image_path = os.path.join(images_dir, xml_file.replace('.xml', ext))
                    if os.path.exists(alt_image_path):
                        image_path = alt_image_path
                        break
            
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                continue
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告: 无法读取图像: {image_path}")
                continue
            
            # 解析标注
            annotations = parse_xml_annotation(xml_path, class_names, concentration_values)
            
            # 提取每个标注区域的特征
            for ann in annotations:
                bbox = ann['bbox']
                x1, y1, x2, y2 = bbox
                roi = image[y1:y2, x1:x2]
                
                # 确保ROI不为空
                if roi.size == 0:
                    continue
                
                # 提取特征
                from utils.feature_extractor import extract_simple_color_features
                features = extract_color_features(roi)
                
                # 添加到数据集
                X.append(features)
                y.append(ann['concentration'])
    
    return np.array(X), np.array(y)


def train_concentration_regressor():
    """
    训练浓度回归模型
    """
    print("正在加载数据集...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("错误: 没有加载到任何训练数据")
        return
    
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征, 浓度范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建模型保存目录
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. 训练SVR模型
    print("\n正在训练SVR模型...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train)
    
    # 评估SVR模型
    y_pred_svr = svr_model.predict(X_test_scaled)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    print(f"SVR - MSE: {mse_svr:.4f}, R²: {r2_svr:.4f}")
    
    # 保存SVR模型和标准化器
    joblib.dump(svr_model, os.path.join(models_dir, "color_concentration_svr.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "color_concentration_scaler.pkl"))
    print("SVR模型已保存到 models/color_concentration_svr.pkl")
    print("标准化器已保存到 models/color_concentration_scaler.pkl")
    
    # 2. 训练随机森林模型
    print("\n正在训练随机森林模型...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 评估随机森林模型
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"随机森林 - MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")
    
    # 保存随机森林模型
    joblib.dump(rf_model, os.path.join(models_dir, "color_concentration_rf.pkl"))
    print("随机森林模型已保存到 models/color_concentration_rf.pkl")
    
    # 3. 训练XGBoost模型（如果安装了xgboost）
    try:
        import xgboost as xgb
        print("\n正在训练XGBoost模型...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        
        # 评估XGBoost模型
        y_pred_xgb = xgb_model.predict(X_test)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        r2_xgb = r2_score(y_test, y_pred_xgb)
        print(f"XGBoost - MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}")
        
        # 保存XGBoost模型
        joblib.dump(xgb_model, os.path.join(models_dir, "color_concentration_xgb.pkl"))
        print("XGBoost模型已保存到 models/color_concentration_xgb.pkl")
    except ImportError:
        print("\nXGBoost未安装，跳过XGBoost模型训练")
    
    print(f"\n所有模型已保存到 {models_dir} 目录")
    
    # 输出模型性能摘要
    print("\n模型性能摘要:")
    print(f"{'模型':<10} {'MSE':<10} {'R²':<10}")
    print(f"{'-'*30}")
    print(f"{'SVR':<10} {mse_svr:<10.4f} {r2_svr:<10.4f}")
    print(f"{'随机森林':<10} {mse_rf:<10.4f} {r2_rf:<10.4f}")
    if 'mse_xgb' in locals():
        print(f"{'XGBoost':<10} {mse_xgb:<10.4f} {r2_xgb:<10.4f}")


if __name__ == "__main__":
    print("浓度回归模型训练程序")
    print("="*50)
    train_concentration_regressor()
    print("="*50)
    print("训练完成!")