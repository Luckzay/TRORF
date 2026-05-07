from ultralytics import YOLO
import torch
import os
import numpy as np
import cv2
from config.config_manager import config


class ConcentrationDetectionModel:
    """
    基于YOLOv8的浓度检测模型
    同时完成比色皿检测和浓度预测任务
    """
    
    def __init__(self, model_type=None):
        """
        初始化模型
        
        Args:
            model_type: YOLO模型类型 ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
        """
        # 使用配置文件中的模型类型
        model_type = model_type or config.model_type
        
        # 检查GPU可用性
        self.device = self._get_device()
        
        # 检查模型文件是否存在 - 支持多种路径格式
        import os
        from pathlib import Path
        
        print(f"DEBUG: 配置的模型路径: {model_type}")
        
        # 如果是相对路径，尝试从项目根目录查找
        model_path = Path(model_type)
        
        print(f"DEBUG: 解析的模型路径对象: {model_path}")
        print(f"DEBUG: 模型路径是否为绝对路径: {model_path.is_absolute()}")
        
        # 如果模型路径存在本地文件，则使用本地文件；否则让YOLO库自动下载
        resolved_model_path = None
        
        if model_path.is_absolute() and model_path.exists():
            # 绝对路径存在
            resolved_model_path = str(model_path)
            print(f"DEBUG: 使用绝对路径: {resolved_model_path}")
        elif model_path.exists():
            # 相对路径存在（当前工作目录下）
            resolved_model_path = str(model_path)
            print(f"DEBUG: 使用当前目录下的相对路径: {resolved_model_path}")
        else:
            print(f"DEBUG: 路径 {model_path} 在当前目录下不存在")
            # 检查是否为项目根目录下的相对路径
            project_root = Path(__file__).resolve().parent.parent  # 获取项目根目录
            print(f"DEBUG: 项目根目录: {project_root}")
            
            full_model_path = project_root / model_type
            print(f"DEBUG: 尝试项目根目录下的完整路径: {full_model_path}")
            print(f"DEBUG: 该路径是否存在: {full_model_path.exists()}")
            
            if full_model_path.exists():
                resolved_model_path = str(full_model_path)
                print(f"DEBUG: 使用项目根目录下的路径: {resolved_model_path}")
            else:
                # 最后尝试在model子目录下查找
                model_subdir_path = project_root / "model" / model_path.name
                print(f"DEBUG: 尝试model子目录下的路径: {model_subdir_path}")
                print(f"DEBUG: 该路径是否存在: {model_subdir_path.exists()}")
                
                if model_subdir_path.exists():
                    resolved_model_path = str(model_subdir_path)
                    print(f"DEBUG: 使用model子目录下的路径: {resolved_model_path}")
        
        if resolved_model_path:
            # 使用本地模型文件
            print(f"使用本地模型文件: {resolved_model_path}")
            self.model = YOLO(resolved_model_path)
        else:
            # 让YOLO库处理模型下载（这可能会导致重新下载）
            print(f"未找到本地模型文件，使用模型标识符: {model_type}")
            self.model = YOLO(model_type)
        
        # 将模型移动到指定设备
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"警告: 无法将模型移动到设备 {self.device}: {e}")
            print("回退到CPU设备")
            self.device = torch.device('cpu')
            self.model.to(self.device)
        
        # 定义类别
        self.class_names = config.class_names
        
        # 浓度映射
        self.concentration_map = dict(zip(self.class_names, config.concentration_values))
        
        # 初始化机器学习模型
        self.ml_models = {}
        
    def _get_device(self):
        """
        获取计算设备
        """
        use_gpu = config.get('gpu.use_gpu', True)
        
        if use_gpu and torch.cuda.is_available():
            device_id = config.get('gpu.device_id', 0)
            try:
                device = torch.device(f'cuda:{device_id}')
                # 测试GPU是否可用
                test_tensor = torch.zeros(1).to(device)
                del test_tensor  # 立即删除测试张量
                torch.cuda.empty_cache()  # 清空CUDA缓存
                print(f"使用GPU设备: {device}")
                return device
            except Exception as e:
                print(f"GPU不可用: {e}")
                print("回退到CPU设备")
                return torch.device('cpu')
        else:
            device = torch.device('cpu')
            print("使用CPU设备")
        
        return device
    
    def train(self, data_yaml_path=None, epochs=None, imgsz=None, batch_size=None):
        """
        训练模型
        
        Args:
            data_yaml_path: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 图像尺寸
            batch_size: 批次大小
        """
        # 使用配置文件中的默认值
        data_yaml_path = data_yaml_path or config.get('paths.data_config_path')
        epochs = epochs or config.training_epochs
        imgsz = imgsz or config.imgsz
        batch_size = batch_size or config.batch_size
        
        # 根据GPU配置调整批次大小
        if config.get('gpu.use_gpu', True) and self.device.type == 'cuda':
            gpu_multiplier = config.get('gpu.batch_size_multiplier', 2)
            # 对于RTX 3060，使用配置的批次大小，但确保不超过安全范围
            batch_size = min(batch_size, 16)  # 限制最大批次大小
        else:
            batch_size = max(1, batch_size // 4)  # CPU模式下使用更小的批次
        
        print(f"使用批次大小: {batch_size}, 图像尺寸: {imgsz}, 设备: {self.device}")
        
        # 开始训练
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,  # 指定设备
            name='concentration_detection',
            cache=False,  # 禁用缓存以节省内存
            workers=min(4, os.cpu_count()),  # 适度减少工作进程数
            project='runs',  # 指定项目目录
            exist_ok=True,  # 如果目录存在则覆盖
            save_period=10  # 每10个epoch保存一次
        )
        
        return results
    
    def predict(self, source, conf_threshold=None):
        """
        进行预测
        
        Args:
            source: 预测源（图像路径、视频路径或图像数组）
            conf_threshold: 置信度阈值
            
        Returns:
            包含检测框和浓度预测的结果
        """
        # 使用配置文件中的默认置信度阈值
        conf_threshold = conf_threshold or config.get('inference.conf_threshold', 0.5)
        
        # 在预测时也指定设备
        results = self.model(source, conf=conf_threshold, device=self.device)
        
        processed_results = []
        for r in results:
            # 获取检测框信息
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else []  # x1, y1, x2, y2
            confidences = r.boxes.conf.cpu().numpy() if r.boxes else []
            class_ids = r.boxes.cls.cpu().numpy() if r.boxes else []
            
            # 处理每个检测到的对象
            detections = []
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[int(cls_id)]
                concentration = self.concentration_map.get(class_name, 0.0)
                
                detection = {
                    'bbox': box.tolist(),  # 边界框 [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class_name': class_name,
                    'concentration': concentration  # 浓度值
                }
                detections.append(detection)
            
            processed_results.append({
                'detections': detections,
                'original_shape': r.orig_img.shape
            })
        
        return processed_results
    
    def predict_continuous_concentration(self, source, conf_threshold=None, method='rf'):
        """
        预测连续浓度值（使用多种机器学习方法）
        
        Args:
            source: 预测源
            conf_threshold: 置信度阈值
            method: 浓度预测方法 ('simple', 'svr', 'rf', 'mlp', 'xgb', 'hybrid')
            测试日志：2026.1.22 ：rf具有极佳的实践效果
        Returns:
            包含检测框和连续浓度预测的结果
        """
        # 使用配置文件中的默认置信度阈值
        conf_threshold = conf_threshold or config.get('inference.conf_threshold', 0.5)
        
        # 在预测时也指定设备
        results = self.model(source, conf=conf_threshold, device=self.device)
        print("use", method, "for regression predict")
        processed_results = []
        for r in results:
            # 获取原始图像
            orig_img = r.orig_img
            
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else []
            confidences = r.boxes.conf.cpu().numpy() if r.boxes else []
            class_ids = r.boxes.cls.cpu().numpy() if r.boxes else []
            
            detections = []
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[int(cls_id)]
                
                # 提取检测框内的图像区域
                x1, y1, x2, y2 = map(int, box)
                roi = orig_img[y1:y2, x1:x2]
                print("roi", roi)
                
                # 根据指定方法预测连续浓度值
                if method == 'svr':
                    continuous_concentration = self._analyze_color_concentration_svr(roi, class_name)
                elif method == 'rf':
                    continuous_concentration = self._analyze_color_concentration_rf(roi, class_name)
                elif method == 'mlp':
                    continuous_concentration = self._analyze_color_concentration_mlp(roi, class_name)
                elif method == 'xgb':
                    continuous_concentration = self._analyze_color_concentration_xgb(roi, class_name)
                elif method == 'hybrid':
                    continuous_concentration = self._analyze_color_concentration_hybrid(roi, class_name)
                else:  # simple
                    continuous_concentration = self._analyze_color_concentration(roi, class_name)
                print(continuous_concentration)
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_name': class_name,
                    'discrete_concentration': self.concentration_map.get(class_name, 0.0),
                    'continuous_concentration': continuous_concentration
                }
                detections.append(detection)
            
            processed_results.append({
                'detections': detections,
                'original_shape': r.orig_img.shape
            })
        
        return processed_results
    

    
    def _load_ml_model(self, model_name):
        """
        加载预训练的机器学习模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            加载的模型对象
        """
        import joblib
        import os
        
        model_path = f"models/{model_name}.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    
    def _analyze_color_concentration(self, roi, base_class_name):
        """
        通过颜色分析预测连续浓度值（原有方法）
        
        Args:
            roi: 感兴趣区域（检测到的比色皿区域）
            base_class_name: 基础类别名称（用于参考）
            
        Returns:
            预测的连续浓度值
        """
        # 转换为HSV颜色空间（更适合颜色分析）
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # 计算平均颜色值
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # 基于经验公式估算连续浓度
        # 这里使用简化的线性模型，实际应用中可能需要更复杂的模型
        base_concentration = self.concentration_map[base_class_name]
        
        # 基于饱和度和亮度的变化调整浓度预测
        # 注意：这只是一个示例算法，实际应用中需要根据具体数据校准
        saturation_factor = s_mean / 100.0  # 归一化饱和度
        brightness_factor = (v_mean - 50) / 100.0  # 归一化亮度（偏移50以中心化）
        
        # 简单的浓度调整（实际应用中需要基于训练数据建立更精确的模型）
        adjusted_concentration = base_concentration * (0.8 + 0.4 * saturation_factor)
        
        return float(adjusted_concentration)
    
    def _analyze_color_concentration_svr(self, roi, base_class_name):
        """
        使用支持向量回归预测连续浓度值
        
        Args:
            roi: 感兴趣区域（检测到的比色皿区域）
            base_class_name: 基础类别名称（用于参考）
            
        Returns:
            预测的连续浓度值
        """
        try:
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler
            import joblib
            import os
            
            # 提取颜色特征
            from utils.feature_extractor import extract_simple_color_features
            features = extract_simple_color_features(roi)
            
            # 尝试加载预训练的SVR模型
            model_path = "models/color_concentration_svr.pkl"
            scaler_path = "models/color_concentration_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                # 加载预训练模型和标准化器
                svr_model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # 标准化特征
                features_scaled = scaler.transform([features])
                
                # 预测
                concentration = svr_model.predict(features_scaled)[0]
            else:
                # 如果没有预训练模型，使用基础方法
                base_concentration = self.concentration_map[base_class_name]
                concentration = base_concentration  # 返回基础浓度作为默认值
                
        except ImportError:
            # 如果sklearn不可用，使用基础方法
            base_concentration = self.concentration_map[base_class_name]
            concentration = base_concentration
            
        return float(concentration)
    
    def _analyze_color_concentration_rf(self, roi, base_class_name):
        """
        使用随机森林回归预测连续浓度值
        
        Args:
            roi: 感兴趣区域（检测到的比色皿区域）
            base_class_name: 基础类别名称（用于参考）
            
        Returns:
            预测的连续浓度值
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            import joblib
            import os
            
            # 提取颜色特征
            from utils.feature_extractor import extract_simple_color_features
            features = extract_simple_color_features(roi)
            
            # 尝试加载预训练的RF模型
            model_path = "models/color_concentration_rf.pkl"
            if os.path.exists(model_path):
                # 加载预训练模型
                rf_model = joblib.load(model_path)
                
                # 预测
                concentration = rf_model.predict([features])[0]
            else:
                # 如果没有预训练模型，使用基础方法
                base_concentration = self.concentration_map[base_class_name]
                concentration = base_concentration  # 返回基础浓度作为默认值
                
        except ImportError:
            # 如果sklearn不可用，使用基础方法
            base_concentration = self.concentration_map[base_class_name]
            concentration = base_concentration
            
        return float(concentration)
    
    def _analyze_color_concentration_mlp(self, roi, base_class_name):
        """
        使用多层感知机预测连续浓度值
        
        Args:
            roi: 感兴趣区域（检测到的比色皿区域）
            base_class_name: 基础类别名称（用于参考）
            
        Returns:
            预测的连续浓度值
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            import numpy as np
            import os
            
            # 提取颜色特征
            from utils.feature_extractor import extract_simple_color_features
            features = extract_simple_color_features(roi)
            
            # 尝试加载预训练的MLP模型
            model_path = "models/color_concentration_mlp.h5"
            if os.path.exists(model_path):
                # 加载预训练模型
                mlp_model = load_model(model_path)
                
                # 预测
                concentration = mlp_model.predict(np.array([features]), verbose=0)[0][0]
            else:
                # 如果没有预训练模型，使用基础方法
                base_concentration = self.concentration_map[base_class_name]
                concentration = base_concentration  # 返回基础浓度作为默认值
                
        except ImportError:
            # 如果tensorflow不可用，使用基础方法
            base_concentration = self.concentration_map[base_class_name]
            concentration = base_concentration
            
        return float(concentration)
    
    def _analyze_color_concentration_xgb(self, roi, base_class_name):
        """
        使用XGBoost回归预测连续浓度值
        
        Args:
            roi: 感兴趣区域（检测到的比色皿区域）
            base_class_name: 基础类别名称（用于参考）
            
        Returns:
            预测的连续浓度值
        """
        try:
            import xgboost as xgb
            import joblib
            import os
            
            # 提取颜色特征
            from utils.feature_extractor import extract_simple_color_features
            features = extract_simple_color_features(roi)
            
            # 尝试加载预训练的XGBoost模型
            model_path = "models/color_concentration_xgb.pkl"
            if os.path.exists(model_path):
                # 加载预训练模型
                xgb_model = joblib.load(model_path)
                
                # 预测
                concentration = xgb_model.predict([features])[0]
            else:
                # 如果没有预训练模型，使用基础方法
                base_concentration = self.concentration_map[base_class_name]
                concentration = base_concentration  # 返回基础浓度作为默认值
                
        except ImportError:
            # 如果xgboost不可用，使用基础方法
            base_concentration = self.concentration_map[base_class_name]
            concentration = base_concentration
            
        return float(concentration)
    
    def _analyze_color_concentration_hybrid(self, roi, base_class_name):
        """
        使用混合模型预测连续浓度值
        
        Args:
            roi: 感兴趣区域（检测到的比色皿区域）
            base_class_name: 基础类别名称（用于参考）
            
        Returns:
            预测的连续浓度值
        """
        # 获取多种算法的预测结果
        conc_svr = self._analyze_color_concentration_svr(roi, base_class_name)
        conc_rf = self._analyze_color_concentration_rf(roi, base_class_name)
        conc_xgb = self._analyze_color_concentration_xgb(roi, base_class_name)
        conc_simple = self._analyze_color_concentration(roi, base_class_name)
        
        # 使用加权平均融合多个预测结果
        # 可以根据验证集性能调整权重
        weights = [0.25, 0.25, 0.25, 0.25]  # 等权重融合
        weighted_conc = (conc_svr * weights[0] + 
                        conc_rf * weights[1] + 
                        conc_xgb * weights[2] + 
                        conc_simple * weights[3])
        
        return float(weighted_conc)
    
    def save_model(self, save_path):
        """保存模型"""
        self.model.save(save_path)
    
    def load_model(self, model_path):
        """加载模型"""
        self.model = YOLO(model_path)
        # 将加载的模型也放到指定设备上
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"警告: 无法将模型移动到设备 {self.device}: {e}")
            print("回退到CPU设备")
            self.device = torch.device('cpu')
            self.model.to(self.device)
        
        # 清空缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # 示例用法
    print("模型功能：")
    print("1. 检测比色皿位置")
    print("2. 预测离散浓度值（分类）")
    print("3. 预测连续浓度值（回归）")
    print("4. 支持VOC数据集格式")
    print("5. 多种机器学习算法支持：SVR、随机森林、XGBoost、MLP、混合模型")