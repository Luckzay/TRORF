import json
from pathlib import Path
import yaml


class Config:
    """
    配置管理器
    """
    def __init__(self, config_path=None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 如果没有指定配置文件路径，则使用默认路径
        if config_path is None:
            # 获取当前文件所在目录，然后拼接配置文件名
            current_dir = Path(__file__).parent
            self.config_path = current_dir / 'config.json'
        else:
            # 如果是相对路径，则相对于当前模块的目录
            if not Path(config_path).is_absolute():
                current_dir = Path(__file__).parent
                self.config_path = current_dir / config_path
            else:
                self.config_path = Path(config_path)
        
        self.config = self.load_config()
    
    def load_config(self):
        """
        加载配置文件
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get(self, key, default=None):
        """
        获取配置值，支持嵌套键访问（用点号分隔）
        
        Args:
            key: 键名，支持嵌套（如 'data.raw_data_dir'）
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key, value):
        """
        更新配置值
        
        Args:
            key: 键名，支持嵌套（如 'data.raw_data_dir'）
            value: 新值
        """
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def save(self):
        """
        保存配置到文件
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
    
    def create_yolo_dataset_config(self, output_path=None):
        """
        根据配置生成YOLO格式的数据集配置文件
        
        Args:
            output_path: 输出路径，默认为配置中的路径
        """
        output_path = output_path or self.get('paths.data_config_path', 'data_config.yaml')
        
        # 如果是相对路径，相对于当前工作目录
        if not Path(output_path).is_absolute():
            output_path = Path(output_path)
        
        # 使用项目根目录作为基准路径
        project_root = Path(__file__).parent.parent  # config目录的父目录
        processed_data_path = project_root / self.get('data.processed_data_dir', './processed_data')
        
        yolo_config = {
            'path': str(processed_data_path),  # 使用绝对路径或相对于项目根的路径
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.get('model.num_classes', 6),
            'names': self.get('model.class_names', ['0ug', '40ug', '80ug', '120ug', '160ug', '200ug'])
        }
        
        # 添加其他可选配置
        optional_configs = {
            'download': self.get('dataset.download', False),
            'kpt_shape': self.get('dataset.kpt_shape', None),
            'flip_idx': self.get('dataset.flip_idx', None),
            'max_det': self.get('dataset.max_det', 300),
            'auto_augment': self.get('dataset.auto_augment', None),
            'rect': self.get('dataset.rect', False),
            'cos_lr': self.get('dataset.cos_lr', False),
            'overlap_mask': self.get('dataset.overlap_mask', True),
            'mask_ratio': self.get('dataset.mask_ratio', 4)
        }
        
        # 只添加非None的可选配置
        for key, value in optional_configs.items():
            if value is not None:
                yolo_config[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"YOLO数据集配置文件已创建: {output_path}")
    
    @property
    def data_dir(self):
        # 使用项目根目录作为基准
        project_root = Path(__file__).parent.parent
        raw_data_dir = self.get('data.raw_data_dir')
        return str(project_root / raw_data_dir)
    
    @property
    def annotations_dir(self):
        # 使用项目根目录作为基准
        project_root = Path(__file__).parent.parent
        annotations_dir = self.get('data.annotations_dir')
        return str(project_root / annotations_dir)
    
    @property
    def images_dir(self):
        # 使用项目根目录作为基准
        project_root = Path(__file__).parent.parent
        images_dir = self.get('data.images_dir')
        return str(project_root / images_dir)
    
    @property
    def processed_data_dir(self):
        # 使用项目根目录作为基准，确保始终指向正确位置
        project_root = Path(__file__).parent.parent
        processed_data_dir = self.get('data.processed_data_dir')
        return str(project_root / processed_data_dir)
    
    @property
    def model_type(self):
        # 检查配置中的模型路径是否为绝对路径或已存在的相对路径
        model_path = self.get('model.type')
        if not Path(model_path).is_absolute():
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / model_path
        return str(model_path)
    
    @property
    def class_names(self):
        return self.get('model.class_names')
    
    @property
    def num_classes(self):
        return self.get('model.num_classes')
    
    @property
    def concentration_values(self):
        return self.get('model.concentration_values')
    
    @property
    def training_epochs(self):
        return self.get('training.epochs')
    
    @property
    def imgsz(self):
        return self.get('training.imgsz')
    
    @property
    def batch_size(self):
        return self.get('training.batch_size')
    
    @property
    def data_config_path(self):
        # 使用项目根目录作为基准
        project_root = Path(__file__).parent.parent
        config_path = self.get('paths.data_config_path', 'data_config.yaml')
        return str(project_root / config_path)


# 全局配置实例
config = Config()