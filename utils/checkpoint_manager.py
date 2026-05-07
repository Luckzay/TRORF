import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any


class ModelCheckpointManager:
    """
    模型检查点管理器
    管理训练过程中的模型保存和最佳模型更新
    """
    
    def __init__(self, save_dir: str = "checkpoints", monitor_metric: str = "map50", 
                 mode: str = "max", save_freq: int = 10):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 模型保存目录
            monitor_metric: 监控的指标（用于比较best和current模型）
            mode: 模式，'max'表示越大越好，'min'表示越小越好
            save_freq: 保存频率（每隔多少epoch保存一次）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_freq = save_freq
        
        # 最佳模型信息
        self.best_metric_value = float('-inf') if mode == 'max' else float('inf')
        self.best_model_path = self.save_dir / "best_concentration_model.pt"
        
        # 最新模型信息
        self.latest_model_path = self.save_dir / "latest_concentration_model.pt"
        
    def save_model_if_better(self, model_path: str, metric_value: float, epoch: int = None):
        """
        如果模型性能更好，则保存为最佳模型
        
        Args:
            model_path: 模型路径
            metric_value: 模型指标值
            epoch: 训练轮数（可选，用于日志）
        """
        epoch_info = f" at epoch {epoch}" if epoch is not None else ""
        
        # 总是更新latest模型
        if Path(model_path).exists():
            # 更新latest模型
            if self.latest_model_path.exists():
                self.latest_model_path.unlink()
            
            # 在Windows上复制文件
            shutil.copy2(model_path, self.latest_model_path)
            
            print(f"已更新最新模型: {self.latest_model_path}{epoch_info}")
        
        # 检查是否需要更新最佳模型
        if self._is_better(metric_value, self.best_metric_value):
            # 更新最佳模型
            if self.best_model_path.exists():
                self.best_model_path.unlink()
            
            shutil.copy2(model_path, self.best_model_path)
            self.best_metric_value = metric_value
            
            print(f"新最佳模型已保存: {self.best_model_path}{epoch_info}, "
                  f"{self.monitor_metric}: {metric_value:.4f}")
        else:
            print(f"当前模型指标: {metric_value:.4f}, "
                  f"最佳指标: {self.best_metric_value:.4f}")
    
    def _is_better(self, current: float, best: float) -> bool:
        """
        判断当前指标是否优于最佳指标
        """
        if self.mode == 'max':
            return current > best
        else:
            return current < best
    
    def load_best_model(self, model_load_func, model_path: str = None):
        """
        加载最佳模型
        
        Args:
            model_load_func: 加载模型的函数
            model_path: 模型路径（如果不指定则使用最佳模型路径）
        """
        model_to_load = model_path or self.best_model_path
        
        if model_to_load and Path(model_to_load).exists():
            # 调用传入的模型加载函数
            model_load_func(model_to_load)
            print(f"已加载模型: {model_to_load}")
        else:
            print("未找到指定模型，使用当前模型")
    
    def get_best_metric_value(self) -> float:
        """
        获取最佳指标值
        """
        return self.best_metric_value


# 全局检查点管理器实例
checkpoint_manager = ModelCheckpointManager()