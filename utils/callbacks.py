"""
训练回调函数
"""
import torch
import torch.nn as nn
import numpy as np
import os


class LossHistory:
    """
    记录训练历史
    """
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.input_shape = input_shape
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 保存模型结构图
        input_image = torch.randn(2, 3, input_shape[0], input_shape[1])
        # 这里可以使用TensorBoard或其他工具来可视化模型
        
        self.train_loss = []
        self.val_loss = []
        
        # 创建writer用于记录
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except:
            self.writer = None
        
    def append_loss(self, epoch, train_loss, val_loss):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        
        if self.writer is not None:
            self.writer.add_scalar('Train_loss', train_loss, epoch)
            self.writer.add_scalar('Val_loss', val_loss, epoch)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            info = str({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            f.write(info)
            f.write("\n")
    
    def close(self):
        if self.writer is not None:
            self.writer.close()


class EvalCallback:
    """
    评估回调函数
    """
    def __init__(self, model, input_shape, class_names, num_classes, val_lines, log_dir, cuda, eval_flag=True, period=10):
        super(EvalCallback, self).__init__()
        
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.eval_flag = eval_flag
        self.period = period
        
        self.maps = [0]
        
    def on_epoch_end(self, epoch, model_eval):
        if not self.eval_flag:
            return
        
        if epoch % self.period == 0:
            # 这里实现评估逻辑
            # 由于实现复杂，暂时留空
            pass


class ModelEMA:
    """ 
    权值平滑
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - np.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def is_parallel(model):
    """
    检查模型是否使用了数据并行
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    """
    复制属性从b到a
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


from copy import deepcopy