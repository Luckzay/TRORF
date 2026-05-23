"""
图像特征提取工具模块
包含各种图像特征提取函数，用于浓度检测和机器学习预测
"""

import cv2
import numpy as np
import os


def extract_color_features(roi):
    """
    从感兴趣区域(ROI)中提取丰富的颜色特征

    Args:
        roi: 感兴趣区域图像 (numpy array)

    Returns:
        list: 包含所有颜色特征的列表
    """
    if roi.size == 0:
        return [0.0] * 42  # 返回固定长度的零向量

    # 确保ROI是3维的RGB格式
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        roi_rgb = roi
    elif len(roi.shape) == 2:
        # 灰度图转RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    else:
        # 其他情况，尝试转换
        roi_rgb = roi

    # 转换为多种颜色空间
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    roi_lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    roi_luv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LUV)
    roi_ycrcb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2YCrCb)

    # 计算各种颜色空间的统计特征
    features = []

    # RGB通道统计特征 (均值、标准差、偏度、峰度)
    for i in range(3):
        channel_data = roi_rgb[:, :, i].flatten()
        features.append(np.mean(channel_data))  # 均值
        features.append(np.std(channel_data))  # 标准差
        features.append(np.median(channel_data))  # 中位数
        features.append(np.max(channel_data))  # 最大值
        features.append(np.min(channel_data))  # 最小值
        features.append(np.percentile(channel_data, 25))  # 25百分位数
        features.append(np.percentile(channel_data, 75))  # 75百分位数

    # HSV通道统计特征
    for i in range(3):
        channel_data = roi_hsv[:, :, i].flatten()
        features.append(np.mean(channel_data))  # 均值
        features.append(np.std(channel_data))  # 标准差
        features.append(np.median(channel_data))  # 中位数
        features.append(np.max(channel_data))  # 最大值
        features.append(np.min(channel_data))  # 最小值
        features.append(np.percentile(channel_data, 25))  # 25百分位数
        features.append(np.percentile(channel_data, 75))  # 75百分位数

    # LAB通道统计特征
    for i in range(3):
        channel_data = roi_lab[:, :, i].flatten()
        features.append(np.mean(channel_data))  # 均值
        features.append(np.std(channel_data))  # 标准差
        features.append(np.median(channel_data))  # 中位数
        features.append(np.max(channel_data))  # 最大值
        features.append(np.min(channel_data))  # 最小值
        features.append(np.percentile(channel_data, 25))  # 25百分位数
        features.append(np.percentile(channel_data, 75))  # 75百分位数

    # LUV通道统计特征
    for i in range(3):
        channel_data = roi_luv[:, :, i].flatten()
        features.append(np.mean(channel_data))  # 均值
        features.append(np.std(channel_data))  # 标准差
        features.append(np.median(channel_data))  # 中位数
        features.append(np.max(channel_data))  # 最大值
        features.append(np.min(channel_data))  # 最小值
        features.append(np.percentile(channel_data, 25))  # 25百分位数
        features.append(np.percentile(channel_data, 75))  # 75百分位数

    # YCrCb通道统计特征
    for i in range(3):
        channel_data = roi_ycrcb[:, :, i].flatten()
        features.append(np.mean(channel_data))  # 均值
        features.append(np.std(channel_data))  # 标准差
        features.append(np.median(channel_data))  # 中位数
        features.append(np.max(channel_data))  # 最大值
        features.append(np.min(channel_data))  # 最小值
        features.append(np.percentile(channel_data, 25))  # 25百分位数
        features.append(np.percentile(channel_data, 75))  # 75百分位数

    # 颜色饱和度相关特征
    saturation = np.sqrt(np.var(roi_rgb[:, :, 0]) + np.var(roi_rgb[:, :, 1]) + np.var(roi_rgb[:, :, 2]))
    features.append(saturation)

    # 颜色均匀性特征
    color_uniformity = np.std(roi_rgb.flatten()) / np.mean(roi_rgb.flatten()) if np.mean(
        roi_rgb.flatten()) != 0 else 0
    features.append(color_uniformity)

    return features


def extract_simple_color_features(roi):
    """
    提取简单的颜色特征用于机器学习模型
    
    Args:
        roi: 感兴趣区域
        
    Returns:
        特征向量
    """
    # 转换为不同颜色空间
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    
    # RGB统计特征
    rgb_means = [np.mean(roi[:,:,i]) for i in range(3)]
    rgb_stds = [np.std(roi[:,:,i]) for i in range(3)]
    
    # HSV统计特征
    hsv_means = [np.mean(hsv[:,:,i]) for i in range(3)]
    hsv_stds = [np.std(hsv[:,:,i]) for i in range(3)]
    
    # LAB统计特征
    lab_means = [np.mean(lab[:,:,i]) for i in range(3)]
    lab_stds = [np.std(lab[:,:,i]) for i in range(3)]
    
    # 颜色直方图特征
    hist_r = cv2.calcHist([roi], [0], None, [8], [0, 256]).flatten()
    hist_g = cv2.calcHist([roi], [1], None, [8], [0, 256]).flatten()
    hist_b = cv2.calcHist([roi], [2], None, [8], [0, 256]).flatten()
    
    # 组合所有特征
    features = (rgb_means + rgb_stds + hsv_means + hsv_stds + 
               lab_means + lab_stds + hist_r.tolist() + 
               hist_g.tolist() + hist_b.tolist())
    
    return features


def extract_log_ratio_features(roi, blank_roi):
    """
    物理先验：Beer-Lambert 律 C ∝ log(I0/I)
    roi / blank_roi -> 1×6 向量
    用于浓度检测的6维对数比值特征
    
    Args:
        roi: 感兴趣区域图像 (numpy array)
        blank_roi: 空白对照区域图像 (numpy array)
        
    Returns:
        numpy array: 6维特征向量
    """
    if roi.size == 0 or blank_roi is None:
        return np.zeros(6)

    # 尺寸对齐
    h, w = roi.shape[:2]
    blank = cv2.resize(blank_roi, (w, h))

    # 避免 log(0)
    I = roi.astype(np.float32) + 1
    I0 = blank.astype(np.float32) + 1

    log_ratio = np.log(I0 / I)
    delta_e = I - I0

    # 6 维：RGB 各通道均值
    feat = [np.mean(log_ratio[:, :, ch]) for ch in range(3)] + \
           [np.mean(delta_e[:, :, ch]) for ch in range(3)]
    return np.array(feat)


def get_blank_from_same_image(img, boxes, clses, class_names):
    """
    从同一图像中获取空白对照区域（类别为'0ug'的框）
    如果没有找到，则返回全图均值
    
    Args:
        img: 原始图像
        boxes: 检测框列表
        clses: 类别列表
        class_names: 类别名称列表
        
    Returns:
        numpy array: 空白对照区域
    """
    blank_rois = []
    for box, cls_id in zip(boxes, clses):
        if class_names[int(cls_id)] == '0ug':
            x1, y1, x2, y2 = map(int, box)
            blank_rois.append(img[y1:y2, x1:x2])
    
    if not blank_rois:
        # 如果没有找到'0ug'类别的框，返回缩小的全图作为fallback
        return cv2.resize(img, (64, 64)).astype(np.float32)
    
    # 取最大的空白区域
    max_size = max(r.shape[0] * r.shape[1] for r in blank_rois)
    blank_resized = [cv2.resize(r, (max_size, max_size)) for r in blank_rois]
    return np.mean(blank_resized, axis=0).astype(np.float32)


def extract_log_ratio(roi, blank):
    """
    物理先验：Beer-Lambert 律 C ∝ log(I0/I)
    roi / blank_roi -> 1×6 向量
    用于浓度检测的6维对数比值特征
    
    Args:
        roi: 感兴趣区域图像 (numpy array)
        blank: 空白对照区域图像 (numpy array)
        
    Returns:
        numpy array: 6维特征向量
    """
    if roi.size == 0 or blank is None:
        return np.zeros(6)
    h, w = roi.shape[:2]
    blank = cv2.resize(blank, (w, h))
    I = roi.astype(np.float32) + 1
    I0 = blank.astype(np.float32) + 1
    log_ratio = np.log(I0 / I)
    delta_e = I - I0
    feat = [np.mean(log_ratio[:, :, ch]) for ch in range(3)] + \
           [np.mean(delta_e[:, :, ch]) for ch in range(3)]
    return np.array(feat)