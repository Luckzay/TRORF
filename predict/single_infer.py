import sys
import os
# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
from model.yolo_model import ConcentrationDetectionModel
import argparse
from config.config_manager import config
import torch
import matplotlib.pyplot as plt


def visualize_results(image_path, results):
    """
    可视化检测结果
    
    Args:
        image_path: 原始图像路径
        results: 模型预测结果
    """
    # 读取原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 绘制检测框和结果
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # 遍历每个检测结果
    for result in results:
        for detection in result['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            discrete_conc = detection['discrete_concentration']
            continuous_conc = detection['continuous_concentration']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            
            # 添加标签
            label = f'{class_name}: {continuous_conc:.1f}ug\nConf: {confidence:.2f}'
            ax.text(x1, y1-10, label, 
                   bbox=dict(facecolor='yellow', alpha=0.7),
                   fontsize=10, color='black')
    
    ax.set_title(f'Concentration Detection Results\nImage: {os.path.basename(image_path)}')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def single_image_inference(image_path):
    """
    对单张图像进行推理
    
    Args:
        image_path: 图像路径
    """
    # 检查GPU可用性
    if config.get('gpu.use_gpu', True) and torch.cuda.is_available():
        device = f'cuda:{config.get("gpu.device_id", 0)}'
        print(f"使用GPU设备进行推理: {device}")
    else:
        print("使用CPU进行推理")
    
    # 加载模型
    model = ConcentrationDetectionModel()
    # 如果有训练好的模型则加载，否则使用预训练模型
    model_path = 'E:\concentration-detection\models\\ulmodel.pt'
    if os.path.exists(model_path):
        model.load_model(model_path)
        print("加载训练好的模型")
    else:
        print("使用预训练模型进行推理（未经浓度数据训练）")

    # 进行预测
    results = model.predict_continuous_concentration(image_path)

    # 打印结果
    print("\n检测结果:")
    for i, result in enumerate(results):
        for detection in result['detections']:
            print(f"检测框: {detection['bbox']}")
            print(f"置信度: {detection['confidence']:.3f}")
            print(f"类别: {detection['class_name']}")
            print(f"离散浓度: {detection['discrete_concentration']} ug")
            print(f"连续浓度: {detection['continuous_concentration']:.3f} ug")
            print("-" * 40)
    
    # 可视化结果
    print("显示可视化结果...")
    visualize_results(image_path, results)


def main():
    # parser = argparse.ArgumentParser(description='单图像推理')
    # parser.add_argument('--image', type=str, required=True, help='图像路径')
    #
    # args = parser.parse_args()
    #
    # if not os.path.exists(args.image):
    #     print(f"错误: 图像文件不存在: {args.image}")
    #     return
    
    single_image_inference('E:\concentration-detection\processed_data\images\\train\\20260117_170237_583.jpg')


if __name__ == "__main__":
    main()