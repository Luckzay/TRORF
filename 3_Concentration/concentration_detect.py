"""
3_Concentration · 溶液浓度检测

功能：
    输入比色皿/样品管图像，自动检测溶液 ROI 并预测浓度。
    pipeline：图像加载 → YOLO 自动检测 ROI（去除瓶壁干扰）
              → 107 维颜色特征提取 → XGBoost 回归 → 浓度数值叠加可视化

用法：
    python concentration_detect.py --image 测试图.jpg
    python concentration_detect.py --image 测试图.jpg --output 结果.jpg

输出：
    - 控制台：每个检测到的比色皿的浓度值 + 边界框坐标
    - results/<原图名>：叠加了检测框和浓度文字的可视化图

注：控制台日志用英文，避免 Windows GBK 控制台中文乱码。
"""
import argparse
import os
import sys
import warnings

# 抑制 XGBoost / sklearn 跨版本 pickle 加载警告
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(
        description='Solution concentration detection (YOLO auto-ROI + XGBoost regression)')
    parser.add_argument('--image', required=True, help='input image path')
    parser.add_argument('--output', default=None,
                        help='output image path, default: results/<image name>')
    args = parser.parse_args()

    # 在 chdir 之前把输入/输出路径转成绝对路径，否则切目录后相对路径会失效
    image_path = os.path.abspath(args.image)
    output_path = os.path.abspath(args.output) if args.output else None

    if not os.path.exists(image_path):
        sys.exit(f'[ERROR] input image not found: {image_path}')

    # 切到脚本所在目录：yolo.py 内部用相对路径加载 detection_model.pth /
    # calibration_model.pkl / model_data/voc_classes_Fe-S.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 延迟导入：确保 cwd 已切换；YOLO 实例化时会加载模型文件
    from PIL import Image
    from yolo import YOLO

    print('[INFO] Loading YOLO detector (detection_model.pth) + XGBoost regressor (calibration_model.pkl)...')
    yolo = YOLO()

    print(f'[INFO] Running inference on: {image_path}')
    image = Image.open(image_path)
    # detect_image_with_concentration 内部完成：YOLO 检测 → ROI 裁剪 →
    # extract_color_features（107 维）→ XGBoost 回归 → 在图上叠加浓度文字
    # 同时会把每个框的浓度值打印到控制台（格式：concentration top left bottom right）
    result_image = yolo.detect_image_with_concentration(image)

    # 保存结果图
    if output_path is None:
        os.makedirs('results', exist_ok=True)
        output_path = os.path.join(script_dir, 'results',
                                   os.path.basename(image_path))
    result_image.save(output_path)
    print(f'[RESULT] Annotated image saved: {output_path}')
    print('[INFO] Per-cuvette concentrations are printed above '
          '(format: concentration top left bottom right)')


if __name__ == '__main__':
    main()
