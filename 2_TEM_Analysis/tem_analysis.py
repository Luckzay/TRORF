#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEM 图像分析代码块
功能：微粒检测 -> 像素转实际距离 -> 绘制粒径分布直方图

默认实现采用 OpenCV 轮廓检测作为可运行的基线检测器；如果提供真实训练好的
YOLO/CNN 权重，可在 load_model()/detect_particles() 中替换为对应模型推理逻辑。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ParticleMeasurement:
    """单个微粒的检测与粒径测量结果。"""

    particle_id: int
    x: int
    y: int
    width_px: int
    height_px: int
    diameter_px: float
    diameter_nm: float
    area_px2: float
    circularity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEM 微粒检测与粒径分布分析：输出检测框图、直方图、CSV 和 summary.json。"
    )
    parser.add_argument("--input", default="test_tem.jpg", help="TEM 输入图像路径。")
    parser.add_argument("--output-dir", default="outputs", help="输出文件夹路径。")
    parser.add_argument("--model", default="model.pth", help="模型/配置权重文件路径。")
    parser.add_argument(
        "--scale-nm",
        type=float,
        default=100.0,
        help="比例尺对应的实际长度，单位 nm。例如 100 表示比例尺为 100 nm。",
    )
    parser.add_argument(
        "--scale-px",
        type=float,
        default=100.0,
        help="比例尺对应的像素长度。例如 100 表示 100 nm = 100 px。",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=30.0,
        help="微粒最小轮廓面积，单位 px^2。用于过滤噪声。",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=10000.0,
        help="微粒最大轮廓面积，单位 px^2。用于过滤聚团或边框。",
    )
    parser.add_argument(
        "--dark-particles",
        action="store_true",
        default=True,
        help="默认检测亮背景上的深色微粒。",
    )
    parser.add_argument(
        "--bright-particles",
        action="store_false",
        dest="dark_particles",
        help="检测暗背景上的亮色微粒。",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="直方图分箱数量。",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    """将相对路径解析到当前工作目录。"""
    return Path(path_like).expanduser().resolve()


def load_image(image_path: Path) -> np.ndarray:
    """加载 TEM 图像。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    return image


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """灰度化、对比度增强与去噪。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred


def load_model(model_path: Path) -> Dict[str, Any]:
    """
    加载微粒检测模型或检测配置。

    本交付包的 model.pth 是基线检测配置，保证代码可直接运行。
    后续接入真实 YOLO/CNN 时，可将返回对象替换为模型实例，并在 detect_particles 中调用。
    """
    default_config: Dict[str, Any] = {
        "model_type": "opencv_contour_baseline",
        "description": "OpenCV contour detector baseline for TEM particle analysis",
        "version": "1.0.0",
    }
    if not model_path.exists():
        return default_config

    # 优先尝试 torch.load，兼容 .pth 权重文件；失败时尝试 pickle。
    try:
        import torch  # type: ignore

        obj = torch.load(str(model_path), map_location="cpu")
        if isinstance(obj, dict):
            return {**default_config, **obj}
        return {**default_config, "raw_model": obj}
    except Exception:
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                return {**default_config, **obj}
        except Exception:
            pass

    return default_config


def threshold_particles(gray: np.ndarray, dark_particles: bool = True) -> np.ndarray:
    """使用 Otsu 阈值将疑似微粒区域分割出来。"""
    threshold_type = cv2.THRESH_BINARY_INV if dark_particles else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, threshold_type + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def contour_circularity(contour: np.ndarray) -> float:
    """计算轮廓圆度：4πA / P²。"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4 * math.pi * area / (perimeter * perimeter))


def detect_particles(
    image_bgr: np.ndarray,
    model: Dict[str, Any],
    nm_per_pixel: float,
    min_area: float,
    max_area: float,
    dark_particles: bool = True,
) -> Tuple[List[ParticleMeasurement], np.ndarray]:
    """
    检测微粒并计算粒径。

    当前基线算法：
    1. 灰度化/CLAHE/高斯滤波；
    2. Otsu 阈值分割；
    3. 轮廓提取；
    4. 过滤噪声和异常区域；
    5. 等效直径 = (检测框宽 + 检测框高) / 2。
    """
    _ = model  # 预留给真实模型推理接口。
    gray = preprocess(image_bgr)
    binary = threshold_particles(gray, dark_particles=dark_particles)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    measurements: List[ParticleMeasurement] = []
    annotated = image_bgr.copy()

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            continue

        # 过滤非常细长的划痕/边界伪影。
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > 3.5:
            continue

        circularity = contour_circularity(contour)
        if circularity < 0.25:
            continue

        diameter_px = (w + h) / 2.0
        diameter_nm = diameter_px * nm_per_pixel
        particle_id = len(measurements) + 1

        measurements.append(
            ParticleMeasurement(
                particle_id=particle_id,
                x=int(x),
                y=int(y),
                width_px=int(w),
                height_px=int(h),
                diameter_px=float(diameter_px),
                diameter_nm=float(diameter_nm),
                area_px2=float(area),
                circularity=float(circularity),
            )
        )

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{particle_id}: {diameter_nm:.1f} nm",
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # 按横坐标/纵坐标排序后重新编号，便于人工检查。
    measurements.sort(key=lambda item: (item.y, item.x))
    for idx, m in enumerate(measurements, start=1):
        m.particle_id = idx

    return measurements, annotated


def save_histogram(diameters_nm: List[float], output_path: Path, bins: int) -> None:
    """保存粒径分布直方图。"""
    plt.figure(figsize=(8, 5))
    plt.hist(diameters_nm, bins=bins, edgecolor="black")
    plt.xlabel("Particle diameter (nm)")
    plt.ylabel("Count")
    plt.title("TEM Particle Size Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_outputs(
    measurements: List[ParticleMeasurement],
    annotated: np.ndarray,
    output_dir: Path,
    bins: int,
    scale_nm: float,
    scale_px: float,
    model_info: Dict[str, Any],
) -> None:
    """保存检测框图、直方图、CSV 和汇总 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / "detected_particles.png"
    histogram_path = output_dir / "particle_size_histogram.png"
    csv_path = output_dir / "particle_measurements.csv"
    summary_path = output_dir / "summary.json"

    cv2.imwrite(str(annotated_path), annotated)

    rows = [asdict(m) for m in measurements]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    diameters_nm = [m.diameter_nm for m in measurements]
    if diameters_nm:
        save_histogram(diameters_nm, histogram_path, bins=bins)
        summary = {
            "particle_count": len(measurements),
            "mean_diameter_nm": float(np.mean(diameters_nm)),
            "median_diameter_nm": float(np.median(diameters_nm)),
            "std_diameter_nm": float(np.std(diameters_nm, ddof=1)) if len(diameters_nm) > 1 else 0.0,
            "min_diameter_nm": float(np.min(diameters_nm)),
            "max_diameter_nm": float(np.max(diameters_nm)),
            "scale_nm": scale_nm,
            "scale_px": scale_px,
            "nm_per_pixel": scale_nm / scale_px,
            "model_info": {k: v for k, v in model_info.items() if k != "raw_model"},
        }
    else:
        summary = {
            "particle_count": 0,
            "message": "未检测到满足过滤条件的微粒，请调整 min_area/max_area 或阈值方向。",
            "scale_nm": scale_nm,
            "scale_px": scale_px,
            "nm_per_pixel": scale_nm / scale_px,
            "model_info": {k: v for k, v in model_info.items() if k != "raw_model"},
        }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("分析完成。输出文件：")
    print(f"- 检测框图：{annotated_path}")
    print(f"- 粒径直方图：{histogram_path if diameters_nm else '未生成，原因：无粒径数据'}")
    print(f"- 粒径数据 CSV：{csv_path}")
    print(f"- 汇总 JSON：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    model_path = resolve_path(args.model)

    if args.scale_px <= 0 or args.scale_nm <= 0:
        raise ValueError("scale-px 和 scale-nm 必须为正数。")

    nm_per_pixel = args.scale_nm / args.scale_px
    image_bgr = load_image(input_path)
    model_info = load_model(model_path)

    measurements, annotated = detect_particles(
        image_bgr=image_bgr,
        model=model_info,
        nm_per_pixel=nm_per_pixel,
        min_area=args.min_area,
        max_area=args.max_area,
        dark_particles=args.dark_particles,
    )

    save_outputs(
        measurements=measurements,
        annotated=annotated,
        output_dir=output_dir,
        bins=args.bins,
        scale_nm=args.scale_nm,
        scale_px=args.scale_px,
        model_info=model_info,
    )


if __name__ == "__main__":
    main()
