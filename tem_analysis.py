#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEM 图像分析 —— 微粒检测 → 像素转实际距离 → 粒径分布直方图。

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
from typing import Any, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 模块级常量
# ---------------------------------------------------------------------------
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_DEFAULT_MIN_AREA = 30.0       # px²
_DEFAULT_MAX_AREA = 10000.0    # px²
_DEFAULT_SCALE_NM = 100.0      # 比例尺对应实际长度 (nm)
_DEFAULT_SCALE_PX = 100.0      # 比例尺对应像素长度
_DEFAULT_BINS = 20


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


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEM 微粒检测与粒径分布分析：输出检测框图、直方图、CSV 和 summary.json。"
    )
    parser.add_argument("--input", default="test_tem.jpg", help="TEM 输入图像路径。")
    parser.add_argument("--output-dir", default="outputs", help="输出文件夹路径。")
    parser.add_argument("--model", default="model.pth", help="模型/配置权重文件路径。")
    parser.add_argument(
        "--scale-nm", type=float, default=_DEFAULT_SCALE_NM,
        help=f"比例尺对应的实际长度 (nm)，默认 {_DEFAULT_SCALE_NM}。",
    )
    parser.add_argument(
        "--scale-px", type=float, default=_DEFAULT_SCALE_PX,
        help=f"比例尺对应的像素长度，默认 {_DEFAULT_SCALE_PX}。",
    )
    parser.add_argument(
        "--min-area", type=float, default=_DEFAULT_MIN_AREA,
        help=f"微粒最小轮廓面积 (px²)，默认 {_DEFAULT_MIN_AREA}。",
    )
    parser.add_argument(
        "--max-area", type=float, default=_DEFAULT_MAX_AREA,
        help=f"微粒最大轮廓面积 (px²)，默认 {_DEFAULT_MAX_AREA}。",
    )
    parser.add_argument(
        "--dark-particles", action="store_true", default=True,
        help="检测亮背景上的深色微粒（默认）。",
    )
    parser.add_argument(
        "--bright-particles", action="store_false", dest="dark_particles",
        help="检测暗背景上的亮色微粒。",
    )
    parser.add_argument(
        "--bins", type=int, default=_DEFAULT_BINS,
        help=f"直方图分箱数量，默认 {_DEFAULT_BINS}。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O 工具
# ---------------------------------------------------------------------------
def resolve_path(path_like: str | Path) -> Path:
    """将相对路径解析为当前工作目录下的绝对路径。"""
    return Path(path_like).expanduser().resolve()


def load_image(image_path: Path) -> np.ndarray:
    """加载 TEM 图像（BGR 色彩）。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    return image


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model(model_path: Path) -> dict[str, Any]:
    """
    加载微粒检测模型或检测配置。

    本交付包的 model.pth 是基线检测配置，保证代码可直接运行。
    后续接入真实 YOLO/CNN 时，可将返回对象替换为模型实例，并在
    detect_particles 中调用。
    """
    default_config: dict[str, Any] = {
        "model_type": "opencv_contour_baseline",
        "description": "OpenCV contour detector baseline for TEM particle analysis",
        "version": "1.0.0",
    }

    if not model_path.exists():
        return default_config

    # 尝试 torch.load（兼容 .pth 权重），失败则回退到 pickle
    for loader in (_try_torch_load, _try_pickle_load):
        obj = loader(model_path)
        if obj is not None:
            if isinstance(obj, dict):
                return {**default_config, **obj}
            return {**default_config, "raw_model": obj}

    return default_config


def _try_torch_load(model_path: Path) -> Any | None:
    """尝试用 torch.load 加载模型文件（优先 safe 模式）。"""
    try:
        import torch  # type: ignore[import-untyped]
        return torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception:
        try:
            return torch.load(str(model_path), map_location="cpu", weights_only=False)
        except Exception:
            return None


def _try_pickle_load(model_path: Path) -> Any | None:
    """尝试用 pickle 加载模型文件。"""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 图像预处理与阈值分割
# ---------------------------------------------------------------------------
def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """灰度化 → CLAHE 对比度增强 → 高斯滤波去噪。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (5, 5), 0)


def threshold_particles(gray: np.ndarray, dark_particles: bool = True) -> np.ndarray:
    """Otsu 自适应阈值 + 形态学开/闭运算，输出二值掩膜。"""
    thresh_type = cv2.THRESH_BINARY_INV if dark_particles else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _MORPH_KERNEL, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2)
    return binary


# ---------------------------------------------------------------------------
# 轮廓特征
# ---------------------------------------------------------------------------
def contour_circularity(contour: np.ndarray) -> float:
    """计算轮廓圆度：4πA / P²（完美圆 = 1）。"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4 * math.pi * area / (perimeter * perimeter))


# ---------------------------------------------------------------------------
# 微粒检测（核心）
# ---------------------------------------------------------------------------
def detect_particles(
    image_bgr: np.ndarray,
    model: dict[str, Any],
    nm_per_pixel: float,
    min_area: float = _DEFAULT_MIN_AREA,
    max_area: float = _DEFAULT_MAX_AREA,
    dark_particles: bool = True,
) -> tuple[list[ParticleMeasurement], np.ndarray]:
    """
    检测微粒并计算粒径。

    流程：预处理 → Otsu 阈值 → 轮廓提取 → 形态/大小过滤 →
          排序/编号 → 绘制标注框图。

    返回 (测量结果列表, 标注后的 BGR 图像)。
    """
    _ = model  # 预留给真实模型推理接口

    # 1. 预处理 + 二值化
    gray = preprocess(image_bgr)
    binary = threshold_particles(gray, dark_particles=dark_particles)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 遍历轮廓，收集符合过滤条件的候选
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            continue

        # 过滤细长划痕 / 边界伪影
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > 3.5:
            continue

        circularity = contour_circularity(contour)
        if circularity < 0.25:
            continue

        candidates.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": float(area), "circularity": circularity,
        })

    # 3. 按位置排序后统一编号
    candidates.sort(key=lambda c: (c["y"], c["x"]))

    measurements: list[ParticleMeasurement] = []
    for idx, c in enumerate(candidates, start=1):
        diameter_px = (c["w"] + c["h"]) / 2.0
        measurements.append(ParticleMeasurement(
            particle_id=idx,
            x=c["x"], y=c["y"],
            width_px=c["w"], height_px=c["h"],
            diameter_px=diameter_px,
            diameter_nm=diameter_px * nm_per_pixel,
            area_px2=c["area"],
            circularity=c["circularity"],
        ))

    # 4. 绘制标注（此时编号已确定，与 CSV 一致）
    annotated = _draw_annotations(image_bgr, measurements)

    return measurements, annotated


def _draw_annotations(
    image_bgr: np.ndarray,
    measurements: Sequence[ParticleMeasurement],
) -> np.ndarray:
    """在图像上绘制检测框与粒径标签。"""
    annotated = image_bgr.copy()
    for m in measurements:
        x, y, w, h = m.x, m.y, m.width_px, m.height_px
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{m.particle_id}: {m.diameter_nm:.1f} nm",
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


# ---------------------------------------------------------------------------
# 输出：直方图 / CSV / JSON
# ---------------------------------------------------------------------------
def build_summary(
    measurements: list[ParticleMeasurement],
    scale_nm: float,
    scale_px: float,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    """根据检测结果构建汇总字典。"""
    model_meta = {k: v for k, v in model_info.items() if k != "raw_model"}
    base: dict[str, Any] = {
        "particle_count": len(measurements),
        "scale_nm": scale_nm,
        "scale_px": scale_px,
        "nm_per_pixel": scale_nm / scale_px,
        "model_info": model_meta,
    }

    if not measurements:
        base["message"] = "未检测到满足过滤条件的微粒，请调整 min_area/max_area 或阈值方向。"
        return base

    diameters = np.array([m.diameter_nm for m in measurements], dtype=np.float64)
    base.update({
        "mean_diameter_nm": float(np.mean(diameters)),
        "median_diameter_nm": float(np.median(diameters)),
        "std_diameter_nm": float(np.std(diameters, ddof=1)) if len(diameters) > 1 else 0.0,
        "min_diameter_nm": float(np.min(diameters)),
        "max_diameter_nm": float(np.max(diameters)),
    })
    return base


def save_histogram(diameters_nm: list[float], output_path: Path, bins: int) -> None:
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
    measurements: list[ParticleMeasurement],
    annotated: np.ndarray,
    output_dir: Path,
    bins: int,
    scale_nm: float,
    scale_px: float,
    model_info: dict[str, Any],
) -> None:
    """保存检测框图、直方图、CSV 和汇总 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / "detected_particles.png"
    histogram_path = output_dir / "particle_size_histogram.png"
    csv_path = output_dir / "particle_measurements.csv"
    summary_path = output_dir / "summary.json"

    # 框图
    cv2.imwrite(str(annotated_path), annotated)

    # CSV
    df = pd.DataFrame([asdict(m) for m in measurements])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 直方图
    diameters_nm = [m.diameter_nm for m in measurements]
    if diameters_nm:
        save_histogram(diameters_nm, histogram_path, bins=bins)

    # 汇总 JSON
    summary = build_summary(measurements, scale_nm, scale_px, model_info)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 控制台输出
    print("分析完成。输出文件：")
    print(f"- 检测框图：{annotated_path}")
    has_hist = f"- 粒径直方图：{histogram_path}" if diameters_nm else "- 粒径直方图：未生成（无粒径数据）"
    print(has_hist)
    print(f"- 粒径数据 CSV：{csv_path}")
    print(f"- 汇总 JSON：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
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
