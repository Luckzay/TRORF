"""
Face Detection Module
---------------------
使用 OpenCV 开源 Haar Cascade 模型完成本地图像人脸定位。

功能流程：
1. 图像加载
2. 图像预处理：BGR -> RGB，同时生成灰度图供检测模型使用
3. 模型加载：读取 OpenCV Haar Cascade XML 模型
4. 推理：输出人脸坐标 (x, y, w, h)
5. 结果展示：绘制矩形框，保存/可选显示结果图像

运行示例：
python face_detect.py --image test_images/test_face_cartoon.jpg --output outputs/result.jpg
python face_detect.py --image your_face.jpg --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

FaceBox = Tuple[int, int, int, int]  # x, y, w, h


def load_image(image_path: str | Path) -> np.ndarray:
    """加载本地图像文件，返回 OpenCV BGR 图像。"""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"图像读取失败，请检查文件格式是否受支持: {image_path}")

    return image_bgr


def preprocess_image(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    图像预处理。

    - image_rgb: 按任务要求完成 BGR -> RGB 转换，可用于后续深度模型输入或展示
    - image_gray: Haar Cascade 检测所需灰度图
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    return image_rgb, image_gray


def load_model(model_path: str | Path) -> cv2.CascadeClassifier:
    """加载 OpenCV Haar Cascade 人脸检测模型。"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    detector = cv2.CascadeClassifier(str(model_path))
    if detector.empty():
        raise RuntimeError(f"模型加载失败: {model_path}")

    return detector


def detect_faces(
    detector: cv2.CascadeClassifier,
    image_gray: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> List[FaceBox]:
    """执行人脸检测，返回坐标列表，坐标格式为 (x, y, w, h)。"""
    boxes = detector.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]


def draw_faces(image_bgr: np.ndarray, face_boxes: List[FaceBox]) -> np.ndarray:
    """在原图上绘制人脸矩形框和编号。"""
    result = image_bgr.copy()

    for idx, (x, y, w, h) in enumerate(face_boxes, start=1):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            result,
            f"Face {idx}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return result


def save_result(image_bgr: np.ndarray, output_path: str | Path) -> None:
    """保存检测结果图像。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image_bgr)
    if not ok:
        raise IOError(f"结果图像保存失败: {output_path}")


def save_boxes_json(face_boxes: List[FaceBox], json_path: str | Path) -> None:
    """可选：保存检测框坐标 JSON，便于测试和接口调用。"""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "count": len(face_boxes),
        "faces": [
            {"x": x, "y": y, "w": w, "h": h}
            for x, y, w, h in face_boxes
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_image(
    image_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    json_output: str | Path | None = None,
    show: bool = False,
) -> List[FaceBox]:
    """完整处理流程：加载 -> 预处理 -> 模型推理 -> 绘制 -> 保存。"""
    image_bgr = load_image(image_path)
    _image_rgb, image_gray = preprocess_image(image_bgr)

    detector = load_model(model_path)
    face_boxes = detect_faces(detector, image_gray)

    result = draw_faces(image_bgr, face_boxes)
    save_result(result, output_path)

    if json_output is not None:
        save_boxes_json(face_boxes, json_output)

    print(f"[INFO] 输入图像: {image_path}")
    print(f"[INFO] 检测到人脸数量: {len(face_boxes)}")
    for idx, (x, y, w, h) in enumerate(face_boxes, start=1):
        print(f"[INFO] Face {idx}: x={x}, y={y}, w={w}, h={h}")
    print(f"[INFO] 结果图像已保存: {output_path}")

    if show:
        cv2.imshow("Face Detection Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return face_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV 人脸检测代码块")
    parser.add_argument(
        "--image",
        type=str,
        default="test_images/test_face_cartoon.jpg",
        help="输入图像路径，默认 test_images/test_face_cartoon.jpg",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/haarcascade_frontalface_default.xml",
        help="OpenCV Haar Cascade 模型 XML 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.jpg",
        help="输出结果图像路径",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="outputs/faces.json",
        help="输出人脸坐标 JSON 路径；如不需要可传空字符串",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否弹窗显示结果图像。服务器/无GUI环境请勿开启。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_output = args.json_output if args.json_output else None
    process_image(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        json_output=json_output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
