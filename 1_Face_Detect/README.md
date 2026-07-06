# 1_Face_Detect：人脸识别 / 人脸定位代码块

## 1. 功能说明

本模块使用 OpenCV 开源 Haar Cascade 人脸检测模型完成本地图像中的人脸定位。程序会读取输入图像，完成预处理，调用模型推理，输出人脸坐标 `(x, y, w, h)`，并在原图上绘制矩形框后保存结果图像。

该代码块对应第一阶段任务说明书中的 **人脸识别代码块**：

1. 图像加载：支持本地图像文件，例如 `test.jpg`
2. 预处理：完成 BGR -> RGB 转换，并生成灰度图用于 OpenCV 检测
3. 模型加载：加载开源人脸检测模型 `haarcascade_frontalface_default.xml`
4. 推理：获取人脸坐标 `(x, y, w, h)`
5. 结果展示：在原图上绘制矩形框，并保存/可选显示结果图像

## 2. 目录结构

```text
1_Face_Detect/
├── face_detect.py
├── requirements.txt
├── README.md
├── model/
│   └── haarcascade_frontalface_default.xml
├── test_images/
│   ├── test_face_cartoon.jpg
│   └── README.md
└── outputs/
    └── .gitkeep
```

## 3. 环境要求

建议使用 Python 3.9 或以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

## 4. 运行方法

在 `1_Face_Detect/` 目录下执行：

```bash
python face_detect.py --image test_images/test_face_cartoon.jpg --output outputs/result.jpg
```

使用自己的测试图片：

```bash
python face_detect.py --image test_images/test.jpg --output outputs/result.jpg
```

如果本地有 GUI 环境，可以增加 `--show` 参数弹窗显示结果：

```bash
python face_detect.py --image test_images/test.jpg --output outputs/result.jpg --show
```

## 5. 输出结果

程序会生成：

- `outputs/result.jpg`：带人脸检测框的结果图像
- `outputs/faces.json`：检测到的人脸坐标，格式如下：

```json
{
  "count": 1,
  "faces": [
    {
      "x": 120,
      "y": 80,
      "w": 160,
      "h": 160
    }
  ]
}
```

终端会打印检测数量和每个人脸框的坐标。

## 6. 注意事项

- `test_face_cartoon.jpg` 是随包附带的示例测试图片，主要用于验证读取、推理、绘图和保存流程。
- 实际测试建议替换为真实含人脸图片，例如 `test_images/test.jpg`。
- Haar Cascade 对光照、角度、遮挡比较敏感；若需要更高精度，可后续替换为 OpenCV DNN、YOLO-Face 或 `face_recognition`。
