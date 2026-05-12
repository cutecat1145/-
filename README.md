# 基于Yolov8face与SFace的树莓派活体检测智能门锁系统
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red)](https://www.raspberrypi.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue)](https://opencv.org/)
这是一个专为嵌入式设备（树莓派 5）优化的智能门锁方案。通过级联高性能的人脸检测与识别模型，实现了**高精度识别 + 眨眼活体防御**。
## ⚖️ 开源声明与致谢 (Acknowledgements)

本项目的核心功能依赖于以下优秀开源模型及仓库，特此致谢：

1. **人脸检测模型 (YOLOv8-face):** 
   - 模型文件：`yolov8n-face-lindevs.onnx`
   - 来源：感谢 [Lindevs](https://github.com/lindevs) 提供的 YOLOv8 人脸检测预训练权重。
   
2. **人脸检测与识别架构 (OpenCV Zoo):**
   - 模型文件：`face_detection_yunet_2023mar.onnx` / `face_recognition_sface.onnx`
   - 来源：[OpenCV Model Zoo](https://github.com/opencv/opencv_zoo)。YuNet 是由 OpenCV 官方维护的高性能轻量化人脸检测模型。

3. **算法框架:**
   - 感谢 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供的 YOLOv8 框架支持。

## 项目亮点
- **双模型联动：** 使用 **YOLOv8-face** 进行快速人脸定位，联合 **SFace** 进行高精度特征比对。
- **活体检测：** 内置基于 Haar Cascade 的眨眼检测算法，有效抵御照片、视频重放攻击。
- **性能优化：** 在树莓派 5 (2GB) 上实现了平均 **5 FPS** 的推理速度，识别延迟稳定在 **300ms** 左右。
- **状态机架构：** 采用严格的状态机设计（IDLE, DETECTED, UNLOCKING, REGISTERING），确保系统运行稳定。

##  Demo 演示


## 技术栈
- **硬件：** Raspberry Pi 5 (2GB), USB Camera
- **算法：** 
  - 检测：YOLOv8-face (ONNX)
  - 识别：SFace (OpenCV Zoo)
  - 活体：Haar Cascades (Blink Detection)
- **语言/库：** Python 3.10+, OpenCV-Python-Contrib, NumPy

