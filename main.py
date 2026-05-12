#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time
import os
import pickle
import numpy as np

# ================= 配置参数 =================
CONFIG = {
    "CAMERA_INDEX": 0,          # 摄像头索引
    "WIDTH": 320,               # 画面宽度（降低以提高树莓派性能）
    "HEIGHT": 240,              # 画面高度
    "CHALLENGE_TIMEOUT": 10.0,  # 眨眼挑战超时时间（秒）
    "UNLOCK_DURATION": 3,       # 开锁持续时间（秒）
    "FACE_CONF_THRESHOLD": 0.5, # YOLO 人脸检测置信度阈值
    "RECOGNITION_THRESHOLD": 0.5, # 人脸识别相似度阈值 (0~1)
}
# ===========================================

class SmartLock:
    def __init__(self, yolo_model="yolov8n-face-lindevs.onnx", sface_model="face_recognition_sface_2021dec.onnx"):
        # ---------- 初始化摄像头 ----------
        self.cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
        if not self.cap.isOpened():
            print("❌ 无法打开摄像头。请尝试执行：sudo modprobe bcm2835-v4l2")
            exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["WIDTH"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["HEIGHT"])
        print("📷 摄像头已就绪")

        # ---------- 加载 YOLO 人脸检测模型 (ONNX) ----------
        if not os.path.exists(yolo_model):
            print(f"❌ YOLO 模型不存在: {yolo_model}")
            exit(1)
        self.face_net = cv2.dnn.readNetFromONNX(yolo_model)
        print("✅ YOLOv8-face ONNX 模型加载完成")

        # ---------- 加载 SFace 人脸识别模型 ----------
        if not os.path.exists(sface_model):
            print(f"❌ SFace 模型不存在: {sface_model}")
            exit(1)
        self.face_recognizer = cv2.FaceRecognizerSF_create(sface_model, "")
        print("✅ SFace 人脸识别模型加载完成")

        # ---------- 加载眼睛检测器 (Haar) ----------
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if self.eye_cascade.empty():
            print("⚠️ 警告：眼睛检测器加载失败，眨眼检测将不可用")

        # ---------- 状态机变量 ----------
        self.state = "IDLE"
        self.face_box = None
        self.challenge_start = 0
        self.unlock_start = 0
        self.frame = None

        # ---------- 眨眼检测相关 ----------
        self.blink_count = 0
        self.eyes_are_open = False

        # ---------- 人脸数据库 ----------
        self.face_database_path = "face_features.pkl"
        self.registered_users = {}
        self.load_face_database()

        # ---------- 注册临时变量 ----------
        self.register_count = 0
        self.register_name = ""
        self.register_features = []

        print("✅ 系统就绪 | 按 'r' 注册 | 按 'd' 删除全部用户 | 按 'q' 退出")

    # ================= 数据库操作 =================
    def load_face_database(self):
        if os.path.exists(self.face_database_path):
            try:
                with open(self.face_database_path, 'rb') as f:
                    self.registered_users = pickle.load(f)
                print(f"📁 已加载 {len(self.registered_users)} 个注册用户")
            except:
                print("⚠️ 人脸数据库文件损坏，将创建新库")

    def save_face_database(self):
        with open(self.face_database_path, 'wb') as f:
            pickle.dump(self.registered_users, f)
        print(f"💾 数据库已保存 (共 {len(self.registered_users)} 人)")

    def delete_all_users(self):
        self.registered_users = {}
        if os.path.exists(self.face_database_path):
            os.remove(self.face_database_path)
        print("🗑️ 所有用户已删除")

    # ================= 人脸检测 =================
    def detect_face(self, frame):
        input_size = (640, 640)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
        self.face_net.setInput(blob)
        outputs = self.face_net.forward()
        outputs = np.squeeze(outputs).T

        boxes = []
        confidences = []
        for detection in outputs:
            confidence = detection[4]
            if confidence > CONFIG["FACE_CONF_THRESHOLD"]:
                x_center, y_center, width, height = detection[:4]
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))

        if not boxes:
            return None

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIG["FACE_CONF_THRESHOLD"], 0.4)
        if indices is None or len(indices) == 0:
            return None

        if isinstance(indices, (list, tuple, np.ndarray)):
            if len(indices) > 0:
                if isinstance(indices[0], (list, tuple, np.ndarray)):
                    best_idx = indices[0][0]
                else:
                    best_idx = indices[0]
            else:
                return None
        else:
            best_idx = indices

        x1, y1, x2, y2 = boxes[int(best_idx)]
        h, w = frame.shape[:2]
        x1 = int(x1 * w / input_size[0])
        y1 = int(y1 * h / input_size[1])
        x2 = int(x2 * w / input_size[0])
        y2 = int(y2 * h / input_size[1])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    # ================= 人脸识别 =================
    def _extract_face_feature(self, face_roi):
        aligned_face = cv2.resize(face_roi, (112, 112))
        feature = self.face_recognizer.feature(aligned_face)
        return feature

    def _recognize_face(self, face_roi):
        if len(self.registered_users) == 0:
            return True, "Guest"
        current_feature = self._extract_face_feature(face_roi)
        best_match = None
        best_similarity = -1.0
        for name, registered_feature in self.registered_users.items():
            similarity = self.face_recognizer.match(
                current_feature, registered_feature, cv2.FaceRecognizerSF_FR_COSINE
            )
            print(f"🔍 {name}: {similarity:.3f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        if best_similarity > CONFIG["RECOGNITION_THRESHOLD"]:
            print(f"✅ 识别成功: {best_match} ({best_similarity:.3f})")
            return True, best_match
        else:
            print(f"❌ 未知人脸 (最高相似度: {best_similarity:.3f})")
            return False, "Unknown"

    # ================= 眨眼检测 =================
    def _detect_eyes(self, face_roi):
        if face_roi.size == 0:
            return 0, []
        h, w = face_roi.shape[:2]
        top_h = int(h * 0.55)
        face_top = face_roi[0:top_h, 0:w]
        gray = cv2.cvtColor(face_top, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
        )
        adjusted_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes]
        return len(adjusted_eyes), adjusted_eyes

    def _check_blink(self):
        if self.face_box is None:
            return False
        x1, y1, x2, y2 = self.face_box
        face_roi = self.frame[y1:y2, x1:x2]
        num_eyes, _ = self._detect_eyes(face_roi)
        if num_eyes >= 2:
            self.eyes_are_open = True
        elif num_eyes == 0 and self.eyes_are_open:
            self.blink_count += 1
            self.eyes_are_open = False
            print(f"😉 检测到眨眼 ({self.blink_count}/2)")
        return self.blink_count >= 2

    # ================= 状态机（修复无人误触发） =================
    def _idle(self):
        self.face_box = self.detect_face(self.frame)
        if self.face_box is None:
            return
        x1, y1, x2, y2 = self.face_box
        face_roi = self.frame[y1:y2, x1:x2]
        recognized, name = self._recognize_face(face_roi)
        if recognized:
            self.state = "DETECTED"
            self.challenge_start = time.time()
            self.blink_count = 0
            self.eyes_are_open = False
            print(f"👤 欢迎 {name} | 请眨眼 2 次")
        else:
            self.face_box = None
            print("❌ 访问被拒绝")

    def _challenge(self):
        # 🔧 修复：如果人脸丢失，立即复位
        if self.face_box is None:
            print("⚠️ 人脸丢失，返回待机")
            self.state = "IDLE"
            return

        elapsed = time.time() - self.challenge_start
        if elapsed > CONFIG["CHALLENGE_TIMEOUT"]:
            print("⏱️ 挑战超时")
            self.state = "IDLE"
            self.face_box = None
            return
        if self._check_blink():
            print("✅ 活体验证通过 | 开锁中...")
            self.state = "UNLOCKING"
            self.unlock_start = time.time()

    def _unlocking(self):
        # 🔧 修复：如果人脸丢失，立即复位
        if self.face_box is None:
            print("⚠️ 人脸丢失，返回待机")
            self.state = "IDLE"
            return

        if time.time() - self.unlock_start >= CONFIG["UNLOCK_DURATION"]:
            print("🔒 落锁完成")
            self.state = "IDLE"
            self.face_box = None

    def _registering(self):
        self.face_box = self.detect_face(self.frame)
        if self.face_box is None:
            return
        x1, y1, x2, y2 = self.face_box
        face_roi = self.frame[y1:y2, x1:x2]
        if self.register_count < 50:
            if self.register_count % 5 == 0 and len(self.register_features) < 10:
                feature = self._extract_face_feature(face_roi)
                self.register_features.append(feature)
                print(f"📷 特征采集进度: {len(self.register_features)}/10")
            self.register_count += 1
        if len(self.register_features) >= 10:
            avg_feature = np.mean(self.register_features, axis=0)
            self.registered_users[self.register_name] = avg_feature
            self.save_face_database()
            print(f"✅ 注册完成！用户: {self.register_name}")
            self.state = "IDLE"
            self.register_features = []
            self.register_count = 0

    def start_register(self):
        self.state = "REGISTERING"
        self.register_count = 0
        self.register_features = []
        user_num = len(self.registered_users) + 1
        self.register_name = f"User_{user_num}"
        print(f"📸 开始注册 {self.register_name} | 请面对摄像头并缓慢转动头部")

    # ================= 主循环 =================
    def run(self):
        # 用于计算平均 FPS 和耗时
        fps_buffer = []
        yolo_times = []
        sface_times = []
        eye_times = []
        total_times = []
        avg_len = 30  # 平滑窗口

        frame_count = 0

        while True:
            loop_start = time.time()

            ret, self.frame = self.cap.read()
            if not ret:
                print("⚠️ 摄像头读帧失败，退出")
                break
            self.frame = cv2.flip(self.frame, 1)

            # ---------- 人脸检测 (YOLO) 计时 ----------
            t0 = time.time()
            if self.state != "IDLE" and self.state != "REGISTERING":
                current_box = self.detect_face(self.frame)
                if current_box is None:
                    self.face_box = None
            yolo_time = time.time() - t0
            yolo_times.append(yolo_time)

            # ---------- SFace 特征提取/匹配计时 ----------
            sface_time = 0.0
            # 注意：实际识别调用在 _idle 或 _registering 内部，我们无法直接计时
            # 这里做近似处理：在状态机调用前标记时间，调用后计算差值
            t1 = time.time()
            # 状态机调度
            if self.state == "IDLE":
                self._idle()
            elif self.state == "DETECTED":
                self._challenge()
            elif self.state == "UNLOCKING":
                self._unlocking()
            elif self.state == "REGISTERING":
                self._registering()
            sface_time = time.time() - t1
            sface_times.append(sface_time)

            # ---------- 眼睛检测计时（实际在 _check_blink 中）----------
            # 这里略过，因为 _check_blink 内部也有耗时，但为简化，我们只计总耗时

            total_time = time.time() - loop_start
            total_times.append(total_time)

            # ---------- 计算平均指标 ----------
            if len(total_times) > avg_len:
                yolo_times.pop(0)
                sface_times.pop(0)
                total_times.pop(0)

            avg_yolo_ms = np.mean(yolo_times) * 1000 if yolo_times else 0
            avg_sface_ms = np.mean(sface_times) * 1000 if sface_times else 0
            avg_total_ms = np.mean(total_times) * 1000 if total_times else 0
            avg_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0

            # 显示到画面（每隔几帧打印一次控制台，避免刷屏）
            frame_count += 1
            if frame_count % 10 == 0:
                print(
                    f"\r[FPS] {avg_fps:.2f} | YOLO: {avg_yolo_ms:.1f}ms | SFace: {avg_sface_ms:.1f}ms | Total: {avg_total_ms:.1f}ms",
                    end="")

            # 在画面上叠加文字
            cv2.putText(self.frame, f"FPS: {avg_fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(self.frame, f"YOLO: {avg_yolo_ms:.0f}ms  SFace: {avg_sface_ms:.0f}ms",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self._draw_ui()
            cv2.imshow("Smart Lock", self.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.start_register()
            elif key == ord('d'):
                self.delete_all_users()
    # ================= UI 绘制 =================
    def _draw_ui(self):
        h, w = self.frame.shape[:2]
        cv2.putText(self.frame, f"State: {self.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.frame, f"Users: {len(self.registered_users)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if self.face_box:
            x1, y1, x2, y2 = self.face_box
            color = (0, 255, 0) if self.state != "REGISTERING" else (255, 0, 0)
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
            face_roi = self.frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                _, eyes = self._detect_eyes(face_roi)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(self.frame, (x1+ex, y1+ey), (x1+ex+ew, y1+ey+eh), (255,0,0), 2)
        if self.state == "DETECTED":
            remaining = CONFIG["CHALLENGE_TIMEOUT"] - (time.time() - self.challenge_start)
            cv2.putText(self.frame, f"Blink 2 times ({remaining:.1f}s)",
                        (w//2-140, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(self.frame, f"Progress: {self.blink_count}/2",
                        (w//2-80, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            eye_status = "Eyes: Open" if self.eyes_are_open else "Eyes: Closed"
            cv2.putText(self.frame, eye_status, (w//2-60, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
        elif self.state == "REGISTERING":
            progress = len(self.register_features)
            cv2.putText(self.frame, f"REGISTERING ({progress}/10)",
                        (w//2-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(self.frame, "Press 'r' register | 'd' delete all | 'q' quit",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)


if __name__ == "__main__":
    try:
        cv2.FaceRecognizerSF_create
    except AttributeError:
        print("❌ 需要 opencv-contrib-python")
        exit(1)
    SmartLock().run()