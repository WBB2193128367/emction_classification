# Thread.py
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import cv2
from .VitTransfer import ViTTransfer, predict


class CameraThread(QThread):
    end_info_signal = pyqtSignal(list,int)
    send_image_signal = pyqtSignal(np.ndarray)
    start_signal = pyqtSignal(bool,str, str)
    end_signal = pyqtSignal(bool,str, str)
    camera_state = pyqtSignal(str)
    recovery_ui_signal=pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.is_running = None
        self.skip_frames = 2
        self.frame_count = 0
        self.target_fps = 30
        self.camera_id = 0
        self.class_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

    @pyqtSlot()
    def run(self):
        self.start_signal.emit(True, "both", "camera")
        # 1. 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("摄像头打不开")
            return
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        print("摄像头检测开始...")
        self.is_running = True
        while self.is_running is True:
            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                break
            # 跳帧处理
            self.frame_count += 1
            if self.frame_count % self.skip_frames != 0:
                self.msleep(10)
                continue
            label, confidence, img = predict(frame, self.class_names)
            list_result = []
            list_result.append([label, f'{confidence:.2%}'])
            self.end_info_signal.emit(list_result,2)
            self.send_image_signal.emit(img)
        # 释放摄像头资源
        self.cleanup()
        self.end_signal.emit(False, "both", "camera")
        self.recovery_ui_signal.emit("摄像头显示区")
        self.camera_state.emit("摄像头已关闭")

    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
            self.cap = None
    @pyqtSlot()
    def stop(self):
        self.is_running = False
        self.msleep(10)

