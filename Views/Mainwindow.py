import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QFileDialog, QTableWidgetItem, QMessageBox, QHeaderView
from .emotion import Ui_Form


class MainWindow(QWidget):
    # 信号
    # 发射图片路径信号
    send_filename_signal = pyqtSignal(str)
    # 检测完发射的信号
    prediction_end_signal = pyqtSignal(str)
    # 发送按钮状态的信号
    button_state_signal = pyqtSignal(bool, str, str)
    warning_signal = pyqtSignal(str)
    open_folder_signal = pyqtSignal()
    open_camera_signal = pyqtSignal()
    start_prediction_signal = pyqtSignal()
    close_camera_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # 设置列宽平分模式
        header = self.ui.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.ui.verticalLayout_2.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.ui.verticalLayout_3.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    @pyqtSlot()
    def openCustomDir(self):
        """带标题和初始目录"""
        filelist, filter = QFileDialog.getOpenFileName(
            self,
            "请选择一张图片",  # 标题
            "D:/image",
            "All Files (*);;"
            "Video Files (*.mp4 *.avi *.mkv);;"
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;"
            "PNG Files (*.png);;"
            "JPEG Files (*.jpg *.jpeg);;"
            "GIF Files(*.gif)"
        )
        if filelist == "":
            self.button_state_signal.emit(False, "button2", "image")
            # 设置标签显示的文字
            self.ui.label.setText("图片显示区")
            # 清空tableWidget的数据
            self.ui.tableWidget.clearContents()
            self.ui.tableWidget.setRowCount(0)
        elif filelist.split('.')[-1].lower() in ["jpg", "png", "jpeg", "bmp", "gif", "ico"]:
            self.send_filename_signal.emit(filelist)
            self.button_state_signal.emit(True, "button2", "image")
            self.ui.tableWidget.clearContents()
            self.ui.tableWidget.setRowCount(0)
            pixmap = QPixmap(filelist)
            scaled_pixmap = pixmap.scaled(self.ui.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.ui.label.setPixmap(scaled_pixmap)
        else:
            self.warning_signal.emit("你选的是图片吗？")

    @pyqtSlot(list, int)
    def setTableText(self, data: list, sign: int) -> None:
        self.ui.tableWidget.setRowCount(len(data))
        for row, (name, value) in enumerate(data):
            item_name = QTableWidgetItem(str(name))
            item_value = QTableWidgetItem(str(value))
            item_name.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  # 水平居中
            item_value.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.tableWidget.setItem(row, 0, item_name)
            self.ui.tableWidget.setItem(row, 1, item_value)
        if sign == 1:
            self.prediction_end_signal.emit('图片检测完成！')
        else:
            pass

    @pyqtSlot(str)
    def hintInfo(self, data: str):
        QMessageBox.information(self, '操作成功', data)

    @pyqtSlot(bool, str, str)
    def setButtonState(self, state: bool, target: str = "both", source: str = "image"):
        if source == "image":
            if target in ("both", "button2"):
                self.ui.pushButton_2.setEnabled(state)
            if target in ("both", "button1"):
                self.ui.pushButton.setEnabled(state)
                self.ui.comboBox.setEnabled(state)
        elif source == "camera":
            if target in ("both", "button2"):
                self.ui.pushButton_2.setEnabled(state)
            if target in ("both", "button1"):
                self.ui.pushButton.setEnabled(not state)
                self.ui.comboBox.setEnabled(not state)

    @pyqtSlot(str)
    def warningInfo(self, data: str):
        QMessageBox.warning(self, '警告', data)

    @pyqtSlot(np.ndarray)
    def showImage(self, frame_rgb: np.ndarray):
        # 获取图像尺寸
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        # numpy array -> QImage -> QPixmap
        qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.ui.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.ui.label.setPixmap(scaled_pixmap)

    @pyqtSlot(str)
    def setFontText(self, text: str):
        if text == "图片识别":
            self.ui.pushButton.setText("打开图片")
            self.ui.label.setText("图片显示区")
            self.ui.pushButton_2.setText("开始检测")
        elif text == "摄像头识别":
            self.ui.pushButton.setText("打开摄像头")
            self.ui.label.setText("摄像头显示区")
            self.ui.pushButton_2.setText("关闭摄像头")
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_2.setEnabled(False)
        # 清空tableWidget的数据
        self.ui.tableWidget.clearContents()
        self.ui.tableWidget.setRowCount(0)

    @pyqtSlot()
    def button1SignalEmitter(self):
        if self.ui.pushButton.text() == "打开图片":
            self.open_folder_signal.emit()
        elif self.ui.pushButton.text() == "打开摄像头":
            self.open_camera_signal.emit()
    @pyqtSlot()
    def button2SignalEmitter(self):
        if self.ui.pushButton_2.text() == "开始检测":
            self.start_prediction_signal.emit()
        elif self.ui.pushButton_2.text() == "关闭摄像头":
            self.close_camera_signal.emit()

    @pyqtSlot()
    def recoveryUi(self):
        self.ui.label.setText("摄像头显示区")
        self.ui.tableWidget.clearContents()
        self.ui.tableWidget.setRowCount(0)