# Thread.py
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from .VitTransfer import predictImage  # 改用快速预测函数


class WorkerThread(QThread):
    end_info_signal = pyqtSignal(list,int)
    start_signal = pyqtSignal(bool, str,str)
    over_signal = pyqtSignal(bool, str,str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.running = None

    @pyqtSlot(str)
    def setData(self, data: str):
        self.data = data

    @pyqtSlot()
    def run(self):
        self.start_signal.emit(False, "both","image")
        class_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
        list_result = []

        self.running = True
        # 使用快速预测（不会重复加载模型）
        label, confidence = predictImage(self.data, class_names)
        self.running = False

        list_result.append([label, f'{confidence:.2%}'])
        self.end_info_signal.emit(list_result,1)
        self.over_signal.emit(True, "both", "image")

    @pyqtSlot()
    def stop(self):
        while self.running is True:
            self.msleep(10)