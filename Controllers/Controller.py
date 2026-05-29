# Controller.py
from Views.Mainwindow import MainWindow
from Models.Thread import WorkerThread
from Models.VitTransfer import init_model  # 导入初始化函数
from Models.Camera_thread import CameraThread


class Controller:
    def __init__(self):
        # ========== 新增：程序启动时加载模型 ==========
        print("程序启动，正在初始化模型...")
        init_model(
            num_classes=5,
            model_name=r"D:\program\python_program\emction_classification\resource\model\vit",
            model_path=r"D:\program\python_program\emction_classification\resource\model\vit_best_model.pth",
            device="cuda"  # 或 "cpu"
        )
        print("模型初始化完成")

        # 创建界面和工作线程
        self.view = MainWindow()
        self.cameraThread = CameraThread()
        self.thread = WorkerThread(self.view)
        self.connect()

    def connect(self):
        ### 图片
        # view -> view
        self.view.ui.pushButton.clicked.connect(self.view.button1SignalEmitter)
        self.view.ui.pushButton_2.clicked.connect(self.view.button2SignalEmitter)
        self.view.open_folder_signal.connect(self.view.openCustomDir)
        self.view.prediction_end_signal.connect(self.view.hintInfo)
        self.view.button_state_signal.connect(self.view.setButtonState)
        self.view.warning_signal.connect(self.view.warningInfo)
        self.view.ui.comboBox.currentTextChanged.connect(self.view.setFontText)
        # model -> view
        self.thread.end_info_signal.connect(self.view.setTableText)
        self.thread.start_signal.connect(self.view.setButtonState)
        self.thread.over_signal.connect(self.view.setButtonState)
        # view -> model
        self.view.destroyed.connect(self.thread.stop)
        self.view.send_filename_signal.connect(self.thread.setData)
        self.view.start_prediction_signal.connect(self.thread.start)

        ### 摄像头
        # view -> model
        self.view.open_camera_signal.connect(self.cameraThread.start)
        self.view.close_camera_signal.connect(self.cameraThread.stop)
        # model -> view
        self.cameraThread.end_info_signal.connect(self.view.setTableText)
        self.cameraThread.send_image_signal.connect(self.view.showImage)
        self.cameraThread.start_signal.connect(self.view.setButtonState)
        self.cameraThread.end_signal.connect(self.view.setButtonState)
        self.cameraThread.camera_state.connect(self.view.hintInfo)
        self.cameraThread.recovery_ui_signal.connect(self.view.recoveryUi)

    def show(self):
        self.view.show()
