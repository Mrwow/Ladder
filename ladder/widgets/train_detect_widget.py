import os
from qtpy import QtWidgets, QtGui
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from ladder.utils import coco2json


class TrainWidget(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super(TrainWidget, self).__init__(*args, **kwargs)

        self.modelSelectBox = QtWidgets.QComboBox()
        self.modelSelectBox.addItems([
            '---Select Model---',
            'yolov8n,3.2M','yolov8s,11.2M' ,'yolov8m,25.9M', 'yolov8l,43.7M','yolov8x,68.2M'
        ])

        self.imgSize = QtWidgets.QLineEdit()
        self.imgSize.setPlaceholderText("Enter like: 640")
        self.epoch = QtWidgets.QLineEdit()
        self.epoch.setPlaceholderText("Enter like: 100")
        self.trainBtn = QtWidgets.QPushButton()
        self.trainBtn.setText("Start training")
        # self.trainBtn.setIcon(QtGui.QIcon("../icons/train.png"))
        # self.trainBtn.resize(200,200)
        self.trainBtn.clicked.connect(self.get_para)

        directDialog = QtWidgets.QPushButton("Browse data")
        directDialog.clicked.connect(self.open_file_dialog)
        self.file_list = QtWidgets.QLineEdit()
        weightDialog = QtWidgets.QPushButton("Browse weight")
        weightDialog.clicked.connect(self.open_weight_dialog)
        self.weight_list = QtWidgets.QLineEdit()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Selected Files:'),0,0)
        layout.addWidget(directDialog,0,1)
        layout.addWidget(self.file_list,1,0,1,2)
        layout.addWidget(QtWidgets.QLabel('Selected Weight:'),2,0)
        layout.addWidget(weightDialog,2,1)
        layout.addWidget(self.weight_list,3,0,1,2)
        layout.addWidget(self.modelSelectBox, 4,0,1,2)
        layout.addWidget(QtWidgets.QLabel('Epoch number:'),5,0)
        layout.addWidget(self.epoch,5,1)
        layout.addWidget(QtWidgets.QLabel('Image size:'),6,0)
        layout.addWidget(self.imgSize,6,1,)
        layout.addWidget(self.trainBtn,7,0,1,2)
        self.setLayout(layout)

    def get_para(self):
        print(self.epoch.text())
        print(self.imgSize.text())
        print(self.modelSelectBox.currentText())
        epoch = int(self.epoch.text())
        imgsz = int(self.imgSize.text())
        model = self.modelSelectBox.currentText()
        data = self.file_list.text()
        weight = self.weight_list.text()
        self.yolov8Train(model= model, data=data,weight=weight,epochs=epoch,imgsz=imgsz)

    def yolov8Train(self, model, data, weight, epochs, imgsz):
        # train
        if not weight:
            weight = model.split(",")[0] + ".pt"
            model = YOLO(weight)
        else:
            model = YOLO(weight)
        results = model.train(data=data, epochs=epochs,imgsz=imgsz)

    def open_file_dialog(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "/home",
            "data (*.yaml)"

        )
        if filenames:
            for file in filenames:
                # dir_path = os.path.dirname(file)
                dir_path = file
                self.file_list.setText(str(dir_path))


    def open_weight_dialog(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "/home",
            "weight (*.pt *.pyt)"

        )
        if filenames:
            for file in filenames:
                self.weight_list.setText(str(file))


class DetectWidget(QtWidgets.QWidget):

    def __init__(self,*args, **kwargs):
        super(DetectWidget, self).__init__(*args, **kwargs)
        self.modelSelectBox = QtWidgets.QComboBox()
        self.modelSelectBox.addItems([
            '---Select Model---',
            'yolov8n(3.2M)','yolov8s(11.2M)' ,'yolov8m(25.9M)', 'yolov8l(43.7M)','yolov8x(68.2M)'
        ])
        self.imgSize = QtWidgets.QLineEdit()
        self.imgSize.setPlaceholderText("Enter like: 640")
        self.iou = QtWidgets.QLineEdit()
        self.iou.setPlaceholderText("Enter like: 0.6")
        self.conf = QtWidgets.QLineEdit()
        self.conf.setPlaceholderText("Enter like: 0.25")
        self.overlap = QtWidgets.QLineEdit()
        self.overlap.setPlaceholderText("Enter like: 0.25")

        self.detectBtn = QtWidgets.QPushButton()
        self.detectBtn.setText("Start Detecting")
        self.detectBtn.clicked.connect(self.get_para)

        directDialog = QtWidgets.QPushButton("Browse data")
        directDialog.clicked.connect(self.open_file_dialog)
        self.file_list = QtWidgets.QLineEdit()
        weightDialog = QtWidgets.QPushButton("Browse weight")
        weightDialog.clicked.connect(self.open_weight_dialog)
        self.weight_list = QtWidgets.QLineEdit()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Selected Files:'),0,0)
        layout.addWidget(directDialog,0,1)
        layout.addWidget(self.file_list,1,0,1,2)
        layout.addWidget(QtWidgets.QLabel('Selected Weight:'),2,0)
        layout.addWidget(weightDialog,2,1)
        layout.addWidget(self.weight_list,3,0,1,2)
        layout.addWidget(self.modelSelectBox, 4,0,1,2)
        layout.addWidget(QtWidgets.QLabel('IoU:'),5,0)
        layout.addWidget(self.iou,5,1)
        layout.addWidget(QtWidgets.QLabel('Confidence:'),6,0)
        layout.addWidget(self.conf,6,1)
        layout.addWidget(QtWidgets.QLabel('Image size:'),7,0)
        layout.addWidget(self.imgSize,7,1)
        layout.addWidget(QtWidgets.QLabel('Overlap:'),8,0)
        layout.addWidget(self.overlap,8,1)
        layout.addWidget(self.detectBtn,9,0,1,2)
        self.setLayout(layout)

        self.path = "."

    def open_file_dialog(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Files or folder",
            self.path,
            "Image (*.png *.jpg *.jpeg)"
        )
        if filenames:
            for file in filenames:
                dir_path = file
                self.file_list.setText(str(dir_path))
                self.path = os.path.dirname(file)

    def open_weight_dialog(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            self.path,
            "weight (*.pt *.pyt)"

        )
        if filenames:
            for file in filenames:
                self.weight_list.setText(str(file))

    def get_para(self):
        print(self.imgSize.text())
        print(self.modelSelectBox.currentText())
        conf = float(self.conf.text())
        iou = float(self.iou.text())
        imgsz = int(self.imgSize.text())
        model = self.modelSelectBox.currentText()
        data = self.file_list.text()
        weight = self.weight_list.text()
        results = self.yolov8Detect(model= model, data=data,weight=weight,imgsz=imgsz, conf=conf,iou=iou)

    def yolov8Detect(self, model, data, weight, imgsz, conf, iou):
        # predict
        if not weight:
            weight = model.split(",")[0] + ".pt"
            model = YOLO(weight)
        else:
            model = YOLO(weight)

        results = model.predict(data, save=True, save_conf=True, save_txt=True,
                                imgsz=imgsz, conf=conf, iou=iou)
        return results

    def sliceDetect(self):
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=self.weight_list.text(),
            confidence_threshold=int(self.conf.text()),
            device="cpu", # or 'cuda:0'
        )
        result = get_sliced_prediction(
            self.file_list.text(),
            detection_model,
            slice_height = int(self.imgSize.text()),
            slice_width = int(self.imgSize.text()),
            overlap_height_ratio = int(self.overlap.text()),
            overlap_width_ratio = int(self.overlap.text())
        )
        result.export_visuals(export_dir=dir)
        # coco = result.to_coco_annotations()
        # coco2json(coco,img_url=img_url)
