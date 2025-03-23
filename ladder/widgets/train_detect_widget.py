import os
import shutil
from qtpy import QtWidgets, QtGui
from ultralytics import YOLO, settings
from ladder.utils import coco2json, ultraResult2Json, sliceDetect, sliceDetectBatch


def copy_folder_with_unique_name(src_folder, dest_folder):
    # Ensure the source folder exists
    if not os.path.exists(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return

    # Get the base folder name (e.g., 'weights')
    base_name = os.path.basename(os.path.normpath(src_folder))
    new_folder_path = os.path.join(dest_folder, base_name)

    # Check if the folder already exists in destination
    counter = 2
    while os.path.exists(new_folder_path):
        new_folder_path = os.path.join(dest_folder, f"{base_name}{counter}")
        counter += 1

    # Copy the folder
    shutil.copytree(src_folder, new_folder_path)
    print(f"Copied '{src_folder}' to '{new_folder_path}'.")

class TrainWidget(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super(TrainWidget, self).__init__(*args, **kwargs)

        # self.modelSelectBox = QtWidgets.QComboBox()
        # self.modelSelectBox.addItems([
        #     '---Select Model---',
        #     'yolov8n,3.2M','yolov8s,11.2M' ,'yolov8m,25.9M', 'yolov8l,43.7M','yolov8x,68.2M'
        # ])

        self.imgSize = QtWidgets.QLineEdit()
        # self.imgSize.setPlaceholderText("Enter like: 640")
        self.imgSize.setText("600")
        self.epoch = QtWidgets.QLineEdit()
        # self.epoch.setPlaceholderText("Enter like: 100")
        self.epoch.setText("100")
        self.trainBtn = QtWidgets.QPushButton()
        self.trainBtn.setText("Start Training")
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
        # layout.addWidget(self.modelSelectBox, 4,0,1,2)
        layout.addWidget(QtWidgets.QLabel('Epoch number:'),5,0)
        layout.addWidget(self.epoch,5,1)
        layout.addWidget(QtWidgets.QLabel('Model Input Image Size:'),6,0)
        layout.addWidget(self.imgSize,6,1,)
        layout.addWidget(self.trainBtn,7,0,1,2)
        self.setLayout(layout)
        self.path = "."

    def get_para(self):
        print(self.epoch.text())
        print(self.imgSize.text())
        # print(self.modelSelectBox.currentText())
        epoch = int(self.epoch.text())
        imgsz = int(self.imgSize.text())
        # model = self.modelSelectBox.currentText()
        data = self.file_list.text()
        weight = self.weight_list.text()
        if data and weight:
            self.yolov8Train(data=data,weight=weight,epochs=epoch,imgsz=imgsz, keep_mid=False)
        else:
            print("please select training data or weight file")

    def yolov8Train(self, data, weight, epochs, imgsz, model="yolov8n", keep_mid=True):
        # train
        if not weight:
            # weight = model.split(",")[0] + ".pt"
            # model = YOLO(weight)
            model = YOLO(model)
        else:
            model = YOLO(weight)

        # change settings
        runs_dir = os.path.dirname(self.file_list.text())

        settings.update({
            'runs_dir': runs_dir
        })
        results = model.train(data=data, epochs=epochs,imgsz=imgsz)
        train_output_dir = os.path.dirname(os.path.dirname(weight))
        weight_path = os.path.join(runs_dir,'detect/train/weights')
        print("=================================")
        print(f'runss_dir is {runs_dir}')
        print(f'train_output_dir is {train_output_dir}')
        print(f'weight_path is {weight_path}')
        print("=================================")

        try:
            copy_folder_with_unique_name(src_folder=weight_path, dest_folder=runs_dir)
            if not keep_mid:
                shutil.rmtree(os.path.join(runs_dir,'detect'))
        except Exception as e:
            raise e


    def open_file_dialog(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            self.path,
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
            self.path,
            "weight (*.pt *.pyt)"

        )
        if filenames:
            for file in filenames:
                self.weight_list.setText(str(file))


class DetectWidget(QtWidgets.QWidget):

    def __init__(self,*args, **kwargs):
        super(DetectWidget, self).__init__(*args, **kwargs)
        # self.modelSelectBox = QtWidgets.QComboBox()
        # self.modelSelectBox.addItems([
        #     '---Select Model---',
        #     'yolov8n(3.2M)','yolov8s(11.2M)' ,'yolov8m(25.9M)', 'yolov8l(43.7M)','yolov8x(68.2M)'
        # ])
        self.singleImg = None
        self.imgSize = QtWidgets.QLineEdit()
        self.imgSize.setText("600")
        # self.imgSize.setPlaceholderText("Enter like: 640")
        self.iou = QtWidgets.QLineEdit()
        self.iou.setText("0.6")
        # self.iou.setPlaceholderText("Enter like: 0.6")
        self.conf = QtWidgets.QLineEdit()
        # self.conf.setPlaceholderText("Enter like: 0.25")
        self.conf.setText("0.25")
        self.overlap = QtWidgets.QLineEdit()
        # self.overlap.setPlaceholderText("Enter like: 0.25")
        self.overlap.setText("0.25")
        self.slice = QtWidgets.QLineEdit()
        # self.slice.setPlaceholderText("Enter like: 2600")
        self.slice.setText("1000")

        self.detectBtn = QtWidgets.QPushButton()
        self.detectBtn.setText("Start Prediction")
        self.detectBtn.clicked.connect(self.star_detect)

        folderDialog = QtWidgets.QPushButton("Select Folder")
        folderDialog.clicked.connect(self.open_folder_dialog)
        self.folder_list = QtWidgets.QLineEdit()

        fileDialog = QtWidgets.QPushButton("Select Image")
        fileDialog.clicked.connect(self.open_file_dialog)
        self.file_list = QtWidgets.QLineEdit()

        weightDialog = QtWidgets.QPushButton("Browse weight")
        weightDialog.clicked.connect(self.open_weight_dialog)
        self.weight_list = QtWidgets.QLineEdit()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Selected Folder:'),0,0)
        layout.addWidget(folderDialog,0,1)
        layout.addWidget(self.folder_list,1,0,1,2)

        layout.addWidget(QtWidgets.QLabel('Selected Image:'),2,0)
        layout.addWidget(fileDialog,2,1)
        layout.addWidget(self.file_list,3,0,1,2)

        layout.addWidget(QtWidgets.QLabel('Selected Weight:'),4,0)
        layout.addWidget(weightDialog,4,1)
        layout.addWidget(self.weight_list,5,0,1,2)
        # layout.addWidget(self.modelSelectBox, 4,0,1,2)
        layout.addWidget(QtWidgets.QLabel('IoU:'),6,0)
        layout.addWidget(self.iou,6,1)
        layout.addWidget(QtWidgets.QLabel('Confidence:'),7,0)
        layout.addWidget(self.conf,7,1)
        layout.addWidget(QtWidgets.QLabel('Model Input Image Size:'),8,0)
        layout.addWidget(self.imgSize,8,1)
        layout.addWidget(QtWidgets.QLabel('Crop Tile size:'),9,0)
        layout.addWidget(self.slice,9,1)
        layout.addWidget(QtWidgets.QLabel('Overlap:'),10,0)
        layout.addWidget(self.overlap,10,1)
        layout.addWidget(self.detectBtn,11,0,1,2)
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
                self.path = os.path.dirname(file)
                self.file_list.setText(str(file))# for single image detection

                # self.file_list.setText(str(self.path)) # for multiple images detection

    def open_folder_dialog(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Files or folder",
            self.path,
            "Image (*.png *.jpg *.jpeg)"
        )
        if filenames:
            for file in filenames:
                self.path = os.path.dirname(file)
                # self.file_list.setText(str(file))# for single image detection
                self.folder_list.setText(str(self.path)) # for multiple images detection

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

    def star_detect(self):
        conf = float(self.conf.text())
        iou = float(self.iou.text())
        imgsz = int(self.imgSize.text())
        # model = self.modelSelectBox.currentText()
        # data = self.file_list.text()
        weight = self.weight_list.text()
        slice_sz = int(self.slice.text())
        overlap = float(self.overlap.text())

        if self.folder_list.text():
            data = self.folder_list.text()
        else:
            data = self.file_list.text()

        if slice_sz and overlap:

            if data:

                if os.path.isfile(data):
                    print("slice detection in single image!")
                    sliceDetect(weight=weight,img=data,conf=conf,iou=iou,img_size=imgsz,
                                img_h=slice_sz,img_w=slice_sz,overlap=overlap,gpu="cpu")
                if os.path.isdir(data):
                    print("slice detection in multiple images!")
                    sliceDetectBatch(weight=weight,img_fd=data,conf=conf,iou=iou,img_size=imgsz,
                                    img_h=slice_sz,img_w=slice_sz,overlap=overlap,gpu="cpu")
            else:
                print("please select a image or a folder")

        else:
            print("standard detection in a whole image!")
            if data:
                self.yolov8Detect(
                    model= model, data=data,weight=weight,imgsz=imgsz, conf=conf,iou=iou,keep_mid=False
                )
            else:
                self.yolov8Detect(
                    model= model, data=self.singleImg,weight=weight,imgsz=imgsz, conf=conf,iou=iou,keep_mid=False
                )

    def yolov8Detect(self, model, data, weight, imgsz, conf, iou, keep_mid=True):
        # predict
        if not weight:
            weight = model.split(",")[0] + ".pt"
            model = YOLO(weight)
        else:
            model = YOLO(weight)

        # changing settings
        runs_dir = os.path.dirname(self.file_list.text())
        settings.update({
            'runs_dir': runs_dir
        })
        print(settings)

        results = model.predict(data, save=True, save_conf=True, save_txt=True,
                                imgsz=imgsz, conf=conf, iou=iou)

        ultraResult2Json(results=results)

        if not keep_mid:
            shutil.rmtree(os.path.join(runs_dir,'detect'))
        return results
