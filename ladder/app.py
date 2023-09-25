# -*- coding: utf-8 -*-
import functools
import io
import os
import PIL.Image
import math
import imgviz
import cv2

from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Qt

from .__init__ import __appname__
from ladder.widgets import Canvas, ZoomWidget, FileDialogPreview, \
    LabelFile, Shape, LabelDialog, UniqueLabelQListWidget,\
    LabelListWidget,LabelListWidgetItem, CropDialog, TrainWidget, DetectWidget
from ladder.actions import baseAction
from ladder.detect import detect_run, train
from ladder.utils import jsonToYolo
# from ladder.detect import detect_run, jsonToYolo, train, imputeMissingBoxes
# from ladder.yolov8 import yolov8Train, yolov8Detect

LABEL_COLORMAP = imgviz.label_colormap()
# LABEL_COLORMAP = [[0,0,0],
#                   [128,0,0],
#                   [0,128,0],
#                   [128,128,0],
#                   [0,0,128]]

class MainWindow(QtWidgets.QMainWindow):
    filename = None
    def __init__(self,filename=None,output_dir=None,output_file=None):
        super(MainWindow,self).__init__()
        self.setWindowTitle(__appname__)
        self.filename = filename
        self.detect_shapes = None
        self.output_dir = output_dir
        self.labelFile = None
        self._noSelectionSlot = False

        # canvas
        self.zoomWidget = ZoomWidget()
        self.canvas = Canvas()
        scrollAreaForCanvas = QtWidgets.QScrollArea()
        scrollAreaForCanvas.setWidget(self.canvas)
        scrollAreaForCanvas.setWidgetResizable(True)
        self.scrollBar = {
            Qt.Horizontal: scrollAreaForCanvas.horizontalScrollBar(),
            Qt.Vertical: scrollAreaForCanvas.verticalScrollBar()
        }

        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.newShape.connect(self.newShape)
        self.canvas.labelUpdate.connect(self.labelUpdate)
        self.setCentralWidget(scrollAreaForCanvas)
        self.canvas.cropImgDig.connect(self.cropImgDig)

        # label dialog for label input and edit
        self.labelDialog = LabelDialog(parent=self)
        self.cropDialog = CropDialog(parent=self)
        # uniqul labelList
        self.uniqLabelList = UniqueLabelQListWidget()

        # polygon label list
        self.labelList = LabelListWidget()
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.shape_dock = QtWidgets.QDockWidget(
            self.tr("Labels List"), self
        )
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.shape_dock)
        # train widget
        self.trainWidget = TrainWidget()
        self.train_dock = QtWidgets.QDockWidget(
            self.tr("Training"), self
        )
        self.train_dock.setWidget(self.trainWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.train_dock)
        # Detection widget
        self.detectWidget = DetectWidget()
        self.detect_dock = QtWidgets.QDockWidget(
            self.tr("Detecting"), self
        )
        self.detect_dock.setWidget(self.detectWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.detect_dock)

        # Top toolbar button actions
        action = functools.partial(baseAction, self)
        open_file = action("&Open",'openFile',self.openFile)
        btn_open_file = QtWidgets.QToolButton()
        btn_open_file.setDefaultAction(open_file)

        open_dir = action("Open a folder", "openDir", self.openDir)
        btn_open_dir = QtWidgets.QToolButton()
        btn_open_dir.setDefaultAction(open_dir)

        detect = action("&Detect",'detect', self.detect)
        btn_detect = QtWidgets.QToolButton()
        btn_detect.setDefaultAction(detect)

        next_img = action("&Next image",'next', self.nextImg)
        btn_next_img = QtWidgets.QToolButton()
        btn_next_img.setDefaultAction(next_img)

        pre_img = action("&Pre image",'prev',self.preImg)
        btn_pre_img = QtWidgets.QToolButton()
        btn_pre_img.setDefaultAction(pre_img)

        zoom_in = action("&Zoom in", "zoom-in", functools.partial(self.zoomValue, 1.1))
        btn_zoom_in = QtWidgets.QToolButton()
        btn_zoom_in.setDefaultAction(zoom_in)

        zoom_out = action("&Zoom out", "zoom-out", functools.partial(self.zoomValue, 0.9))
        btn_zoom_out = QtWidgets.QToolButton()
        btn_zoom_out.setDefaultAction(zoom_out)

        edit_shape = action("&Edit", "edit", self.editShape)
        btn_edit_shape = QtWidgets.QToolButton()
        btn_edit_shape.setDefaultAction(edit_shape)

        draw_rect = action("&Draw", 'rectangular', self.drawRec)
        btn_draw_rect = QtWidgets.QToolButton()
        btn_draw_rect.setDefaultAction(draw_rect)

        save_file = action("&Save", "save", self.saveFile)
        btn_save_file = QtWidgets.QToolButton()
        btn_save_file.setDefaultAction(save_file)

        crop_img = action("&Crop", "crop", self.cropImg)
        btn_crop_img = QtWidgets.QToolButton()
        btn_crop_img.setDefaultAction(crop_img)

        del_shape = action("&Delete", "delete", self.deletFile)
        btn_del_shape = QtWidgets.QToolButton()
        btn_del_shape.setDefaultAction(del_shape)

        train = action("&Train", "train", self.train)
        btn_train = QtWidgets.QToolButton()
        btn_train.setDefaultAction(train)

        # Top toolbar
        toolbar = QtWidgets.QToolBar()
        toolbar.layout().setSpacing(0)
        toolbar.layout().setContentsMargins(0,0,0,0)
        toolbar.setContentsMargins(0,0,0,0)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        toolbar.addWidget(btn_open_file)
        toolbar.addWidget(btn_crop_img)
        # toolbar.addWidget(btn_open_dir)
        toolbar.addWidget(btn_detect)
        toolbar.addWidget(btn_zoom_in)
        toolbar.addWidget(btn_zoom_out)
        toolbar.addWidget(btn_edit_shape)
        toolbar.addWidget(btn_draw_rect)
        # toolbar.addWidget(btn_next_img)
        # toolbar.addWidget(btn_pre_img)

        toolbar.addWidget(btn_save_file)
        toolbar.addWidget(btn_del_shape)
        
        toolbar.addWidget(btn_train)

        self.addToolBar(Qt.TopToolBarArea,toolbar)

        self.image = QtGui.QImage()
        self.imagePath = None
        self.otherData = None
        self.output_file = output_file
        self.lastOpenDir = None

        self.settings = QtCore.QSettings("ladder", "ladder")
        self.window_size = (1200, 800)
        size = self.settings.value("window/size", QtCore.QSize(self.window_size[0],self.window_size[1]))
        self.resize(size)


    def openFile(self, _value=False):
        if self.filename:
            path = os.path.dirname(str(self.filename))
        else:
            path = "."

        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats
        )

        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
            )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            self.filename = fileDialog.selectedFiles()[0]
            if self.filename:
                self.loadFile(self.filename)

    def openDir(self):
        print("open dir")
        return

    def loadShapes(self, shapes, replace=True):
        for shape in shapes:
            self._update_shape_color(shape)
            # add into labelDialog
            self.labelDialog.addLabelHistory(shape.label)
            # add into label list
            label_list_item = LabelListWidgetItem(shape.label, shape)
            self.labelList.addItem(label_list_item)
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()
            s.append(shape)
        self.loadShapes(s)

    def loadFile(self,filename=None):
        print("load file to canvas")
        self.labelList.clear()
        self.canvas.setEnabled(False)
        filename = str(filename)
        label_file = os.path.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = os.path.basename(label_file)
            label_file = os.path.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file):
            self.labelFile = LabelFile(label_file)
            self.imageData = self.labelFile.imageData
            self.imagePath = os.path.join(
                os.path.dirname(label_file),
                self.labelFile.imagePath,
            )
        else:
            self.imageData = LabelFile.load_image_file(filename)
            print("load images")
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
            self.canvas.shapes = []
        image = QtGui.QImage.fromData(self.imageData)
        self.canvas.pixmap = QtGui.QPixmap.fromImage(image)
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
        self.canvas.setEnabled(True)
        self.zoomValueInitial()
        self.canvas.update()

    def zoomValueInitial(self):
        img_w, img_h = self.canvas.pixmap.width(), self.canvas.pixmap.height()
        win_w = self.window_size[0]
        win_h = self.window_size[1] - 200
        scale_init = min(win_w/img_w, win_h/img_h)
        self.canvas.scale = scale_init * 0.8
        self.zoomWidget.setValue(int(100 * scale_init * 0.8))

    def zoomValue(self,increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.zoomWidget.setValue(zoom_value)
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def detect(self):
        print("load model and detect")
        if self.filename:
            path = os.path.dirname(str(self.filename))
        else:
            path = "."

        filter = "yolov3 model weight (*.pt)"
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filter)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Detection Model Weight") % __appname__,
            )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            self.detection_weight = fileDialog.selectedFiles()[0]
            if self.detection_weight and self.filename:
                self.yolov3(self.filename,self.detection_weight, add=False)
            else:
                print("please load image and detection model")
    #
    # def detectYolov3(self):
    #     print("load model and detect")
    #     if self.filename:
    #         path = os.path.dirname(str(self.filename))
    #     else:
    #         path = "."
    #
    #     filter = "yolov3 model weight (*.pt)"
    #     fileDialog = FileDialogPreview(self)
    #     fileDialog.setFileMode(FileDialogPreview.ExistingFile)
    #     fileDialog.setNameFilter(filter)
    #     fileDialog.setWindowTitle(
    #         self.tr("%s - Choose Detection Model Weight") % __appname__,
    #         )
    #     fileDialog.setWindowFilePath(path)
    #     fileDialog.setViewMode(FileDialogPreview.Detail)
    #     if fileDialog.exec_():
    #         self.detection_weight = fileDialog.selectedFiles()[0]
    #         if self.detection_weight and self.filename:
    #             self.yolov3(self.filename,self.detection_weight, add=False)
    #         else:
    #             print("please load image and detection model")
    # def yolov3(self,img,weight, add=True):
    #     # self.detect_shapes = run(source=img,weights=weight, imgsz=3000, save_txt=True)
    #     self.detect_shapes = detect_run(source=img,weights=weight, save_txt=True, imgsz=1100,conf_thres=0.5,iou_thres=0.6)
    #     print("++++++boxes 11111++++++")
    #     print(len(self.detect_shapes))
    #     if add:
    #         # self.detect_shapes = imputeMissingBoxes(img=img, shapes=self.detect_shapes)
    #         print("++++++boxes 2222++++++")
    #         print(len(self.detect_shapes))
    #         print(self.detect_shapes)
    #     self.loadLabels(self.detect_shapes)
    #
    # def trainYolov3(self, _value=False, dirpath=None):
    #     print("select image data to re-train yolov3")
    #     defaultOpenDirPath = dirpath if dirpath else "."
    #     if self.lastOpenDir and os.path.exists(self.lastOpenDir):
    #         defaultOpenDirPath = self.lastOpenDir
    #     else:
    #         defaultOpenDirPath = (
    #             os.path.dirname(self.filename) if self.filename else "."
    #         )
    #
    #     targetDirPath = str(
    #         QtWidgets.QFileDialog.getExistingDirectory(
    #             self,
    #             self.tr("%s - Open Directory") % __appname__,
    #             defaultOpenDirPath,
    #             QtWidgets.QFileDialog.ShowDirsOnly
    #             | QtWidgets.QFileDialog.DontResolveSymlinks,
    #             )
    #     )
    #
    #     print(targetDirPath)
    #     if targetDirPath:
    #         data_dict = jsonToYolo(targetDirPath)
    #         train(data_dict,targetDirPath)

    def train(self, _value=False, dirpath=None):
        print("select image data to re-train yolov3")
        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = (
                os.path.dirname(self.filename) if self.filename else "."
            )

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
                )
        )

        print(targetDirPath)
        if targetDirPath:
            data_dict = jsonToYolo(targetDirPath)
            train(data_dict,targetDirPath)

    def deletFile(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, "
            "proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
                self, self.tr("Attention"), msg, yes | no, yes
        ):
            shapes = self.canvas.deleteSelected()
            for shape in shapes:
                item = self.labelList.findItemByShape(shape)
                self.labelList.removeItem(item)

    def editShape(self):
        print("edit")
        self.canvas.mode = self.canvas.EDIT
        return

    def drawRec(self):
        print("draw")
        self.canvas.mode = self.canvas.CREATE
        return

    def nextImg(self):
        print("nextImg")
        return

    def preImg(self):
        print("preImg")
        return

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)

        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        # self.actions.delete.setEnabled(n_selected)
        # self.actions.duplicate.setEnabled(n_selected)
        # self.actions.copy.setEnabled(n_selected)
        # self.actions.edit.setEnabled(n_selected == 1)

    def saveFile(self, _value=False):
        # assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            print("finish save")
            # self.addRecentFile(filename)
            # self.setClean()

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = os.path.basename(os.path.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = os.path.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = os.path.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
            )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def saveLabels(self, filename):
        print(filename)

        lf = LabelFile()
        #
        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label= s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags={},
                )
            )
            return data

        shapes = [format_shape(item) for item in self.canvas.shapes]
        flags = {}
        try:
            print("%s label shapes"%(len(self.canvas.shapes)))
            print(self.imagePath)
            print(os.path.dirname(filename))
            imagePath = os.path.relpath(self.imagePath, os.path.dirname(filename))
            print(imagePath)
            imageData = self.imageData

            if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            print("start save")
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf

            return True
        except ValueError:
            return False

    def currentPath(self):
        return os.path.dirname(str(self.filename)) if self.filename else "."

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    #
    def labelUpdate(self):
        if self.canvas.hShape.label:
            previous_label = self.canvas.hShape.label
            print("already have a label [%s] and edit" % previous_label)
            self.labelDialog.edit.setText(previous_label)
            text, flags, group_id = self.labelDialog.popUp()
            if text:
                self.canvas.hShape.label = text
                print("update shape label to %s" % self.canvas.hShape.label)
                self._update_shape_color(self.canvas.hShape)
                self.labelDialog.addLabelHistory(self.canvas.hShape.label)
                item = self.currentItem()
                item.setText(self.canvas.hShape.label)

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        item = self.uniqLabelList.findItemByLabel(label)
        # print(item)
        if item is None:
            item = self.uniqLabelList.createItemFromLabel(label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(label)
            self.uniqLabelList.setItemLabel(item, label, rgb)
        label_id = self.uniqLabelList.indexFromItem(item).row() + 1
        return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def newShape(self):
        flags = {}
        group_id = None
        text = None
        if not text:
            print('new shape and label')
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id = self.labelDialog.popUp()
            print(text)
            print(flags)
            print(group_id)
            self.labelDialog.edit.setText(previous_text)
            print("finish new shape and label")
        if text:
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            self._update_shape_color(shape)
            self.labelDialog.addLabelHistory(shape.label)
            label_list_item = LabelListWidgetItem(shape.label, shape)
            self.labelList.addItem(label_list_item)
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def cropImg(self):
        # print("croping mode")
        self.canvas.mode = self.canvas.CROP
        print(self.canvas.mode)

    def cropImgDig(self):
        msg = self.cropDialog.popUp()
        if msg:
            print(self.filename)
            img_crop_name = self.canvas.cropImage(img_url=self.filename, pts=self.canvas.cropPoints)
            self.filename = img_crop_name
            if self.filename:
                self.loadFile(self.filename)





