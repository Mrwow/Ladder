# -*- coding: utf-8 -*-
import functools
import os
import math
import imgviz
from shutil import copy2
from collections import Counter

from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Qt

from .__init__ import __appname__
from ladder.widgets import Canvas, ZoomWidget, FileDialogPreview, \
    LabelFile, Shape, LabelDialog, UniqueLabelQListWidget,\
    LabelListWidget,LabelListWidgetItem, CropDialog, TrainWidget, DetectWidget
from ladder.actions import baseAction
from ladder.utils import jsonToYolo, checkBox, checkBox_batch, checkBoxImg

LABEL_COLORMAP = imgviz.label_colormap()
# LABEL_COLORMAP = [[0,0,0],
#                   [128,0,0],
#                   [0,128,0],
#                   [128,128,0],
#                   [0,0,128]]


def _safe_shape_points(shape):
    """Return iterable of QPointF-like points for a shape, best-effort."""
    pts = None
    # Common patterns
    for attr in ("points", "pts", "vertices"):
        if hasattr(shape, attr):
            pts = getattr(shape, attr)
            try:
                # some implementations use a method points()
                if callable(pts):
                    pts = pts()
            except Exception:
                pass
            if pts:
                return pts
    # Fallback: dict-like
    try:
        if isinstance(shape, dict) and "points" in shape:
            return shape["points"]
    except Exception:
        pass
    return []


def _shape_bbox_xywh(shape):
    """Compute (x, y, w, h) from shape geometry, best-effort."""
    # Preferred: boundingRect()
    try:
        if hasattr(shape, "boundingRect"):
            r = shape.boundingRect()
            return float(r.left()), float(r.top()), float(r.width()), float(r.height())
    except Exception:
        pass

    pts = _safe_shape_points(shape)
    xs, ys = [], []
    for p in pts:
        try:
            # QPointF
            xs.append(float(p.x()))
            ys.append(float(p.y()))
        except Exception:
            # tuple/list [x,y]
            try:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            except Exception:
                continue
    if not xs or not ys:
        return None, None, None, None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, y0, (x1 - x0), (y1 - y0)




def _shape_diag(w, h):
    if w is None or h is None:
        return None
    try:
        return round(float((w * w + h * h) ** 0.5))
    except Exception:
        return None
def _shape_confidence(shape):
    """Extract prediction confidence if present; otherwise return None."""
    # attribute-based
    for attr in ("confidence", "conf", "score", "prob", "probability"):
        try:
            if hasattr(shape, attr):
                v = getattr(shape, attr)
                if callable(v):
                    v = v()
                if v is not None:
                    return float(v)
        except Exception:
            pass

    # dict-like metadata
    for key in ("confidence", "conf", "score", "prob", "probability"):
        try:
            if isinstance(shape, dict) and key in shape:
                return float(shape[key])
        except Exception:
            pass

    # Some implementations store extra data in `other_data` or similar
    for meta_attr in ("other_data", "meta", "extra", "attributes"):
        try:
            if hasattr(shape, meta_attr):
                meta = getattr(shape, meta_attr)
                if isinstance(meta, dict):
                    for key in ("confidence", "conf", "score", "prob", "probability"):
                        if key in meta:
                            return float(meta[key])
        except Exception:
            pass

    return None


def _format_conf(v):
    if v is None:
        return ""
    try:
        return round(float(v), 3)
    except Exception:
        return ""


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



class MainWindow(QtWidgets.QMainWindow):
    filename = None
    print("start--")
    def __init__(self,filename=None,output_dir=None,output_file=None):
        super(MainWindow,self).__init__()
        self.setWindowTitle(__appname__)
        self.filename = filename
        self.detect_shapes = None
        self.output_dir = output_dir
        self.labelFile = None
        self._noSelectionSlot = False
        self.image = QtGui.QImage()
        self.imagePath = None
        self.otherData = None
        self.output_file = output_file
        self.lastOpenDir = None
        self.trainFolder = None
        self.shape_color_rgb_counter = Counter()

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
        # Update label-list metrics after geometry edits (move/resize/reshape)
        self.canvas.shapeMoved.connect(self.shapeEdited)
        self.setCentralWidget(scrollAreaForCanvas)
        self.canvas.cropImgDig.connect(self.cropImgDig)

        # label dialog for label input and edit
        self.labelDialog = LabelDialog(parent=self)
        self.cropDialog = CropDialog(parent=self)

        # polygon label list
        self.labelList = LabelListWidget()
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.shape_dock = QtWidgets.QDockWidget(
            self.tr("Labels List"), self
        )
        # print(f'===={self.labelList.}=====')
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)
        self.shape_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.shape_dock)

        # uniq label list for hold on
        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.itemDoubleClicked.connect(self._on_unique_label_activate)
        self.uniqLabelList_dock = QtWidgets.QDockWidget(
            self.tr("Unique Labels"), self
        )
        self.uniqLabelList_dock.setWidget(self.uniqLabelList)
        self.uniqLabelList_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.uniqLabelList_dock)

        # train widget
        self.trainWidget = TrainWidget()
        self.train_dock = QtWidgets.QDockWidget(
            self.tr("Training Setting"), self
        )
        self.train_dock.setWidget(self.trainWidget)
        self.train_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.train_dock)

        # Detection widget
        self.detectWidget = DetectWidget()
        self.detect_dock = QtWidgets.QDockWidget(
            self.tr("Prediction Setting"), self
        )
        self.detect_dock.setWidget(self.detectWidget)
        self.detect_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.detect_dock)

        # Top toolbar button actions
        action = functools.partial(baseAction, self)
        open_file = action("&Open",'openFile',self.openFile)
        btn_open_file = QtWidgets.QToolButton()
        btn_open_file.setDefaultAction(open_file)

        open_dir = action("Open a folder", "openDir", self.openDir)
        btn_open_dir = QtWidgets.QToolButton()
        btn_open_dir.setDefaultAction(open_dir)

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

        edit_shape = action(
            "&Edit", 
            "edit", 
            self.editShape, 
            )
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

        train_file = action("&Yolo Format", "train", self.train_file_format)
        btn_train = QtWidgets.QToolButton()
        btn_train.setDefaultAction(train_file)

        resetParas = action("&Reset", "reset", self.resetTrainDetectParas)
        btn_reset = QtWidgets.QToolButton()
        btn_reset.setDefaultAction(resetParas)

        self.actions = struct(
            edit_shape=edit_shape, 
            draw_rect=draw_rect,
            crop_img=crop_img
            )

        # Top toolbar
        toolbar = QtWidgets.QToolBar()
        toolbar.layout().setSpacing(0)
        toolbar.layout().setContentsMargins(0,0,0,0)
        toolbar.setContentsMargins(0,0,0,0)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        toolbar.addWidget(btn_open_file)
        toolbar.addWidget(btn_zoom_in)
        toolbar.addWidget(btn_zoom_out)
        toolbar.addWidget(btn_draw_rect)
        toolbar.addWidget(btn_edit_shape)
        toolbar.addWidget(btn_crop_img)
        # toolbar.addWidget(btn_next_img)
        # toolbar.addWidget(btn_pre_img)
        # toolbar.addWidget(btn_open_dir)
        # toolbar.addWidget(btn_detect)
        toolbar.addWidget(btn_save_file)
        toolbar.addWidget(btn_del_shape)
        # toolbar.addWidget(btn_train)
        toolbar.addWidget(btn_reset)
        self.addToolBar(Qt.TopToolBarArea,toolbar)


        self.settings = QtCore.QSettings("ladder", "ladder")
        self.window_size = (1200, 800)
        size = self.settings.value("window/size", QtCore.QSize(self.window_size[0],self.window_size[1]))
        self.resize(size)

    #<<<<<<<<<<<<<<<<open file>>>>>>>>>>>>>>>
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
                self.detectWidget.singleImg = self.filename
                self.trainFolder = os.path.join(os.path.dirname(self.filename), "labels")
                # if not os.path.exists(self.trainFolder):
                #     os.makedirs(self.trainFolder)
                # copy2(self.filename,self.trainFolder)
        print(f"open img {self.filename}")


    def loadFile(self,filename=None):
        print("load file to canvas++")
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
            # self.uniqLabelList.clear()
            self.loadLabels(self.labelFile.shapes)
        self.canvas.setEnabled(True)
        self.zoomValueInitial()
        self.canvas.update()
        # self._refresh_unique_label_counts()


    def loadLabels(self, shapes):
        self.shape_color_rgb_counter = Counter()
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            other_data = shape["other_data"]
            # print(points)

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
            )
            for x, y in points:
                try:
                    shape.addPoint(QtCore.QPointF(x, y))
                except :
                    print("find a error in json here")

            shape.close()
            s.append(shape)
        self.loadShapes(s)


    def loadShapes(self, shapes, replace=True):
        self.labelList.clear()
        shape_color_rgb_counter = Counter()
        shape_label_list = []
        for shape in shapes:
            shape_color_rgb = self._update_shape_color(shape)
            # add into labelDialog
            self.labelDialog.addLabelHistory(shape.label)
            # add into label list (multi-column, sortable)
            x, y, w, h = _shape_bbox_xywh(shape)
            conf = _format_conf(_shape_confidence(shape))
            label_list_item = LabelListWidgetItem(shape.label, shape)
            cols = [
                QtGui.QStandardItem("" if x is None else str(int(round(x)))),
                QtGui.QStandardItem("" if y is None else str(int(round(y)))),
                QtGui.QStandardItem("" if w is None else str(int(round(w)))),
                QtGui.QStandardItem("" if h is None else str(int(round(h)))),
                QtGui.QStandardItem("" if _shape_diag(w, h) is None else str(round(_shape_diag(w, h), 3))),
                QtGui.QStandardItem("" if conf == "" else str(conf)),
            ]
            # ensure numeric sorting for numeric columns
            for it, val in zip(cols, [x, y, w, h, _shape_diag(w, h), None if conf=="" else float(conf)]):
                it.setEditable(False)
                if isinstance(val, (int, float)) and val is not None:
                    it.setData(float(val), Qt.EditRole)
            self.labelList.addItem(label_list_item, cols)
            shape_color_rgb_counter[shape.label] = shape_color_rgb
            shape_label_list.append(shape.label)


        # 3) Compute the current count of this label from the shape list model
        print(shape_color_rgb_counter)
        shape_label_num_counter = Counter(shape_label_list)
        print(shape_label_num_counter)
        self.uniqLabelList.clear()
        for key in shape_label_num_counter:
            print(f'creat uniq label list for {key}')
            item = self.uniqLabelList.createItemFromLabel(key)
            self.uniqLabelList.addItem(item)
            self.uniqLabelList.setItemLabel(item, key, shape_color_rgb_counter[key], shape_label_num_counter[key])

        self.canvas.loadShapes(shapes, replace=replace)
        self.shape_color_rgb_counter = shape_color_rgb_counter

    def _update_shape_color(self, shape):
        label = shape.label
        print(f'update the color for the {label}')
        if label not in self.shape_color_rgb_counter:
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            r, g, b = LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
            print(f"length of the LABEL_COLORMAP is {len(LABEL_COLORMAP)} when label_id is {label_id} and index is {label_id % len(LABEL_COLORMAP)}")
            self.shape_color_rgb_counter[shape.label] = (r, g, b)
        else:
            print(f"shape_color_rgb_counter is {self.shape_color_rgb_counter}")
            r, g, b = self.shape_color_rgb_counter[label]

        print(f"r g b is {r}, {g}, {b} for shape {shape.label}")
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        return (r,g,b)


    def openDir(self):
        print("open dir")
        return
    #<<<<<<<<<<<<<<<<open file>>>>>>>>>>>>>>>

    #<<<<<<<<<<<<<<<< zoom >>>>>>>>>>>>>>>
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
    #<<<<<<<<<<<<<<<< zoom >>>>>>>>>>>>>>>

    #<<<<<<<<<<<<<<<<save file>>>>>>>>>>>>>>>
    def saveFile(self, _value=False):
        print(f"shape length is {len(self.canvas.shapes)}")
        try:
            if len(self.canvas.shapes) < 1:
                print("cannot save empty json")
                return
            # else:
            #     basename = os.path.basename(os.path.splitext(self.filename)[0])
            #     default_labelfile_name = os.path.join(self.currentPath(), basename + LabelFile.suffix)
            #     print(default_labelfile_name)
            #     self.saveFile(default_labelfile_name)
            #     print(f"save {len(self.canvas.shapes)} shapes to the {default_labelfile_name}")

            if self.labelFile:
                # overwrite when in directory
                self._saveFile(self.labelFile.filename)
            elif self.output_file:
                self._saveFile(self.output_file)
                self.close()
            else:
                basename = os.path.basename(os.path.splitext(self.filename)[0])
                default_labelfile_name = os.path.join(self.currentPath(), basename + LabelFile.suffix)
                print(default_labelfile_name) 
                self._saveFile(default_labelfile_name)

        except Exception as e:
            print(f"Error: {e}")
            pass

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            pass
            # print(f"label file will save to {filename}")
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
        w = self.image.width()
        h = self.image.height()
        print(f"start save json file {filename}")

        lf = LabelFile()
        #
        def checkPoints(points, w, h):
            # for differnt way to drone bbox
            print(f"s.point has {len(points)} points")
            new_box= []
            x1 = points[0].x()
            y1 = points[0].y()
            x2 = points[1].x()
            y2 = points[1].y()
            print(f"x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}")

            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = max(x2, 0)
            y2 = max(y2, 0)
            print(f"x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}")
            box = [[x1,y1],[x2,y2]]

            if x1 == x2 or y1 == y2:
                pass
            elif x1 < x2 and y1 < y2:
                new_box = box
            elif x1 > x2 and y1 > y2:
                new_box = [[x2,y2],[x1,y1]]
            elif x1 < x2 and y1 > y2:
                new_box = [[x1,y2],[x2,y1]]
            elif x1 > x2 and y1 < y2:
                new_box = [[x2,y1],[x1,y2]]
            return  new_box


        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label= s.label,
                    # points=[(p.x(), p.y()) for p in s.points],
                    points= checkPoints(points=s.points, w=w, h = h),
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags={},
                )
            )
            return data

        shapes = [format_shape(item) for item in self.canvas.shapes]
        # shapes = [checkBoxImg(item,img=self.filename) for item in self.canvas.shapes]

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
            self._refresh_unique_label_counts()

            return True
        except ValueError:
            return False
    #<<<<<<<<<<<<<<<<save file>>>>>>>>>>>>>>>

    #<<<<<<<<<<<<<<<< shape and label >>>>>>>>>>>>>>>
    def currentPath(self):
        return os.path.dirname(str(self.filename)) if self.filename else "."

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

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
                self._refresh_unique_label_counts()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def _refresh_unique_label_counts(self):
        """Rebuild the unique-label list from the current contents of labelList.

        This is called when opening a new file or after bulk edits so that:
        - all previous unique items are removed
        - new unique labels are discovered from the shape list
        - each unique label shows its current count
        """
        # 1) Clear all previous unique items
        self.uniqLabelList.clear()

        # 2) Count labels from the label list (all shapes)
        counts = Counter()
        root = self.labelList.model().invisibleRootItem()
        for rrow in range(root.rowCount()):
            it = root.child(rrow, 0)
            if it is None:
                continue
            shape = it.data(Qt.UserRole)
            lab = getattr(shape, "label", None) or it.text()
            if lab:
                counts[lab] += 1

        # 3) Create unique items for each label and set text + color + count
        for idx, (lab, cnt) in enumerate(sorted(counts.items())):
            print(f"before refresh the unique label count, check the color counter {self.shape_color_rgb_counter}")
            uitem = self.uniqLabelList.createItemFromLabel(lab)
            self.uniqLabelList.addItem(uitem)
            # pick a color for this label (keep your existing colormap logic)
            # r, g, b = LABEL_COLORMAP[(idx + 1) % len(LABEL_COLORMAP)]
            color_uniq = self.shape_color_rgb_counter[lab]
            # show:  Label (N)  ●
            self.uniqLabelList.setItemLabel(uitem, lab, color_uniq, cnt)


    def _on_unique_label_activate(self, uitem):
        old_label = uitem.data(Qt.UserRole)
        if not old_label:
            return

        new_label, ok = QtWidgets.QInputDialog.getText(
            self, "Rename label", f"Rename '{old_label}' to:"
        )
        if not ok:
            return

        new_label = new_label.strip()
        if not new_label or new_label == old_label:
            return

        self._rename_label_globally(old_label, new_label)


    def _rename_label_globally(self, old_label: str, new_label: str):
        """
        Rename all shapes that use `old_label` to `new_label`.
        Updates labelList items, underlying shape objects (if they have .label),
        and keeps the unique label list consistent (merging if needed).
        """
        # 1) Rename in the per-shape list (LabelListWidget)
        root = self.labelList.model().invisibleRootItem()
        renamed_count = 0
        for r in range(root.rowCount()):
            it = root.child(r, 0)
            if it is None:
                continue
            shape = it.data(Qt.UserRole)
            # authoritative label comes from shape if available
            lab = getattr(shape, "label", None) or it.text()
            if lab == old_label:
                # update underlying shape object (if present)
                if shape is not None and hasattr(shape, "label"):
                    setattr(shape, "label", new_label)
                # update visible text of the list item
                it.setText(new_label)
                renamed_count += 1

        if renamed_count == 0:
            # Nothing matched; still make sure the unique list is sane
            self._refresh_unique_label_counts()
            return

        # 2) Update the unique labels list
        # If a unique item for the new label already exists, we will MERGE
        u_old = self.uniqLabelList.findItemByLabel(old_label)
        u_new = self.uniqLabelList.findItemByLabel(new_label)

        if u_new is None and u_old is not None:
            # Retitle the old unique item to the new label
            u_old.setData(Qt.UserRole, new_label)
        elif u_new is not None and u_old is not None and u_new is not u_old:
            # Merge: remove the old unique item; the counts will be recomputed anyway
            idx = self.uniqLabelList.row(u_old)
            self.uniqLabelList.takeItem(idx)
            u_old = None

        # 3) Recompute counts & redraw unique labels
        if new_label not in self.shape_color_rgb_counter:
            item = self.uniqLabelList.findItemByLabel(new_label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(new_label)
                self.uniqLabelList.addItem(item)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            r, g, b = LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
            self.shape_color_rgb_counter[new_label] = (r, g, b)
        else:
            r, g, b = self.shape_color_rgb_counter[old_label]

        print(f"check color before rename labels {self.shape_color_rgb_counter}")
        self.loadShapes(self.canvas.shapes)
        self._refresh_unique_label_counts()


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

    #
    def labelUpdate(self):
        if self.canvas.hShape.label:
            previous_label = self.canvas.hShape.label
            self.labelDialog.edit.setText(previous_label)
            text, flags, group_id = self.labelDialog.popUp()
            if text:
                self.canvas.hShape.label = text
                # Update label list row immediately
                try:
                    item = self.labelList.findItemByShape(self.canvas.hShape)
                    item.setText(text)
                    x, y, w, h = _shape_bbox_xywh(self.canvas.hShape)
                    conf = _format_conf(_shape_confidence(self.canvas.hShape))
                    self.labelList.setRowData(item, {
                        'x': '' if x is None else int(round(x)),
                        'y': '' if y is None else int(round(y)),
                        'w': '' if w is None else int(round(w)),
                        'l': '' if h is None else int(round(h)),
                        'diag': '' if _shape_diag(w, h) is None else float(_shape_diag(w, h)),
                        'conf': '' if conf == '' else float(conf),
                    })
                except Exception:
                    pass
                self._update_shape_color(self.canvas.hShape)
                self.labelDialog.addLabelHistory(self.canvas.hShape.label)
                item = self.currentItem()
                item.setText(self.canvas.hShape.label)
                self._refresh_unique_label_counts()


    def shapeEdited(self):
        """Called after a shape geometry edit is committed on the canvas."""
        shape = getattr(self.canvas, 'hShape', None)
        if shape is None:
            return
        try:
            item = self.labelList.findItemByShape(shape)
        except Exception:
            return

        x, y, w, h = _shape_bbox_xywh(shape)
        diag = _shape_diag(w, h)
        conf = _format_conf(_shape_confidence(shape))
        self.labelList.setRowData(item, {
            'x': '' if x is None else int(round(x)),
            'y': '' if y is None else int(round(y)),
            'w': '' if w is None else int(round(w)),
            'l': '' if h is None else int(round(h)),
            'diag': '' if diag is None else int(round(diag)),
            'conf': '' if conf == '' else float(conf),
        })


    def newShape(self):
        text, flags, group_id = self.labelDialog.popUp()
        print(self.canvas.current)

        if text:
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            self._update_shape_color(shape)
            self.labelDialog.addLabelHistory(shape.label)
            x, y, w, h = _shape_bbox_xywh(shape)
            conf = _format_conf(_shape_confidence(shape))
            label_list_item = LabelListWidgetItem(shape.label, shape)
            diag = _shape_diag(w, h)
            cols = [
                QtGui.QStandardItem("" if x is None else str(int(round(x)))),
                QtGui.QStandardItem("" if y is None else str(int(round(y)))),
                QtGui.QStandardItem("" if w is None else str(int(round(w)))),
                QtGui.QStandardItem("" if h is None else str(int(round(h)))),
                QtGui.QStandardItem("" if _shape_diag(w, h) is None else str(round(_shape_diag(w, h), 3))),
                QtGui.QStandardItem("" if conf == "" else str(conf)),
            ]
            for it, val in zip(cols, [x, y, w, h, _shape_diag(w, h), None if conf=="" else float(conf)]):
                it.setEditable(False)
                if isinstance(val, (int, float)) and val is not None:
                    it.setData(float(val), Qt.EditRole)
            self.labelList.addItem(label_list_item, cols)
            print("finish new shape and label")
        else:
            print("no label, need to cancle this new shape adding")
            # remove the last move point
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()
            # reset the first point
            self.canvas.current = None
            self.canvas.update()
        self._refresh_unique_label_counts()

    #<<<<<<<<<<<<<<<< shape and label >>>>>>>>>>>>>>>

    #<<<<<<<<<<<<<<<<crop image>>>>>>>>>>>>>>>
    def cropImg(self):
        self.canvas.mode = self.canvas.CROP
        print(self.canvas.mode)
        self.actions.crop_img.setEnabled(False)
        self.actions.draw_rect.setEnabled(True)
        self.actions.edit_shape.setEnabled(True)


    def cropImgDig(self):
        msg = self.cropDialog.popUp()
        if msg:
            print(self.filename)
            if not os.path.exists(self.trainFolder):
                os.makedirs(self.trainFolder)
            img_crop_name = self.canvas.cropImage(img_url=self.filename, pts=self.canvas.cropPoints, out_dir=self.trainFolder)
            self.filename = img_crop_name
            if self.filename:
                self.loadFile(self.filename)
    #<<<<<<<<<<<<<<<<crop image>>>>>>>>>>>>>>>

    def editShape(self):
        print("edit")
        self.canvas.mode = self.canvas.EDIT
        self.actions.edit_shape.setEnabled(False)
        self.actions.draw_rect.setEnabled(True)
        self.actions.crop_img.setEnabled(True)
        return

    def drawRec(self):
        print("draw")
        self.canvas.mode = self.canvas.CREATE
        self.actions.edit_shape.setEnabled(True)
        self.actions.crop_img.setEnabled(True)
        self.actions.draw_rect.setEnabled(False)
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

    def resetTrainDetectParas(self):
        self.trainWidget.imgSize.setText("608")
        self.trainWidget.epoch.setText("100")
        self.detectWidget.imgSize.setText("608")
        self.detectWidget.iou.setText("0.6")
        self.detectWidget.conf.setText("0.25")
        self.detectWidget.overlap.setText("0.25")
        self.detectWidget.slice.setText("1000")
        # print("reset the parameters in the train and detection")


    def train_file_format(self):
        print(self.filename)
        training_data_fd = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select training data folder",
        )
        print(f"training data folder is {training_data_fd}")
        if training_data_fd:
            train_data_dict = jsonToYolo(training_data_fd)
        else:
            pass