# -*- coding: utf-8 -*-
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from .shape import Shape, distance

import numpy as np
import cv2
from ladder.utils import cropJson
import os

CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor
CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor

class Canvas(QtWidgets.QWidget):
    CREATE, EDIT, CROP = 0, 1, 2
    _createMode = "polygon"
    _fill_drawing = False

    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    newShape = QtCore.Signal()
    shapeMoved = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    labelUpdate = QtCore.Signal()
    cropImgDig = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__()
        self.shapes = []
        self.pixmap = QtGui.QPixmap()
        self.painter = QtGui.QPainter()
        self.scale = 1
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None

        self.prevMovePoint = QtCore.QPoint()
        self.selectedShapes = []
        self.prevPoint = QtCore.QPoint()
        self.prevhVertex = None
        self.prevhEdge = None
        self.prevhShape = None
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.num_backups = kwargs.pop("num_backups", 10)
        self.shapesBackups = []

        self.visible = {}
        self.mode = self.EDIT
        self.createMode = "rectangle"
        self.line = Shape()
        self.cropRec = None
        self.cropPoints = []
        self.hideBackround = False
        self._hideBackround = False
        self.snapping = True
        self.movingShape = False
        self.hShapeIsSelected = False
        self.selectedShapesCopy = []
        self.selectedShapes = []

        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)


    def paintEvent(self,event):
        p = self.painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0,0,self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(
                    shape
            ):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)

        if self.current:
            self.current.paint(p)
            self.line.paint(p)

        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if self.cropRec:
            self.cropRec.paint(p)
            self.line.paint(p)

        if (
            self.fillDrawing()
            and self.createMode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)

        p.end()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    # pixmap size
    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSizeHint()

    def minimumSizeHint(self) -> QtCore.QSize:
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    # postion transform with the scale value
    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(x, y)

    # Cursor shape change under differnt usages
    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def selectedVertex(self):
        return self.hVertex is not None

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(x, y)

    def closeEnough(self, p1, p2):
        return distance(p1 - p2) < (self.epsilon / self.scale)

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def selectedEdge(self):
        return self.hEdge is not None

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        print(shape.selected)
                        print("want select this")
                        if multiple_selection_mode:
                            print("update shapes")
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPoint(x1, y1), QtCore.QPoint(x2, y2)

    def fillDrawing(self):
        return self._fill_drawing

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPoint((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QtCore.QPoint(x2, y2))
                yield d, i, (x, y)

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()


    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def croping(self):
        return self.mode == self.CROP

    def mouseMoveEvent(self, event):
        pos = self.transformPos(event.localPos())
        self.prevMovePoint = pos
        self.restoreCursor()
        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)

            if not self.current:
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                    self.snapping
                    and len(self.current) > 1
                    and self.createMode == "polygon"
                    and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)

            if self.createMode in ["polygon", "linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.close()

            self.repaint() # paintEvent()
            self.current.highlightClear()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & event.buttons():
            if self.selectedVertex():
                # print("111")
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                # print("222")
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # hovering over to highlight
        if self.editing():
            for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
                index = shape.nearestVertex(pos, self.epsilon / self.scale)
                index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
                if index is not None:
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex = index
                    self.prevhShape = self.hShape = shape
                    self.prevhEdge = self.hEdge
                    self.hEdge = None
                    shape.highlightVertex(index, shape.MOVE_VERTEX)
                    self.overrideCursor(CURSOR_POINT)
                    self.setToolTip(self.tr("Click & drag to move point"))
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break
                elif index_edge is not None and shape.canAddPoint():
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex
                    self.hVertex = None
                    self.prevhShape = self.hShape = shape
                    self.prevhEdge = self.hEdge = index_edge
                    self.overrideCursor(CURSOR_POINT)
                    self.setToolTip(self.tr("Click to create point"))
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break
                elif shape.containsPoint(pos):
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex
                    self.hVertex = None
                    self.prevhShape = self.hShape = shape
                    self.prevhEdge = self.hEdge
                    self.hEdge = None
                    self.setToolTip(
                        self.tr("Click & drag to move shape '%s'") % shape.label
                    )
                    self.setStatusTip(self.toolTip())
                    self.overrideCursor(CURSOR_GRAB)
                    self.update()
                    break
            else:  # Nothing found, clear highlights, reset state.
                self.unHighlight()
            self.vertexSelected.emit(self.hVertex is not None)


    def mousePressEvent(self, event):
        '''
        left click and right click in windows
        one finger click and two finger click in MacOS
        '''
        pos = self.transformPos(event.localPos())
        if event.button() == QtCore.Qt.LeftButton:

            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(event.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == "point":
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                        self.selectedVertex()
                        and int(event.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(event.modifiers()) == QtCore.Qt.ControlModifier
                print("group_mode")
                print(group_mode)
                print(len(self.selectedShapes))
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                print(len(self.selectedShapes))
                self.prevPoint = pos
                self.repaint()
            elif self.croping():
                print(pos)
                if self.cropRec:
                    self.createMode == "polygon"

                    if len(self.cropRec.points) <= 3:
                        self.cropRec.addPoint(pos)
                        self.line[0] = self.cropRec[-1]
                        print(len(self.cropRec.points))
                        self.cropPoints.append(np.array([pos.x(),pos.y()]))
                    if len(self.cropRec.points) >= 4:
                        self.cropImgDig.emit()
                        self.cropRec = None
                        self.cropPoints = []
                elif not self.outOfPixmap(pos):
                    self.cropRec = Shape(shape_type="polygon")
                    self.cropRec.addPoint(pos)
                    self.cropPoints.append(np.array([pos.x(),pos.y()]))
                    print(len(self.cropRec.points))
                    self.line.points = [pos,pos]
                    self.drawingPolygon.emit(True)
                    self.update()
                self.repaint()


        elif event.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(event.modifiers()) == QtCore.Qt.ControlModifier

            if not self.selectedShapes or (
                    self.hShape is not None
                    and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            print("edit label")
            if self.editing():
                # self.labelUpdate.emit()
                self.labelUpdate.emit()
                self.update()
        elif event.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                        self.hShape is not None
                        and self.hShapeIsSelected
                        and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (
                    self.shapesBackups[-1][index].points
                    != self.shapes[index].points
            ):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False


    # rotate image by cv2 perspective transform
    def rotateImg(self, img, pts):
        """
        Perspectively project assigned area (pts) to a rectangle image
        -----
        param.
        -----
        img: 2-d numpy array
        pts: a vector of xy coordinate, length is 4. Must be in the order as:
             (NW, NE, SE, SW)
        """

        # define input coordinates
        # pts = np.float32(pts)

        # assign sorted pts
        # pt_NW, pt_NE, pt_SE, pt_SW = sortPts(pts)
        # w, h = self.pixmap.width(), self.pixmap.height()
        pt_NW = pts[0]
        pt_NE = pts[1]
        pt_SE = pts[2]
        pt_SW = pts[3]


        # estimate output dimension
        img_W = (sum((pt_NE-pt_NW)**2)**(1/2)+sum((pt_SE-pt_SW)**2)**(1/2))/2
        img_H = (sum((pt_SE-pt_NE)**2)**(1/2)+sum((pt_SW-pt_NW)**2)**(1/2))/2

        shape = (int(img_W), int(img_H))
        print(shape)

        # generate target point
        pts2 = np.float32(
            # NW,    NE,            SE,                   SW
            [[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]])

        print(pts2)
        pts = np.float32(np.row_stack(pts))
        # transformation
        H = cv2.getPerspectiveTransform(pts, pts2)
        dst = cv2.warpPerspective(img, H, (shape[0], shape[1]))
        dst = np.array(dst).astype(np.uint8)

        # return cropped image and H matrix
        return dst

    def cropImage(self,img_url, pts):
        print("start cropping")
        img = cv2.imread(img_url)
        (x1,y1) = pts[0]
        (x2,y2) = pts[1]
        (x3,y3) = pts[2]
        (x4,y4) = pts[3]

        x_min = int(min(x1,x2,x3,x4))
        x_max = int(max(x1,x2,x3,x4))
        y_min = int(min(y1,y2,y3,y4))
        y_max = int(max(y1,y2,y3,y4))

        img_crop = img[y_min:y_max,x_min:x_max]
        img_dir = os.path.dirname(str(img_url))
        img_name = os.path.basename(str(img_url)).split('.')
        img_crop_name = img_name[0] + f"_{x_min}_{y_min}_{x_max}_{y_max}." + img_name[1]
        img_crop_name = os.path.join(img_dir,img_crop_name)
        cv2.imwrite(img_crop_name,img_crop)

        json = img_url.split(".")[0] + ".json"
        if os.path.isfile(json):
            cropJson(img_url=img_url, json_url=json, pts=[x_min, y_min, x_max, y_max])

        return img_crop_name













