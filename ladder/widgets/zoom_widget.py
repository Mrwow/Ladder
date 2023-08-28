from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class ZoomWidget(QtWidgets.QSpinBox):
    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons) #QAbstractSpinBox
        self.setRange(1, 1000) # QSpinBox, QAbstractSpinBox
        self.setSuffix(" %") # QSpinBox, QAbstractSpinBox
        self.setValue(value) # QSpinBox, QAbstractSpinBox
        self.setToolTip("Zoom Level")
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter) #QAbstractSpinBox

    def minimumSizeHint(self):
        height = super(ZoomWidget, self).minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QtCore.QSize(width, height)