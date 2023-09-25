from qtpy import QtWidgets

class ZoomWidget(QtWidgets.QSpinBox):
    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()
        self.setRange(1, 1000) # QSpinBox, QAbstractSpinBox
        self.setSuffix(" %") # QSpinBox, QAbstractSpinBox
        self.setValue(value) # QSpinBox, QAbstractSpinBox
