# -*- coding: utf-8 -*-
from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

class CropDialog(QtWidgets.QDialog):

    def __init__(self,parent=None):
        super(CropDialog, self).__init__(parent)

        self.setWindowTitle("Cropping image!")
        Qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.validate)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        message = QtWidgets.QLabel("Cropping current image?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.msg = False
        self.img = None
        self.croped_img = None

    def validate(self):
        print("starting croping")
        self.msg = True
        if self.msg:
            self.accept()

    def popUp(self):
        if self.exec_():
            return self.msg
        else:
            return None


