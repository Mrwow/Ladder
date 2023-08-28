# -*- coding: utf-8 -*-
import sys
from qtpy import QtWidgets, QtGui
from ladder.app import MainWindow
import os.path as osp


def main():
    print("starting---------")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ladder")
    iconlog = QtGui.QIcon('ladder/icons/logo.png')
    app.setWindowIcon(iconlog)
    win = MainWindow()
    win.show()
    win.raise_()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()