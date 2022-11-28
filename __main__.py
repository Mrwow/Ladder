# -*- coding: utf-8 -*-
import sys

from qtpy import QtWidgets
from app import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ladder")
    win = MainWindow()
    win.show()
    win.raise_()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()