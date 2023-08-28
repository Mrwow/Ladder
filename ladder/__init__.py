# -*- coding: utf-8 -*-
import sys
from qtpy import QT_VERSION

__appname__ = "ladder"
__version__ = "0.0.1"
QT4 = QT_VERSION[0] == "4"
QT5 = QT_VERSION[0] == "5"
PY2 = sys.version[0] == "2"
PY3 = sys.version[0] == "3"
