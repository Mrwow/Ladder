# -*- coding: utf-8 -*-
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import os

def baseAction(parent,text,icon_name,slot,shortcut=None, tip=None, checkable=False,enabled=True,checked=False):
    base_action = QtWidgets.QAction(text)
    here = os.path.dirname(os.path.abspath(__file__))
    icons_path = os.path.join(here,'icons')
    icon = QtGui.QIcon(os.path.join(icons_path,"%s.png"%icon_name))
    base_action.setIcon(icon)
    base_action.triggered.connect(slot)
    if checkable:
        base_action.setCheckable(True)
    base_action.setEnabled(enabled)
    base_action.setChecked(checked)
    return base_action
