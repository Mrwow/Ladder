# -*- coding: utf-8 -*-
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or "")
        self.setShape(shape)

        # self.setCheckable(True)
        # self.setCheckState(Qt.Checked)
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignBottom)

    def clone(self):
        return LabelListWidgetItem(self.text(), self.shape())

    def setShape(self, shape):
        self.setData(shape, Qt.UserRole)

    def shape(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class StandardItemModel(QtGui.QStandardItemModel):

    itemDropped = QtCore.Signal()

    def removeRows(self, *args, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        self.itemDropped.emit()
        return ret


class LabelListWidget(QtWidgets.QTreeView):

    itemDoubleClicked = QtCore.Signal(LabelListWidgetItem)
    itemSelectionChanged = QtCore.Signal(list, list)

    def __init__(self):
        super(LabelListWidget, self).__init__()
        self._selectedItems = []

        self.setWindowFlags(Qt.Window)

        model = StandardItemModel()
        self.setModel(model)
        # Prototype only affects column 0 items (LabelListWidgetItem)
        model.setItemPrototype(LabelListWidgetItem())

        # View configuration: behave like a table
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)

        # Selection: select whole rows
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Drag/drop row reordering (optional)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

        # Columns
        self._headers = ["Label", "X", "Y", "Width", "Length", "Diag", "Conf."]
        model.setColumnCount(len(self._headers))
        model.setHorizontalHeaderLabels(self._headers)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in range(1, len(self._headers)):
            self.header().setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)

        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)



    def __len__(self):
        return self.model().rowCount()

    # def __getitem__(self, i):
    #     return self.model().item(i)
    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield self[i]

    def __getitem__(self, i):
        return self.model().item(i, 0)

    def __iter__(self):
        for row in range(self.model().rowCount()):
            yield self.model().item(row, 0)



    @property
    def itemDropped(self):
        return self.model().itemDropped

    @property
    def itemChanged(self):
        return self.model().itemChanged

    def itemSelectionChangedEvent(self, selected, deselected):
        # QTreeView selectionChanged reports a range of indexes across columns.
        # Convert to unique rows and return the first-column LabelListWidgetItem for each row.
        def _rows(sel):
            rows = set()
            for idx in sel.indexes():
                rows.add(idx.row())
            return sorted(rows)

        selected_items = [self.model().item(r, 0) for r in _rows(selected)]
        deselected_items = [self.model().item(r, 0) for r in _rows(deselected)]
        self.itemSelectionChanged.emit(selected_items, deselected_items)


    def itemDoubleClickedEvent(self, index):
        self.itemDoubleClicked.emit(self.model().item(index.row(), 0))

    def selectedItems(self):
        rows = sorted({i.row() for i in self.selectedIndexes()})
        return [self.model().item(r, 0) for r in rows]

    def scrollToItem(self, item):
        self.scrollTo(self.model().indexFromItem(item))

    def addItem(self, item, columns=None):
        """Add a row.
        Parameters
        ----------
        item : LabelListWidgetItem
            The first-column item. Must carry the shape in Qt.UserRole.
        columns : list[QtGui.QStandardItem] | None
            Optional items for columns 1..N-1. If omitted, blanks are inserted.
        """
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")

        row = self.model().rowCount()
        self.model().setItem(row, 0, item)

        # Fill remaining columns
        col_count = self.model().columnCount()
        if columns is None:
            columns = []
        for c in range(1, col_count):
            extra = columns[c-1] if (c-1) < len(columns) and columns[c-1] is not None else QtGui.QStandardItem("")
            extra.setEditable(False)
            self.model().setItem(row, c, extra)


    def removeItem(self, item):
        index = self.model().indexFromItem(item)
        self.model().removeRows(index.row(), 1)

    def selectItem(self, item):
        index = self.model().indexFromItem(item)
        if not index.isValid():
            return
        left = self.model().index(index.row(), 0)
        right = self.model().index(index.row(), self.model().columnCount() - 1)
        sel = QtCore.QItemSelection(left, right)
        self.selectionModel().select(sel, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    def findItemByShape(self, shape):
        for row in range(self.model().rowCount()):
            item = self.model().item(row, 0)
            if item.shape() == shape:
                return item
        raise ValueError("cannot find shape: {}".format(shape))


    def setRowData(self, item: LabelListWidgetItem, values: dict):
        """Update columns (except label text) for an existing first-column item."""
        index = self.model().indexFromItem(item)
        if not index.isValid():
            return
        row = index.row()
        # Expected keys: x, y, w, h (length), conf
        mapping = {
            1: values.get('x', ''),
            2: values.get('y', ''),
            3: values.get('w', ''),
            4: values.get('l', ''),
            5: values.get('diag', ''),
            6: values.get('conf', ''),
        }
        for col, v in mapping.items():
            it = self.model().item(row, col)
            if it is None:
                it = QtGui.QStandardItem()
                it.setEditable(False)
                self.model().setItem(row, col, it)
            # Use Qt.EditRole for numeric sorting
            if isinstance(v, (int, float)) and v != "":
                it.setData(v, Qt.EditRole)
                it.setText(str(v))
            else:
                it.setData(v, Qt.EditRole)
                it.setText("" if v is None else str(v))
    def clear(self):
        self.model().clear()
        self.model().setColumnCount(len(self._headers))
        self.model().setHorizontalHeaderLabels(self._headers)
