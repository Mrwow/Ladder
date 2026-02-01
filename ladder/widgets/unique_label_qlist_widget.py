# -*- encoding: utf-8 -*-

import html

from qtpy.QtCore import Qt
from qtpy import QtWidgets

# from .escapable_qlist_widget import EscapableQListWidget

class EscapableQListWidget(QtWidgets.QListWidget):
    def keyPressEvent(self, event):
        super(EscapableQListWidget, self).keyPressEvent(event)
        if event.key() == Qt.Key_Escape:
            self.clearSelection()

class UniqueLabelQListWidget(EscapableQListWidget):
    def mousePressEvent(self, event):
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemByLabel(self, label):
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                return item

    def createItemFromLabel(self, label):
        if self.findItemByLabel(label):
            raise ValueError(
                "Item for label '{}' already exists".format(label)
            )

        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.UserRole, label)
        return item

    def setItemLabel(self, item, label, color=None, count=None):
            """
            Render a label item as:  "<label> (N)  ●"
            - label: str
            - color: optional (r, g, b) tuple
            - count: optional int; if None, the count text is omitted
            """
            qlabel = QtWidgets.QLabel()

            # Build the visible text
            parts = [html.escape(label)]
            if count is not None:
                parts.append(f"({count})")

            if color is None:
                text = " ".join(parts)
            else:
                (r, g, b) = color
                text = " ".join(parts) + f' <font color="#{r:02x}{g:02x}{b:02x}">●</font>'

            qlabel.setText(text)
            qlabel.setAlignment(Qt.AlignBottom)

            item.setSizeHint(qlabel.sizeHint())
            self.setItemWidget(item, qlabel)

    def removeItemLabel(self, label: str) -> bool:
        """
        Remove the unique-label list entry for `label`.
        Returns True if an item was removed, False if not found.
        (Does NOT modify shapes; handle that in app.py if desired.)
        """
        item = self.findItemByLabel(label)
        if item is None:
            return False

        row = self.row(item)
        # If we used setItemWidget, clean up the widget to avoid leaks
        w = self.itemWidget(item)
        if w is not None:
            w.deleteLater()
        # Remove the QListWidgetItem
        self.takeItem(row)
        return True


    def removeAllItemLabels(self) -> int:
        """
        Remove ALL unique-label entries from the list.
        Returns the number of items removed.
        (Does NOT modify shapes; handle that in app.py if desired.)
        """
        removed = 0
        # Iterate backwards so row indices remain valid as we remove
        for row in reversed(range(self.count())):
            it = self.item(row)
            w = self.itemWidget(it)
            if w is not None:
                w.deleteLater()
            self.takeItem(row)
            removed += 1
        return removed


