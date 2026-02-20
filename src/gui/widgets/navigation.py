"""
PlotGroupNavigation - Prev/Next navigation for paged plot grids.
Replaces duplicated navigation controls in Distance_plot_window.py and cir_window.py.
"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal, Qt
from src.gui.theme import COLORS


class PlotGroupNavigation(QWidget):
    """
    A navigation bar with Prev/Next buttons and a 'Group X/Y' label.
    Emits group_changed(int) when the current group index changes.
    """

    group_changed = pyqtSignal(int)

    def __init__(self, total_groups=1, parent=None):
        super().__init__(parent)
        self.current_group = 0
        self.total_groups = max(1, total_groups)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                padding: 4px 12px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_dim']};
            }}
        """)
        self.prev_btn.clicked.connect(self._go_prev)
        layout.addWidget(self.prev_btn)

        self.group_label = QLabel()
        self.group_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        self.group_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.group_label, 1)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setStyleSheet(self.prev_btn.styleSheet())
        self.next_btn.clicked.connect(self._go_next)
        layout.addWidget(self.next_btn)

        self._update_label()

    def set_total_groups(self, total):
        """Update total groups and reset to first group."""
        self.total_groups = max(1, total)
        self.current_group = 0
        self._update_label()

    def _go_prev(self):
        if self.current_group > 0:
            self.current_group -= 1
            self._update_label()
            self.group_changed.emit(self.current_group)

    def _go_next(self):
        if self.current_group < self.total_groups - 1:
            self.current_group += 1
            self._update_label()
            self.group_changed.emit(self.current_group)

    def _update_label(self):
        self.group_label.setText(f"Group {self.current_group + 1}/{self.total_groups}")
        self.prev_btn.setEnabled(self.current_group > 0)
        self.next_btn.setEnabled(self.current_group < self.total_groups - 1)
