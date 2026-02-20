"""
PersistentWindow - Base class for plot windows that hide on close.
Replaces duplicated QMainWindow + closeEvent + stylesheet patterns in
Distance_plot_window.py, cir_window.py, and imu_window.py.
"""
from PyQt5.QtWidgets import QMainWindow
from src.gui.theme import MODERN_STYLESHEET, COLORS


class PersistentWindow(QMainWindow):
    """
    A QMainWindow that applies the dark theme and hides instead of closing.
    Subclasses should call super().__init__() and then build their UI.
    """

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        if title:
            self.setWindowTitle(title)
        self.setStyleSheet(MODERN_STYLESHEET + f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
        """)

    def closeEvent(self, event):
        """Hide instead of close to preserve state."""
        self.hide()
        event.ignore()
