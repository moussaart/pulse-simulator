"""
Slider widgets - TimeWindowSlider and LabeledSlider.
Replaces duplicated time-window slider patterns in Distance_plot_window.py and imu_window.py.
"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, pyqtSignal
from src.gui.theme import COLORS


class TimeWindowSlider(QWidget):
    """
    A horizontal slider with a label showing the current value + suffix.
    Commonly used for 'time window' controls in plot windows.
    Emits value_changed(int) when slider moves.
    """

    value_changed = pyqtSignal(int)

    def __init__(self, min_val=1, max_val=60, default=10, suffix="s", label_text="Time Window:", parent=None):
        super().__init__(parent)
        self.suffix = suffix
        self._setup_ui(min_val, max_val, default, label_text)

    def _setup_ui(self, min_val, max_val, default, label_text):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        label.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default)
        self.slider.valueChanged.connect(self._on_changed)
        layout.addWidget(self.slider)

        self.value_label = QLabel(f"{default}{self.suffix}")
        self.value_label.setStyleSheet(f"color: {COLORS['text_bright']}; min-width: 30px;")
        layout.addWidget(self.value_label)

    def _on_changed(self, value):
        self.value_label.setText(f"{value}{self.suffix}")
        self.value_changed.emit(value)

    def value(self):
        """Return the current slider value."""
        return self.slider.value()

    def setValue(self, val):
        """Set the slider value."""
        self.slider.setValue(val)


class LabeledSlider(QWidget):
    """
    Generic slider with a paired value label.
    Emits value_changed(int) when the slider moves.
    """

    value_changed = pyqtSignal(int)

    def __init__(self, label_text="", min_val=0, max_val=100, default=50,
                 suffix="", orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.suffix = suffix
        self._setup_ui(label_text, min_val, max_val, default, orientation)

    def _setup_ui(self, label_text, min_val, max_val, default, orientation):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if label_text:
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {COLORS['text']};")
            layout.addWidget(label)

        self.slider = QSlider(orientation)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default)
        self.slider.valueChanged.connect(self._on_changed)
        layout.addWidget(self.slider)

        self.value_label = QLabel(f"{default}{self.suffix}")
        self.value_label.setStyleSheet(f"color: {COLORS['text_bright']}; min-width: 30px;")
        layout.addWidget(self.value_label)

    def _on_changed(self, value):
        self.value_label.setText(f"{value}{self.suffix}")
        self.value_changed.emit(value)

    def value(self):
        return self.slider.value()

    def setValue(self, val):
        self.slider.setValue(val)
