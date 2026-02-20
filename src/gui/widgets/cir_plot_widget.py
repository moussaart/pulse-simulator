from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from src.gui.theme import COLORS
from src.gui.widgets.plot_helpers import create_themed_plot

# Fix for PyQtGraph alignment on high-DPI screens
import platform
import ctypes
try:
    if platform.system() == 'Windows' and float(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
except Exception:
    pass

class CIRPlotWidget(QWidget):
    def __init__(self, anchor_id, parent=None):
        super().__init__(parent)
        self.anchor_id = anchor_id
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create plot
        self.plot = create_themed_plot(
            title=f"Anchor {self.anchor_id}", 
            y_label="Amplitude",
            x_label="Time", 
            x_units="s"
        )
        
        # Curve: Init with Primary color (LOS default)
        self.curve = self.plot.plot(pen=pg.mkPen(COLORS['primary'], width=2))
        
        # First Path Line
        self.fp_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('w', width=1, style=Qt.DashLine), label='ToA')
        self.plot.addItem(self.fp_line)
        self.fp_line.hide()
        
        # Threshold Line
        self.th_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', width=1, style=Qt.DotLine), label='Thresh')
        self.plot.addItem(self.th_line)
        self.th_line.hide()

        layout.addWidget(self.plot)

    def update_data(self, time_vector, amplitude, is_los, first_path_index=None, snr_db=None, auto_scale=True, show_threshold=True):
        # Update Curve Color
        color = COLORS['primary'] if is_los else COLORS['error']
        self.curve.setPen(pg.mkPen(color, width=2))
        
        # Update Data
        self.curve.setData(time_vector, amplitude)
        
        # Auto-Scale
        if auto_scale:
            self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes)
        else:
            self.plot.disableAutoRange()
            
        # Update Markers
        if first_path_index is not None and 0 <= first_path_index < len(time_vector):
            toa = time_vector[first_path_index]
            self.fp_line.setPos(toa)
            self.fp_line.show()
            
            threshold_val = amplitude[first_path_index]
            self.th_line.setPos(threshold_val)
            self.th_line.setVisible(show_threshold)
        else:
            self.fp_line.hide()
            self.th_line.hide()

        # Update Title
        los_text = "(LOS)" if is_los else "(NLOS)"
        title = f"Anchor {self.anchor_id} - {los_text}"
        if snr_db is not None:
            title += f" [SNR: {snr_db:.1f} dB]"
        self.plot.setTitle(title, color='w')

    def reset(self):
        self.curve.setData([], [])
        self.fp_line.hide()
        self.th_line.hide()
        self.plot.setTitle(f"Anchor {self.anchor_id}", color='w')
