from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
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

class DistancePlotWidget(QWidget):
    def __init__(self, anchor_id, time_window=10.0, parent=None):
        super().__init__(parent)
        self.anchor_id = anchor_id
        self.time_window = time_window
        
        # Data storage
        self.times = []
        self.measured_distances = []
        self.true_distances = []
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Create plot
        self.plot = create_themed_plot(
            title=f"Anchor {self.anchor_id} Distance Measurements",
            y_label="Distance (m)",
            x_label="Time (s)",
            enable_legend=True
        )
        self.plot.setXRange(0, self.time_window, padding=0)

        # Create curves
        self.measured_curve = self.plot.plot(
            pen=pg.mkPen(COLORS['primary'], width=2), 
            name='Measured Distance'
        )
        self.true_curve = self.plot.plot(
            pen=pg.mkPen(COLORS['success'], width=2), 
            name='True Distance'
        )
        
        # Legend setup
        self.legend = self.plot.plotItem.legend
        
        # NLOS indicator
        self.nlos_curve = pg.PlotDataItem(
            pen=pg.mkPen(COLORS['error'], width=2),
            name='NLOS'
        )
        self.legend.addItem(self.nlos_curve, 'NLOS')
        self.nlos_curve.hide()

        layout.addWidget(self.plot)

        # Stats Label
        self.stats_label = QLabel("RMSE: 0.000m | STD: 0.000m | Avg Err: 0.000m")
        self.stats_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold; margin-top: 5px;")
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)

    def update_data(self, time, measured, true, is_los):
        self.times.append(time)
        self.measured_distances.append(measured)
        self.true_distances.append(true)
        
        self.update_stats()
        self.update_curves(is_los)

    def update_stats(self):
        if not self.measured_distances:
            return

        measured = np.array(self.measured_distances)
        true = np.array(self.true_distances)
        
        errors = np.abs(measured - true)
        real_errors = measured - true
        avg_error = np.mean(errors)
        rmse = np.sqrt(np.mean(real_errors**2))
        std = np.std(real_errors)
        
        self.stats_label.setText(
            f"RMSE: {rmse:.3f}m | STD: {std:.3f}m | Avg Err: {avg_error:.3f}m"
        )

    def update_curves(self, is_los):
        if is_los:
            pen = pg.mkPen(COLORS['primary'], width=2)
            self.nlos_curve.hide()
        else:
            pen = pg.mkPen(COLORS['error'], width=2, style=Qt.DashLine)
            self.nlos_curve.show()
        
        self.measured_curve.setPen(pen)
        self.measured_curve.setData(self.times, self.measured_distances)
        self.true_curve.setData(self.times, self.true_distances)
        
        # Handle scrolling
        latest_time = self.times[-1]
        
        # Smooth transition logic could be here, but direct set is safer for now. 
        # The window had smooth scrolling, maybe I should port that?
        # The window used a timer for smooth updates. 
        # For simplicity, I'll stick to direct updates first, as the user asked to extract the widget.
        # If smooth scrolling is needed, it can be added to the widget or managed by the window.
        # But wait, the window calls `smooth_update_plots` on a timer. 
        # If I extract the widget, do I keep the smoothness?
        # The user said "extract the plot part as a widget".
        # I will expose a method `set_x_range` or just handle it in `update_curves` directly for now.
        
        if latest_time > self.time_window:
            self.plot.setXRange(latest_time - self.time_window, latest_time, padding=0)
        else:
            self.plot.setXRange(0, self.time_window, padding=0)

    def set_time_window(self, window):
        self.time_window = window
        if self.times:
            latest_time = self.times[-1]
            if latest_time > self.time_window:
                self.plot.setXRange(latest_time - self.time_window, latest_time, padding=0)
            else:
                self.plot.setXRange(0, self.time_window, padding=0)
        else:
            self.plot.setXRange(0, self.time_window, padding=0)

    def reset(self):
        self.times = []
        self.measured_distances = []
        self.true_distances = []
        self.measured_curve.setData([], [])
        self.true_curve.setData([], [])
        self.stats_label.setText("RMSE: 0.000m | STD: 0.000m | Avg Err: 0.000m")
