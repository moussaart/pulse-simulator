from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGridLayout
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from src.gui.theme import COLORS
from src.gui.widgets import PersistentWindow, TimeWindowSlider, create_themed_plot

class IMUData:
    """
    IMUData class:
    Contains the data for the IMU.
    """
    def __init__(self):
        # Initialize as lists instead of numpy arrays
        self.timestamps = []
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []

        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []
    
    def add_measurement(self, timestamp, ax, ay, az, gx, gy, gz):
        self.timestamps.append(timestamp)
        self.acc_x.append(ax)
        self.acc_y.append(ay)
        self.acc_z.append(az)
        self.gyro_x.append(gx)
        self.gyro_y.append(gy)
        self.gyro_z.append(gz)
        
        # Keep only last 1000 measurements
        if len(self.timestamps) > 1000:
            self.timestamps = self.timestamps[-1000:]
            self.acc_x = self.acc_x[-1000:]
            self.acc_y = self.acc_y[-1000:]
            self.acc_z = self.acc_z[-1000:]
            self.gyro_x = self.gyro_x[-1000:]
            self.gyro_y = self.gyro_y[-1000:]
            self.gyro_z = self.gyro_z[-1000:]
            
class IMUWindow(PersistentWindow):
    def __init__(self, parent=None):
        super().__init__(title="IMU Data", parent=parent)
        self.parent_app = parent
        self.setGeometry(100, 100, 1200, 800)
        
        self.time_window = 10.0  # Show last 10 seconds of data
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create time window control
        self.time_control = TimeWindowSlider(
            min_val=1, max_val=30, default=10, suffix="s",
            label_text="Time Window:"
        )
        self.time_control.value_changed.connect(self.update_time_window)
        layout.addWidget(self.time_control)
        
        # Create plots
        self.create_plots(layout)
        
        # Initialize time axis
        self.sync_time_axis()
        
        # Setup update timer (30 FPS)
        self.update_timer = pg.QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(33)

    def update_time_window(self, value):
        self.time_window = float(value)
        self.sync_time_axis()

    def _create_imu_plot(self, title, y_label, color, grid, row, col):
        """Helper to create a themed IMU plot with a zero line."""
        plot = create_themed_plot(title=title, y_label=y_label, x_label="Time", x_units="s")
        plot.enableAutoRange(y=True)
        plot.addLine(y=0, pen=pg.mkPen('w', width=1, style=Qt.DashLine))
        grid.addWidget(plot, row, col)
        curve = plot.plot(pen=pg.mkPen(color, width=2))
        return plot, curve

    def create_plots(self, layout):
        plot_grid = QGridLayout()
        
        # Accelerometer plots (X, Y, Z)
        self.acc_x_plot, self.acc_x_curve = self._create_imu_plot(
            "Accelerometer X", "Acceleration (m/s²)", 'r', plot_grid, 0, 0)
        self.acc_y_plot, self.acc_y_curve = self._create_imu_plot(
            "Accelerometer Y", "Acceleration (m/s²)", 'g', plot_grid, 0, 1)
        self.acc_z_plot, self.acc_z_curve = self._create_imu_plot(
            "Accelerometer Z", "Acceleration (m/s²)", 'b', plot_grid, 0, 2)
        
        # Gyroscope plots (X, Y, Z)
        self.gyro_x_plot, self.gyro_x_curve = self._create_imu_plot(
            "Gyroscope X", "Angular Velocity (rad/s)", 'r', plot_grid, 1, 0)
        self.gyro_y_plot, self.gyro_y_curve = self._create_imu_plot(
            "Gyroscope Y", "Angular Velocity (rad/s)", 'g', plot_grid, 1, 1)
        self.gyro_z_plot, self.gyro_z_curve = self._create_imu_plot(
            "Gyroscope Z", "Angular Velocity (rad/s)", 'b', plot_grid, 1, 2)
        
        layout.addLayout(plot_grid)
    
    def sync_time_axis(self):
        """Synchronize time axis with current simulation data"""
        if not self.isVisible() or not hasattr(self.parent_app, 'tag'):
            return

        imu_data = self.parent_app.tag.imu_data
        if len(imu_data.timestamps) == 0:
            # No data yet, show initial range from 0 to time_window
            start_time = 0.0
            end_time = self.time_window
        else:
            # Show the last time_window seconds of data
            latest_time = imu_data.timestamps[-1]
            start_time = max(0.0, latest_time - self.time_window)
            end_time = latest_time
        
        # Update X range for all plots
        self.acc_x_plot.setXRange(start_time, end_time)
        self.acc_y_plot.setXRange(start_time, end_time)
        self.acc_z_plot.setXRange(start_time, end_time)
        self.gyro_x_plot.setXRange(start_time, end_time)
        self.gyro_y_plot.setXRange(start_time, end_time)
        self.gyro_z_plot.setXRange(start_time, end_time)
    
    def update_plots(self):
        if not self.isVisible() or not hasattr(self.parent_app, 'tag'):
            return

        imu_data = self.parent_app.tag.imu_data
        
        if len(imu_data.timestamps) == 0:
            # Clear plots if no data
            self.reset_plots()
            return
            
        # Get the time window of data to display
        latest_time = imu_data.timestamps[-1]
        start_time = max(0.0, latest_time - self.time_window)
        
        # Convert lists to numpy arrays and find indices for the time window
        timestamps = np.array(imu_data.timestamps)
        mask = timestamps >= start_time
        timestamps = timestamps[mask]
        
        # Update accelerometer plots
        self.acc_x_curve.setData(timestamps, np.array(imu_data.acc_x)[mask])
        self.acc_y_curve.setData(timestamps, np.array(imu_data.acc_y)[mask])
        self.acc_z_curve.setData(timestamps, np.array(imu_data.acc_z)[mask])
        
        # Update gyroscope plots
        self.gyro_x_curve.setData(timestamps, np.array(imu_data.gyro_x)[mask])
        self.gyro_y_curve.setData(timestamps, np.array(imu_data.gyro_y)[mask])
        self.gyro_z_curve.setData(timestamps, np.array(imu_data.gyro_z)[mask])
        
        # Synchronize time axis with current data
        self.sync_time_axis()
    
    def reset_plots(self):
        """Reset plots when simulation restarts"""
        # Clear all plot data
        self.acc_x_curve.setData([], [])
        self.acc_y_curve.setData([], [])
        self.acc_z_curve.setData([], [])
        self.gyro_x_curve.setData([], [])
        self.gyro_y_curve.setData([], [])
        self.gyro_z_curve.setData([], [])
        
        # Reset time axis to show from 0
        if hasattr(self, 'acc_x_plot'):
            self.sync_time_axis()
    # closeEvent inherited from PersistentWindow