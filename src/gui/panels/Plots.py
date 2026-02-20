from PyQt5.QtWidgets import ( QPushButton, QGraphicsProxyWidget)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np

from src.gui.theme import MODERN_STYLESHEET, COLORS


class LocalizationErrorPlot:
    """A class to handle the localization error plotting functionality"""
    def __init__(self, parent=None):
        # Create error plot with enhanced styling
        self.error_plot = pg.PlotWidget()
        self.error_plot.setBackground(COLORS['background'])
        self.error_plot.setTitle("Localization Error Analysis", 
                                color=COLORS['text'], 
                                size='12pt', 
                                bold=True)
        
        # Enhance axis labels with better styling
        label_style = {'color': COLORS['text'], 
                      'font-size': '10pt'}
        self.error_plot.setLabel('left', 'Error Distance', 
                                units='m', 
                                **label_style)
        self.error_plot.setLabel('bottom', 'Time', 
                                units='s', 
                                **label_style)
        
        # Improve grid appearance
        self.error_plot.showGrid(x=True, y=True, alpha=0.2)
        self.error_plot.getAxis('left').setGrid(100)
        self.error_plot.getAxis('bottom').setGrid(100)
        
        # Set axis color
        self.error_plot.getAxis('bottom').setPen(pg.mkPen(color=COLORS['text'], width=1))
        self.error_plot.getAxis('left').setPen(pg.mkPen(color=COLORS['text'], width=1))
        
        # Add legend with better positioning and style
        self.error_plot.addLegend(
            offset=(-10, 10),
            labelTextColor=COLORS['text'],
            brush=pg.mkBrush(COLORS['background']),
            pen=pg.mkPen(color='#404040')
        )
        
        # Enable auto-range for Y axis with some padding
        self.error_plot.setAutoVisible(y=True)
        self.error_plot.enableAutoRange(axis='y')
        
        # Set minimum range to prevent plot from collapsing when error is very small
        self.error_plot.setYRange(0, 0.1, padding=0.1)
        
        # Initialize error curve with improved styling
        self.error_curve = self.error_plot.plot(
            pen=pg.mkPen({
                'color': '#ff5555',
                'width': 2,
                'style': Qt.SolidLine,
                'cosmetic': True
            }),
            name='Instantaneous Error',
            symbol='o',
            symbolSize=4,
            symbolBrush='#ff5555',
            symbolPen=None
        )
        
        # Initialize moving average curve with improved styling
        self.ma_curve = self.error_plot.plot(
            pen=pg.mkPen({
                'color': '#55ff55',
                'width': 3,
                'style': Qt.SolidLine,
                'cosmetic': True
            }),
            name='Moving Average',
            shadowPen=pg.mkPen('#55ff55', width=5, cosmetic=True, alpha=0.3)
        )
        
        # Add zero error reference line
        self.zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen({
                'color': '#404040',
                'width': 1,
                'style': Qt.DashLine
            })
        )
        self.error_plot.addItem(self.zero_line)

        # Initialize data storage
        self.times = []
        self.errors = []
        self.parameter_change_times = []
        self.error_plot_paused = False

    def update_error(self, simulation_time, error, ma_window):
        """Update error plot with new data point"""
        if not self.error_plot_paused and np.isfinite(error):
            self.times.append(simulation_time)
            self.errors.append(error)
            
            # Keep fixed window size
            window_size = 100
            if len(self.times) > window_size:
                self.times = self.times[-window_size:]
                self.errors = self.errors[-window_size:]
                # Also trim change markers
                self.parameter_change_times = [t for t in self.parameter_change_times 
                                            if t >= self.times[0]]
            
            # Calculate moving average
            if len(self.errors) < ma_window:
                padded_errors = [0] * (ma_window - len(self.errors)) + self.errors
                moving_avg = np.convolve(padded_errors, np.ones(ma_window)/ma_window, mode='valid')
                moving_avg = np.repeat(moving_avg[0], len(self.errors))
            else:
                moving_avg = np.convolve(self.errors, np.ones(ma_window)/ma_window, mode='same')
            
            # Filter out any non-finite values
            valid_indices = np.isfinite(self.errors)
            if np.any(valid_indices):
                valid_times = [t for t, v in zip(self.times, valid_indices) if v]
                valid_errors = [e for e, v in zip(self.errors, valid_indices) if v]
                valid_ma = [ma for ma, v in zip(moving_avg, valid_indices) if v]
                
                # Update plots
                self.error_curve.setData(valid_times, valid_errors)
                self.ma_curve.setData(valid_times, valid_ma)
                
                # Update Y range with auto-ranging
                max_error = max(max(valid_errors), max(valid_ma))
                min_error = min(min(valid_errors), min(valid_ma))
                y_min = max(0, min_error * 0.8)
                y_max = max(0.1, max_error * 1.2)
                
                current_range = self.error_plot.getViewBox().state['viewRange'][1]
                if (abs(current_range[0] - y_min) > 0.01 or 
                    abs(current_range[1] - y_max) > 0.01):
                    self.error_plot.setYRange(y_min, y_max, padding=0.1)

    def mark_parameter_change(self, simulation_time):
        """Mark a parameter change on the error plot"""
        self.parameter_change_times.append(simulation_time)
        if len(self.parameter_change_times) > 10:
            self.parameter_change_times = self.parameter_change_times[-10:]
        
        # Add vertical line for parameter change
        if self.times and self.times[0] <= simulation_time <= self.times[-1]:
            change_line = pg.InfiniteLine(
                pos=simulation_time, 
                angle=90, 
                pen=pg.mkPen('w', width=1, style=Qt.DashLine)
            )
            self.error_plot.addItem(change_line)
            # Remove old line after a delay
            QTimer.singleShot(2000, lambda: self.error_plot.removeItem(change_line))

    def clear_data(self):
        """Clear all error data"""
        self.times.clear()
        self.errors.clear()
        self.parameter_change_times.clear()
        self.error_curve.setData([], [])
        self.ma_curve.setData([], [])

    def get_widget(self):
        """Return the plot widget"""
        return self.error_plot

    def pause(self):
        """Pause error plotting"""
        self.error_plot_paused = True

    def resume(self):
        """Resume error plotting"""
        self.error_plot_paused = False

    def reset_plot(self):
        """Fully reset the error plot"""
        # Clear data
        self.times = []
        self.errors = []
        self.parameter_change_times = []
        
        # Clear existing curves but don't remove them
        self.error_curve.setData([], [])
        self.ma_curve.setData([], [])
        
        # Reset Y range to default
        self.error_plot.setYRange(0, 0.1, padding=0.1)
        
        # Reset pause state
        self.error_plot_paused = False
        
        # Remove any parameter change lines
        for item in self.error_plot.items():
            if isinstance(item, pg.InfiniteLine) and item != self.zero_line:
                self.error_plot.removeItem(item)
        
        # Make sure zero line is present
        if self.zero_line not in self.error_plot.items():
            self.error_plot.addItem(self.zero_line)