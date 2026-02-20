from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QSlider, QGroupBox, QComboBox, QPushButton, QCheckBox,
                            QTextEdit, QSpinBox, QScrollArea, QMenu, QAction, QDialog,
                            QGridLayout, QFileDialog, QMessageBox, QFrame, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from src.core.uwb.uwb_devices import Anchor, Tag, Position
from src.core.uwb.channel_model import ChannelConditions, PolygonNLOSZone, PathLossParams, NLOSZone
import time
from collections import deque
from src.gui.windows.nlos_config_window import NLOSConfigWindow
from src.gui.theme import COLORS
from src.gui.widgets import PersistentWindow, PlotGroupNavigation, TimeWindowSlider, create_themed_plot, DistancePlotWidget

class DistancePlotsWindow(PersistentWindow):
    def __init__(self, parent=None):
        super().__init__(title="Distance Measurements", parent=parent)
        self.parent_app = parent
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Create header with title
        header = QLabel("Distance Measurements Dashboard")
        header.setFont(QFont('Arial', 16, QFont.Bold))
        header.setStyleSheet(f"color: {COLORS['text']}; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Create time window control
        self.time_control = TimeWindowSlider(
            min_val=1, max_val=30, default=10, suffix="s",
            label_text="Time Window:"
        )
        self.time_control.value_changed.connect(self._on_time_changed)
        layout.addWidget(self.time_control)
        
        # Create navigation controls
        self.nav = PlotGroupNavigation(total_groups=1)
        self.nav.group_changed.connect(self._on_group_changed)
        layout.addWidget(self.nav)
        
        # Replace container widget setup with a stacked layout
        self.plot_stacks = []
        self.current_container = None
        
        # Create main container
        self.main_container = QWidget()
        layout.addWidget(self.main_container)
        
        # Initialize data storage
        self.plots = {}
        
        self.time_window = 10.0
        
        self.time_window = 10.0
        
        # Initialize group tracking
        self.plots_per_group = 4
        self.current_group = 0
        self.total_groups = 0
        
        # Configure plot update timer
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.smooth_update_plots)
        self.plot_timer.start(33)  # ~30 FPS
        
        # Add dictionary to store average errors
        self.average_errors = {}



    def reset_plots(self):
        """Reset all plot data"""
        self.average_errors = {}
        
        # Reset widgets
        for plot_widget in self.plots.values():
            if isinstance(plot_widget, DistancePlotWidget):
                plot_widget.reset()

    def _on_time_changed(self, value):
        self.time_window = float(value)
        for plot_widget in self.plots.values():
            if isinstance(plot_widget, DistancePlotWidget):
                plot_widget.set_time_window(self.time_window)

    def smooth_update_plots(self):
        """Update plots smoothly"""
        # NOTE: Smooth scrolling logic is currently handled directly by widget updates or simplified.
        # If strict smooth scrolling is needed, we would iterate over visible widgets here.
        pass

    def _on_group_changed(self, group_index):
        self.current_group = group_index
        self.update_visible_group()

    def update_visible_group(self):
        # Sync the PlotGroupNavigation widget
        self.nav.total_groups = max(1, self.total_groups)
        self.nav.current_group = self.current_group
        self.nav._update_label()
        
        # Add transition effect
        for container, _ in self.plot_stacks:
            container.setVisible(False)
        
        # Show current container
        if self.current_group < len(self.plot_stacks):
            self.plot_stacks[self.current_group][0].setVisible(True)

    def create_group_container(self):
        """Create a new container for a group of plots"""
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)
        container.setVisible(False)
        self.main_container.layout().addWidget(container)
        return container, grid

    def update_anchors(self, anchors):
        # Remove plots for anchors that no longer exist
        current_ids = {anchor.id for anchor in anchors}
        removed_ids = set(self.plots.keys()) - current_ids
        for anchor_id in removed_ids:
            if anchor_id in self.plots:
                self.plots[anchor_id].close()
                del self.plots[anchor_id]
        
        # Clear existing plot stacks
        for container, _ in self.plot_stacks:
            container.deleteLater()
        self.plot_stacks.clear()
        
        # Create new layout for main container
        if self.main_container.layout() is None:
            QVBoxLayout(self.main_container)
        
        # Add plots for new anchors
        current_container = None
        current_grid = None
        row = 0
        col = 0
        
        for i, anchor in enumerate(anchors):
            # Create new container for each group
            if i % self.plots_per_group == 0:
                current_container, current_grid = self.create_group_container()
                self.plot_stacks.append((current_container, current_grid))
                row = 0
                col = 0
            
            if anchor.id not in self.plots:
                # Create DistancePlotWidget
                plot_widget = DistancePlotWidget(anchor.id, self.time_window)
                
                # Add plot widget to grid in 2x2 layout
                current_grid.addWidget(plot_widget, row, col)
                
                # Update grid position for 2x2 layout
                col += 1
                if col >= 2:  # 2 columns
                    col = 0
                    row += 1
                
                # Store references
                self.plots[anchor.id] = plot_widget
        
        # Update total groups and visibility
        self.total_groups = len(self.plot_stacks)
        self.current_group = min(self.current_group, max(0, self.total_groups - 1))
        self.update_visible_group()

    def update_distances(self, anchor_id, time, measured_distance, true_distance, is_los):
        if anchor_id in self.plots:
            plot_widget = self.plots[anchor_id]
            if isinstance(plot_widget, DistancePlotWidget):
                plot_widget.update_data(time, measured_distance, true_distance, is_los)



    def closeEvent(self, event):
        """Stop timer, then hide via PersistentWindow"""
        self.plot_timer.stop()
        super().closeEvent(event)