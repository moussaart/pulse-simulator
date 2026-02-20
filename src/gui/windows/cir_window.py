import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QCheckBox, QGridLayout, QPushButton)
from PyQt5.QtCore import Qt
from src.gui.widgets import PersistentWindow, PlotGroupNavigation, CIRPlotWidget
from src.core.uwb.uwb_types import RangingResult

class CIRWindow(PersistentWindow):
    """
    Window to visualize the Channel Impulse Response (CIR) for all anchors
    in groups of 4 (2x2 grid).
    """
    
    def __init__(self, parent=None):
        super().__init__(title="Channel Impulse Response (CIR)", parent=parent)
        self.resize(1000, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # --- Control Panel ---
        control_layout = QHBoxLayout()
        
        self.auto_scale_chk = QCheckBox("Auto Scale")
        self.auto_scale_chk.setChecked(True)
        control_layout.addWidget(self.auto_scale_chk)
        
        self.show_threshold_chk = QCheckBox("Show Detection Threshold")
        self.show_threshold_chk.setChecked(True)
        control_layout.addWidget(self.show_threshold_chk)
        
        self.pause_btn = QCheckBox("⏸ Pause All Plots")
        self.pause_btn.setStyleSheet("""
            QCheckBox { spacing: 5px; font-weight: bold; }
            QCheckBox::indicator { width: 18px; height: 18px; }
        """)
        control_layout.addWidget(self.pause_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # --- Navigation Controls ---
        self.nav = PlotGroupNavigation(total_groups=1)
        self.nav.group_changed.connect(self._on_group_changed)
        layout.addWidget(self.nav)
        
        # --- Plots Area ---
        self.main_container = QWidget()
        layout.addWidget(self.main_container)
        
        # Data storage
        self.plots = {}
        self.plot_stacks = []
        self.current_group = 0
        self.plots_per_group = 4
        self.total_groups = 0
        
    def create_group_container(self):
        """Create a new container for a group of 4 plots"""
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)
        container.setVisible(False)
        
        # Add to main layout (stacked efficiently via visibility)
        if self.main_container.layout() is None:
            QVBoxLayout(self.main_container)
        self.main_container.layout().addWidget(container)
        
        return container, grid

    def update_anchors(self, anchors):
        """Rebuild plot grids based on available anchors."""
        # Clear existing
        self.plots.clear()
        for container, _ in self.plot_stacks:
            container.deleteLater()
        self.plot_stacks.clear()
        
        # Re-create layout for main container if needed
        if self.main_container.layout() is None:
            QVBoxLayout(self.main_container)
            
        current_container = None
        current_grid = None
        row = 0
        col = 0
        
        for i, anchor in enumerate(anchors):
            # Create new group every 4 anchors
            if i % self.plots_per_group == 0:
                current_container, current_grid = self.create_group_container()
                self.plot_stacks.append((current_container, current_grid))
                row = 0
                col = 0
            
            # --- Create Plot for Anchor ---
            plot_widget = CIRPlotWidget(anchor.id)
            
            # Add to grid
            current_grid.addWidget(plot_widget, row, col)
            
            # Update grid position (2x2)
            col += 1
            if col >= 2:
                col = 0
                row += 1
            
            # Store references
            self.plots[anchor.id] = plot_widget
            
        self.total_groups = len(self.plot_stacks)
        self.current_group = 0
        self.update_visible_group()

    def _on_group_changed(self, group_index):
        self.current_group = group_index
        self.update_visible_group()

    def update_visible_group(self):
        """Show only the plots for the current group."""
        self.nav.set_total_groups(max(1, self.total_groups))
        self.nav.current_group = self.current_group
        self.nav._update_label()
        
        for i, (container, _) in enumerate(self.plot_stacks):
            container.setVisible(i == self.current_group)

    def update_cir_data(self, anchor_id: str, ranging_result: RangingResult):
        """
        Update the CIR plot for a specific anchor.
        """
        if self.pause_btn.isChecked():
            return
            
        if anchor_id not in self.plots:
            return
            
        plot_data = self.plots[anchor_id]
            
        # Use widget method to update
        if isinstance(plot_data, CIRPlotWidget):
             plot_data.update_data(
                time_vector=ranging_result.cir_time_vector,
                amplitude=ranging_result.cir_amplitude,
                is_los=ranging_result.is_los,
                first_path_index=ranging_result.cir_first_path_index,
                snr_db=ranging_result.snr_db,
                auto_scale=self.auto_scale_chk.isChecked(),
                show_threshold=self.show_threshold_chk.isChecked()
            )

    def reset_plot(self):
        """Clear all plots."""
        for p in self.plots.values():
            if isinstance(p, CIRPlotWidget):
                p.reset()
