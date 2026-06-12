"""
Plot Manager Module
Handles all plot creation and visualization updates.
Optimized for performance: reuses Qt objects instead of recreating each frame,
and accepts pre-computed LOS results from the simulation manager's frame cache.
"""
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import Qt
from collections import deque
from src.gui.theme import COLORS
from src.gui.panels.Plots import LocalizationErrorPlot


class PlotManager:
    """Manages all plotting operations for the localization app.
    
    Performance optimizations:
    - Anchor ScatterPlotItem / TextItem objects are reused across frames (only colors update)
    - Measurement line PlotDataItem objects are reused (only data/pen update)
    - LOS conditions are received from the simulation manager's per-frame cache
    """
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.anchor_points = []
        
        # Initialize trajectory history storage using deque
        self.tag_history_x = deque(maxlen=100)
        self.tag_history_y = deque(maxlen=100)
        self.estimated_history_x = deque(maxlen=100)
        self.estimated_history_y = deque(maxlen=100)
        
        # Path visibility state
        self.paths_visible = True
        self.lines_visible = True
        
        # --- Object reuse caches for anchor visualization ---
        # Maps anchor_id -> {'point': ScatterPlotItem, 'text': TextItem}
        self._anchor_visuals = {}
        # Set of anchor IDs from last frame (to detect add/remove)
        self._last_anchor_ids = set()
        
        # --- Object reuse cache for measurement lines ---
        # Maps anchor_id -> PlotDataItem
        self._measurement_lines = {}
        
    def create_position_plot(self):
        """Create and configure the main position plot"""
        position_plot = pg.PlotWidget()
        position_plot.setBackground(COLORS['background'])
        position_plot.setTitle("Real-Time Localization View", color=COLORS['text'], size='12pt')
        
        # Add axis labels with units
        position_plot.setLabel('left', 'Y Position (m)', color=COLORS['text'], size='10pt')
        position_plot.setLabel('bottom', 'X Position (m)', color=COLORS['text'], size='10pt')
        
        # Improve grid appearance
        position_plot.showGrid(x=True, y=True, alpha=0.3)
        position_plot.getAxis('left').setGrid(100)
        position_plot.getAxis('bottom').setGrid(100)
        
        # Add axis settings
        position_plot.setAspectLocked(True)
        position_plot.setRange(xRange=(-10, 10), yRange=(-10, 10))
        
        # Add legend with better positioning and style
        position_plot.addLegend(offset=(-10, 10))
        
        # Enable mouse interaction
        position_plot.setMouseEnabled(x=True, y=True)
        position_plot.getViewBox().setAspectLocked(True)
        position_plot.getViewBox().enableAutoRange(enable=False)
        
        return position_plot
    
    def create_plot_items(self, position_plot):
        """Create all plot items (trajectories, points, etc.)"""
        plot_items = {}
        
        # Add trajectory plan plot FIRST
        plot_items['trajectory_plan'] = position_plot.plot(
            [], [], 
            pen=pg.mkPen('#007FFF', width=2, style=Qt.DashLine),
            name="Planned Path"
        )
        
        # Add trajectory history plots
        plot_items['tag_trajectory'] = position_plot.plot(
            [], [], 
            pen=pg.mkPen('#ff5500', width=2, alpha=150),
            name="Tag Path History"
        )
        
        plot_items['estimated_trajectory'] = position_plot.plot(
            [], [], 
            pen=pg.mkPen('w', width=2, style=Qt.DashLine, alpha=150),
            name="Estimated Path History"
        )
        
        # Tag current position
        plot_items['tag_point'] = position_plot.plot(
            [0], [0], 
            pen=None, 
            symbol='o',
            symbolBrush='#ff5500', 
            symbolSize=15,
            name="Current Tag Position"
        )
        
        # Estimated position
        plot_items['estimated_point'] = position_plot.plot(
            [0], [0], 
            pen=None, 
            symbol='o',
            symbolBrush='#ffffff', 
            symbolSize=15,
            name="Estimated Position"
        )
        
        # Target point marker
        plot_items['target_point_marker'] = position_plot.plot(
            [], [], 
            pen=None,
            symbol='x',
            symbolBrush='#9C27B0',
            symbolSize=15,
            name="Target Point"
        )
        
        return plot_items
    
    def create_error_plot(self):
        """Create error plot handler"""
        return LocalizationErrorPlot()
    
    def add_coordinate_labels(self, position_plot):
        """Add coordinate system labels and markers"""
        origin_text = pg.TextItem("(0,0)", color=COLORS['text'], anchor=(0.5, 1.5))
        origin_text.setPos(0, 0)
        position_plot.addItem(origin_text)
    
    def update_anchor_visualization(self, position_plot, anchors, channel_conditions, tag, los_results=None):
        """Update anchor visualization with proper LOS/NLOS coloring.
        
        Performance: reuses existing ScatterPlotItem and TextItem objects.
        Only creates/destroys items when anchors are added or removed.
        
        Args:
            position_plot: The pyqtgraph PlotWidget
            anchors: List of anchor objects
            channel_conditions: Channel conditions (fallback for LOS if no cache)
            tag: Tag object
            los_results: Optional pre-computed list of (anchor, is_los) tuples from frame cache.
                        If None, LOS is computed here (backward compatibility).
        """
        current_anchor_ids = {a.id for a in anchors}
        
        # Detect if anchor set changed (additions or removals)
        if current_anchor_ids != self._last_anchor_ids:
            # Remove visuals for anchors that no longer exist
            removed_ids = self._last_anchor_ids - current_anchor_ids
            for aid in removed_ids:
                if aid in self._anchor_visuals:
                    vis = self._anchor_visuals.pop(aid)
                    position_plot.removeItem(vis['point'])
                    position_plot.removeItem(vis['text'])
            
            # Create visuals for new anchors
            added_ids = current_anchor_ids - self._last_anchor_ids
            for anchor in anchors:
                if anchor.id in added_ids:
                    color = pg.mkColor('#00ff00')
                    point = pg.ScatterPlotItem(
                        [anchor.position.x], [anchor.position.y],
                        symbol='s', size=20,
                        pen=pg.mkPen(color, width=2),
                        brush=color,
                        name=f"Anchor {anchor.id}"
                    )
                    position_plot.addItem(point)
                    
                    text = pg.TextItem(
                        f"{anchor.id}\n({anchor.position.x:.1f}, {anchor.position.y:.1f})",
                        color='w', anchor=(0.5, 0)
                    )
                    text.setPos(anchor.position.x, anchor.position.y + 0.5)
                    position_plot.addItem(text)
                    
                    self._anchor_visuals[anchor.id] = {'point': point, 'text': text}
            
            self._last_anchor_ids = current_anchor_ids
        
        # If no pre-computed LOS, compute here (backward compatibility fallback)
        if los_results is None:
            los_results = [(a, channel_conditions.check_los_to_anchor(a.position, tag.position))
                          for a in anchors]
        
        # Update colors and positions for ALL anchors (fast — no alloc/dealloc)
        for anchor, is_los in los_results:
            if anchor.id not in self._anchor_visuals:
                continue
            vis = self._anchor_visuals[anchor.id]
            
            color = pg.mkColor('#00ff00') if is_los else pg.mkColor('#ff0000')
            vis['point'].setBrush(color)
            vis['point'].setPen(pg.mkPen(color, width=2))
            vis['point'].setData([anchor.position.x], [anchor.position.y])
            
            vis['text'].setText(f"{anchor.id}\n({anchor.position.x:.1f}, {anchor.position.y:.1f})")
            vis['text'].setPos(anchor.position.x, anchor.position.y + 0.5)
    
    def update_measurement_lines(self, position_plot, anchors, tag, channel_conditions, los_results=None):
        """Draw measurement lines between anchors and tag.
        
        Performance: reuses existing PlotDataItem objects instead of destroying/recreating.
        
        Args:
            position_plot: The pyqtgraph PlotWidget
            anchors: List of anchor objects
            tag: Tag object
            channel_conditions: Channel conditions (fallback for LOS)
            los_results: Optional pre-computed list of (anchor, is_los) tuples from frame cache.
        """
        current_anchor_ids = {a.id for a in anchors}
        
        # Remove lines for anchors that no longer exist
        removed_ids = set(self._measurement_lines.keys()) - current_anchor_ids
        for aid in removed_ids:
            line = self._measurement_lines.pop(aid)
            position_plot.removeItem(line)
        
        # Hide all lines if not visible
        if not self.lines_visible:
            for line in self._measurement_lines.values():
                line.setVisible(False)
            return
        
        # If no pre-computed LOS, compute here (backward compatibility fallback)
        if los_results is None:
            los_results = [(a, channel_conditions.check_los_to_anchor(a.position, tag.position))
                          for a in anchors]
        
        for anchor, is_los in los_results:
            try:
                pen = pg.mkPen('w', width=1, style=Qt.DashLine) if is_los else pg.mkPen('r', width=1, style=Qt.DashLine)
                
                if anchor.id in self._measurement_lines:
                    # Reuse existing line — just update data and pen
                    line = self._measurement_lines[anchor.id]
                    line.setData(
                        [anchor.position.x, tag.position.x],
                        [anchor.position.y, tag.position.y]
                    )
                    line.setPen(pen)
                    line.setVisible(True)
                else:
                    # Create new line for this anchor
                    line = pg.PlotDataItem(
                        [anchor.position.x, tag.position.x],
                        [anchor.position.y, tag.position.y],
                        pen=pen
                    )
                    line.measurement_line = True
                    position_plot.addItem(line)
                    self._measurement_lines[anchor.id] = line
            except Exception as e:
                print(f"Error drawing measurement line: {e}")
    
    def update_trajectory_histories(self, tag_pos, estimated_pos, plot_items):
        """Update trajectory history plots"""
        self.tag_history_x.append(tag_pos[0])
        self.tag_history_y.append(tag_pos[1])
        self.estimated_history_x.append(estimated_pos[0])
        self.estimated_history_y.append(estimated_pos[1])
        
        # Update trajectory plots with visibility check
        if len(self.tag_history_x) > 1 and self.paths_visible:
            plot_items['tag_trajectory'].setData(
                list(self.tag_history_x), 
                list(self.tag_history_y)
            )
            plot_items['estimated_trajectory'].setData(
                list(self.estimated_history_x), 
                list(self.estimated_history_y)
            )
    
    def clear_trajectory_histories(self, plot_items):
        """Clear all trajectory history data"""
        self.tag_history_x.clear()
        self.tag_history_y.clear()
        self.estimated_history_x.clear()
        self.estimated_history_y.clear()
        plot_items['tag_trajectory'].setData([], [])
        plot_items['estimated_trajectory'].setData([], [])
    
    def toggle_path_visibility(self, plot_items):
        """Toggle visibility of path-related plots"""
        self.paths_visible = not self.paths_visible
        plot_items['tag_trajectory'].setVisible(self.paths_visible)
        plot_items['estimated_trajectory'].setVisible(self.paths_visible)
        plot_items['trajectory_plan'].setVisible(self.paths_visible)
        return self.paths_visible
    
    def toggle_measurement_lines_visibility(self):
        """Toggle visibility of measurement lines"""
        self.lines_visible = not self.lines_visible
        return self.lines_visible

