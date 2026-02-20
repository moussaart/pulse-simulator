"""
Plot Manager Module
Handles all plot creation and visualization updates with parallel computing optimization
"""
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import Qt
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from src.gui.theme import COLORS
from src.gui.panels.Plots import LocalizationErrorPlot


class PlotManager:
    """Manages all plotting operations for the localization app with parallel LOS checks"""
    
    # Class-level thread pool for parallel operations
    _executor = None
    
    @classmethod
    def get_executor(cls):
        """Get or create the thread pool executor"""
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(max_workers=4)
        return cls._executor
    
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
        self.use_parallel = True  # Enable/disable parallel processing
        
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
    
    def _check_single_los(self, anchor, tag_position, channel_conditions):
        """Check LOS condition for a single anchor (used for parallel execution)"""
        try:
            is_los = channel_conditions.check_los_to_anchor(anchor.position, tag_position)
            return (anchor, is_los)
        except Exception as e:
            print(f"Error checking LOS for anchor {anchor.id}: {e}")
            return (anchor, True)  # Default to LOS on error
    
    def _get_los_conditions_parallel(self, anchors, tag_position, channel_conditions):
        """Get LOS conditions for all anchors in parallel"""
        executor = self.get_executor()
        futures = []
        for anchor in anchors:
            future = executor.submit(
                self._check_single_los, 
                anchor, 
                tag_position, 
                channel_conditions
            )
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=0.5)  # 500ms timeout
                results.append(result)
            except Exception as e:
                print(f"Parallel LOS check error: {e}")
                results.append((anchors[len(results)], True))  # Default to LOS
        
        return results
    
    def update_anchor_visualization(self, position_plot, anchors, channel_conditions, tag):
        """Update anchor visualization with proper LOS/NLOS coloring using parallel LOS checks"""
        # Clear existing anchor points and labels
        for point in self.anchor_points:
            position_plot.removeItem(point)
        self.anchor_points.clear()
        
        # Remove existing text items
        for item in position_plot.items():
            if isinstance(item, pg.TextItem):
                position_plot.removeItem(item)
        
        # Recreate coordinate labels
        self.add_coordinate_labels(position_plot)
        
        # Get LOS conditions - use parallel processing for 3+ anchors
        if self.use_parallel and len(anchors) >= 3:
            los_results = self._get_los_conditions_parallel(anchors, tag.position, channel_conditions)
        else:
            los_results = [(anchor, channel_conditions.check_los_to_anchor(anchor.position, tag.position)) 
                          for anchor in anchors]
        
        # Create new anchor points with labels
        for anchor, is_los in los_results:
            # Set color based on LOS condition
            color = pg.mkColor('#00ff00') if is_los else pg.mkColor('#ff0000')
            
            # Plot anchor point
            point = pg.ScatterPlotItem(
                [anchor.position.x], [anchor.position.y],
                symbol='s',
                size=20,
                pen=pg.mkPen(color, width=2),
                brush=color,
                name=f"Anchor {anchor.id}"
            )
            position_plot.addItem(point)
            self.anchor_points.append(point)
            
            # Add anchor ID label with position
            text = pg.TextItem(
                f"{anchor.id}\n({anchor.position.x:.1f}, {anchor.position.y:.1f})",
                color='w',
                anchor=(0.5, 0)
            )
            text.setPos(anchor.position.x, anchor.position.y + 0.5)
            position_plot.addItem(text)
    
    def update_measurement_lines(self, position_plot, anchors, tag, channel_conditions):
        """Draw measurement lines between anchors and tag with parallel LOS checks"""
        # Clear old lines
        for item in position_plot.items():
            if isinstance(item, pg.PlotDataItem) and hasattr(item, 'measurement_line'):
                position_plot.removeItem(item)
        
        # Draw new lines only if visible
        if not self.lines_visible:
            return
        
        # Get LOS conditions - use parallel processing for 3+ anchors
        if self.use_parallel and len(anchors) >= 3:
            los_results = self._get_los_conditions_parallel(anchors, tag.position, channel_conditions)
        else:
            los_results = [(anchor, channel_conditions.check_los_to_anchor(anchor.position, tag.position)) 
                          for anchor in anchors]
        
        for anchor, is_los in los_results:
            try:
                # Set line color based on LOS condition
                if is_los:
                    pen = pg.mkPen('w', width=1, style=Qt.DashLine)
                else:
                    pen = pg.mkPen('r', width=1, style=Qt.DashLine)
                
                line = pg.PlotDataItem(
                    [anchor.position.x, tag.position.x],
                    [anchor.position.y, tag.position.y],
                    pen=pen
                )
                line.measurement_line = True
                position_plot.addItem(line)
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

