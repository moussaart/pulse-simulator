"""
Event Handlers Module
Handles all mouse events and user interactions
"""
import numpy as np
from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5.QtWidgets import QMenu, QAction, QApplication
from src.core.uwb.uwb_devices import Anchor, Position


class PlotEventFilter(QObject):
    """Event filter to capture mouse press/release events for drag-and-drop"""
    
    def __init__(self, event_handler):
        super().__init__()
        self.event_handler = event_handler
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.GraphicsSceneMousePress:
            if event.button() == Qt.LeftButton:
                self.event_handler.handle_mouse_press(event)
        elif event.type() == QEvent.GraphicsSceneMouseRelease:
            if event.button() == Qt.LeftButton:
                self.event_handler.handle_mouse_release(event)
        elif event.type() == QEvent.GraphicsSceneMouseMove:
            # Check if left button is still pressed during move
            # This serves as a fallback to detect when button was released
            if self.event_handler.is_dragging:
                buttons = event.buttons()
                if not (buttons & Qt.LeftButton):
                    # Mouse button was released but we missed the release event
                    self.event_handler.handle_mouse_release(event)
        return False  # Don't consume the event


class EventHandler:
    """Handles all event-related logic for the application"""
    
    def __init__(self, parent):
        self.parent = parent
        
        # Interaction states
        self.adding_anchor = False
        self.adding_anchor = False
        self.deleting_anchor = False
        self.deleting_zone = False
        self.dragging_anchor = None
        self.is_dragging = False  # Track if we're actively dragging (mouse button held)
        self.anchor_drag_threshold = 0.5
        self.selecting_target = False
        
        # Drawing states
        self.drawing_line = False
        self.drawing_polygon = False
        self.drawing_line = False
        self.drawing_polygon = False
        self.drawing_trajectory = False
        self.picking_moving_trajectory = False
        self.is_recording = False
        
        # Drawing data
        self.start_pos = None
        self.current_line = None
        self.current_polygon = None
        self.polygon_points = []
        self.trajectory_points = []
        self.line_width = 0.1
        
        # Event filter for mouse press/release
        self.event_filter = None
    
    def install_event_filter(self):
        """Install event filter on the plot scene for mouse press/release events"""
        if self.event_filter is None:
            self.event_filter = PlotEventFilter(self)
            self.parent.position_plot.scene().installEventFilter(self.event_filter)
    
    def map_to_plot_coords(self, pos):
        """Convert screen position to plot coordinates"""
        view_pos = self.parent.position_plot.plotItem.vb.mapSceneToView(pos)
        return view_pos.x(), view_pos.y()
    
    def handle_mouse_press(self, event):
        """Handle mouse press events for drag-and-drop"""
        pos = event.scenePos()
        x, y = self.map_to_plot_coords(pos)
        
        # Check for multi-selection mode first
        if hasattr(self.parent, 'selection_manager') and self.parent.selection_manager.handle_mouse_press(x, y, event):
            return
            
        # Check for anchor multi-selection mode
        if hasattr(self.parent, 'anchor_selection_manager') and self.parent.anchor_selection_manager.handle_mouse_press(x, y, event):
            return
        
        # Don't start drag if in a special mode
        if self.adding_anchor or self.deleting_anchor or self.drawing_line or \
           self.drawing_polygon or self.drawing_trajectory or self.selecting_target or \
           self.picking_moving_trajectory:
            return
        
        # Try to start anchor drag
        closest_anchor = None
        min_dist = float('inf')
        
        for anchor in self.parent.anchors:
            dist = np.sqrt((x - anchor.position.x)**2 + (y - anchor.position.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_anchor = anchor
        
        if closest_anchor and min_dist < self.anchor_drag_threshold:
            self.dragging_anchor = closest_anchor
            self.is_dragging = True
            self.parent.position_plot.setCursor(Qt.ClosedHandCursor)
            # Disable plot panning while dragging anchor
            self.parent.position_plot.plotItem.vb.setMouseEnabled(x=False, y=False)
    
    def handle_mouse_release(self, event):
        """Handle mouse release events to stop dragging"""
        pos = event.scenePos()
        x, y = self.map_to_plot_coords(pos)
        
        # Check for multi-selection mode first
        if hasattr(self.parent, 'selection_manager') and self.parent.selection_manager.handle_mouse_release(x, y, event):
            return

        # Check for anchor multi-selection mode
        if hasattr(self.parent, 'anchor_selection_manager') and self.parent.anchor_selection_manager.handle_mouse_release(x, y, event):
            return

        if self.is_dragging or self.dragging_anchor:
            self.finalize_anchor_drag()
    
    def finalize_anchor_drag(self):
        """Finalize anchor dragging and reset drag state"""
        if self.dragging_anchor is None and not self.is_dragging:
            return  # Nothing to finalize
        
        # Reset dragging state
        self.dragging_anchor = None
        self.is_dragging = False
        self.parent.position_plot.setCursor(Qt.ArrowCursor)
        
        # Re-enable plot panning after dragging is complete
        self.parent.position_plot.plotItem.vb.setMouseEnabled(x=True, y=True)
        
        # Update distance plots if window is open
        if self.parent.distance_plots_window is not None:
            self.parent.distance_plots_window.update_anchors(self.parent.anchors)
    
    def handle_plot_click(self, event):
        """Handle mouse click events on the plot"""
        pos = event.scenePos()
        x, y = self.map_to_plot_coords(pos)
        
        # Handle target point selection
        if self.selecting_target and event.button() == Qt.LeftButton:
            self.handle_target_point_selection(x, y)
            return
        
        # Skip if we just finished dragging (click is the release)
        if self.dragging_anchor is not None:
            return
        
        # Handle right-click to end drawing
        if event.button() == Qt.RightButton:
            if self.handle_right_click_drawing(x, y):
                return
        
        if event.button() != Qt.LeftButton:
            return
            
        # Handle zone deletion
        if self.deleting_zone:
            self.handle_zone_deletion(x, y)
            return
        
        # Check if clicked on existing NLOS zone (only if not in special modes)
        if not self.adding_anchor and not self.deleting_anchor:
            if self.check_nlos_zone_click(x, y, event):
                return
        
        # Handle polygon drawing
        if self.drawing_polygon:
            self.handle_polygon_point_add(x, y)
            return
        
        # Handle line drawing
        if self.drawing_line:
            self.handle_line_drawing(x, y)
            return
        
        # Handle anchor deletion
        if self.deleting_anchor:
            self.handle_anchor_deletion(x, y)
            return
        
        # Handle anchor addition
        if self.adding_anchor:
            self.handle_anchor_addition(x, y)
            return
        
        # Handle trajectory drawing
        if self.drawing_trajectory:
            self.handle_trajectory_drawing(x, y, event)
            return

        # Handle moving zone trajectory picking
        if self.picking_moving_trajectory:
            self.handle_moving_zone_picking(x, y)
            return
    
    def handle_mouse_move(self, pos):
        """Handle mouse movement events"""
        x, y = self.map_to_plot_coords(pos)
        
        # Check for multi-selection mode first
        if hasattr(self.parent, 'selection_manager') and self.parent.selection_manager.handle_mouse_move(x, y):
            return

        # Check for anchor multi-selection mode
        if hasattr(self.parent, 'anchor_selection_manager') and self.parent.anchor_selection_manager.handle_mouse_move(x, y):
            return
        
        # Handle anchor dragging (only if actively dragging with mouse button held)
        if self.is_dragging and self.dragging_anchor:
            # Fallback check: verify mouse button is still pressed
            # This handles cases where the release event was missed
            mouse_buttons = QApplication.mouseButtons()
            if not (mouse_buttons & Qt.LeftButton):
                # Mouse button was released - finalize the drag
                self.finalize_anchor_drag()
                return
            
            self.update_anchor_position(x, y)
            return
        
        # Handle polygon preview
        if self.drawing_polygon and len(self.polygon_points) > 0:
            self.update_polygon_preview(x, y)
            return
        
        # Handle line preview
        if self.drawing_line and self.start_pos is not None:
            self.update_line_preview(x, y)
            return
        
        # Handle trajectory recording
        if self.drawing_trajectory and self.is_recording:
            self.add_trajectory_point(x, y)
            return

        # Handle moving zone preview
        if self.picking_moving_trajectory and self.start_pos is not None:
            self.update_line_preview(x, y)
            return
    
    def handle_target_point_selection(self, x, y):
        """Handle target point selection for Fixed Point pattern"""
        self.parent.point = (x, y)
        self.parent.plot_items['target_point_marker'].setData([x], [y])
        self.parent.target_point_btn.setChecked(False)
        self.toggle_target_point_mode()
        
        if self.parent.pattern_combo.currentText() == "Fixed Point":
            self.parent.trajectory_manager.update_trajectory_plan()
    
    def handle_right_click_drawing(self, x, y):
        """Handle right-click to finish drawing operations"""
        if self.drawing_line and self.start_pos is not None:
            self.parent.nlos_manager.create_line_zone(self.start_pos, (x, y))
            self.reset_line_drawing()
            return True
        
        elif self.drawing_polygon and len(self.polygon_points) >= 3:
            self.parent.nlos_manager.create_polygon_from_lines(self.polygon_points)
            self.reset_polygon_drawing()
            return True
        
        elif self.drawing_line or self.drawing_polygon:
            self.cancel_drawing()
            return True
        
        return False
    
    def check_nlos_zone_click(self, x, y, event):
        """Check if click is on an existing NLOS zone"""
        from src.core.uwb.channel_model import NLOSZone, PolygonNLOSZone
        
        point = Position(x, y)
        
        for zone in self.parent.channel_conditions.nlos_zones:
            is_in_zone = False
            if isinstance(zone, NLOSZone):
                is_in_zone = (zone.x1 <= x <= zone.x2 and zone.y1 <= y <= zone.y2)
            else:  # PolygonNLOSZone
                is_in_zone = zone.contains_point(point)
            
            if is_in_zone:
                self.show_zone_context_menu(zone, event)
                return True
        
        return False
    
    def show_zone_context_menu(self, zone, event):
        """Show context menu for NLOS zone"""
        menu = QMenu(self.parent)
        edit_action = QAction("Edit Zone", self.parent)
        edit_action.triggered.connect(lambda: self.parent.nlos_manager.edit_nlos_zone(zone))
        menu.addAction(edit_action)
        
        # Add preset configurations
        if self.parent.nlos_config_manager.get_config_names():
            preset_menu = menu.addMenu("Apply Preset")
            for config_name in self.parent.nlos_config_manager.get_config_names():
                action = QAction(config_name, self.parent)
                action.triggered.connect(
                    lambda checked, z=zone, name=config_name: 
                    self.parent.nlos_manager.apply_preset_to_zone(z, name))
                preset_menu.addAction(action)
        
        menu.exec_(event.screenPos().toPoint())
    
    def handle_polygon_point_add(self, x, y):
        """Add point to polygon drawing"""
        self.polygon_points.append((x, y))
        
        if len(self.polygon_points) > 1 and self.current_polygon is not None:
            preview_points = self.polygon_points + [self.polygon_points[0]]
            x_values, y_values = zip(*preview_points)
            self.current_polygon.setData(x_values, y_values)
    
    def handle_line_drawing(self, x, y):
        """Handle line drawing clicks"""
        if self.start_pos is None:
            self.start_pos = (x, y)
        else:
            self.parent.nlos_manager.create_line_zone(self.start_pos, (x, y))
            self.start_pos = None
            if self.current_line:
                self.parent.position_plot.removeItem(self.current_line)
            import pyqtgraph as pg
            self.current_line = pg.PlotDataItem(pen='r')
            self.parent.position_plot.addItem(self.current_line)
    
    def handle_anchor_deletion(self, x, y):
        """Handle anchor deletion - stays in delete mode after deletion"""
        anchor = self.find_nearest_anchor(x, y)
        if anchor:
            if len(self.parent.anchors) > 3:
                # Delete the anchor (only once)
                self.parent.anchors.remove(anchor)
                self.parent.renormalize_anchors()
                # Stay in delete mode - don't reset deleting_anchor flag
            else:
                # Cannot delete - minimum 3 anchors required
                self.parent.show_info_message(
                    "Cannot Delete", 
                    "Minimum 3 anchors required for localization."
                )
    
    def handle_anchor_addition(self, x, y):
        """Handle anchor addition"""
        new_anchor = Anchor(Position(x, y))
        self.parent.anchors.append(new_anchor)
        
        self.parent.renormalize_anchors()

    def handle_zone_deletion(self, x, y):
        """Handle NLOS zone deletion"""
        self.parent.nlos_manager.delete_zone_at(x, y)
    
    def handle_trajectory_drawing(self, x, y, event):
        """Handle trajectory drawing clicks"""
        if event.button() == Qt.LeftButton:
            if not self.is_recording:
                self.is_recording = True
                self.trajectory_points = [[x, y]]
                self.parent.draw_trajectory_btn.setText("Click to Stop Recording")
            else:
                self.trajectory_points.append([x, y])
                self.is_recording = False
                self.drawing_trajectory = False
                self.parent.draw_trajectory_btn.setText("Draw New Trajectory")
                self.parent.position_plot.setCursor(Qt.ArrowCursor)
                self.parent.trajectory_manager.save_trajectory_dialog()

    def handle_moving_zone_picking(self, x, y):
        """Handle picking start and end points for moving zone"""
        if self.start_pos is None:
            # First click: Set start point
            self.start_pos = (x, y)
            self.parent.nlos_widgets['add_moving_btn'].setText("Click End Point")
        else:
            # Second click: Set end point and finish
            end_pos = (x, y)
            
            # Clean up preview
            if self.current_line:
                self.parent.position_plot.removeItem(self.current_line)
                self.current_line = None
            
            # Reset state but keep start_pos for dialog call
            start_pos = self.start_pos
            self.start_pos = None
            
            # Toggle off the mode
            self.parent.nlos_widgets['add_moving_btn'].setChecked(False)
            self.parent.nlos_manager.toggle_moving_zone_placement()
            
            # Open the dialog with the picked coordinates
            self.parent.nlos_manager.add_moving_zone_dialog(initial_pos=start_pos, final_pos=end_pos)
    
    def update_anchor_position(self, x, y):
        """Update dragged anchor position"""
        self.dragging_anchor.position.x = x
        self.dragging_anchor.position.y = y
        self.parent.plot_manager.update_anchor_visualization(
            self.parent.position_plot,
            self.parent.anchors,
            self.parent.channel_conditions,
            self.parent.tag
        )
        self.parent.update_anchor_list()
    
    def update_polygon_preview(self, x, y):
        """Update polygon preview while drawing"""
        if self.current_polygon is None:
            return
            
        preview_points = self.polygon_points + [(x, y)]
        if len(self.polygon_points) > 0:
            preview_points.append(self.polygon_points[0])
        
        if len(preview_points) > 1:
            x_values, y_values = zip(*preview_points)
            self.current_polygon.setData(x_values, y_values)
    
    def update_line_preview(self, x, y):
        """Update line preview while drawing"""
        if self.current_line is None:
            import pyqtgraph as pg
            self.current_line = pg.PlotDataItem(pen=pg.mkPen('r', style=Qt.DashLine))
            self.parent.position_plot.addItem(self.current_line)
            
        if self.start_pos is None:
            return
            
        x_values = [self.start_pos[0], x]
        y_values = [self.start_pos[1], y]
        self.current_line.setData(x_values, y_values)
    
    def add_trajectory_point(self, x, y):
        """Add point to trajectory recording"""
        self.trajectory_points.append([x, y])
        x_values = [p[0] for p in self.trajectory_points]
        y_values = [p[1] for p in self.trajectory_points]
        self.parent.trajectory_preview.setData(x_values, y_values)
    
    def find_nearest_anchor(self, x, y, max_distance=1.0):
        """Find nearest anchor to a point"""
        nearest = None
        min_dist = max_distance
        
        for anchor in self.parent.anchors:
            dist = np.sqrt((anchor.position.x - x)**2 + (anchor.position.y - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = anchor
        
        return nearest
    
    def reset_line_drawing(self):
        """Reset line drawing state"""
        self.drawing_line = False
        self.parent.draw_mode_btn.setChecked(False)
        self.parent.draw_mode_btn.setText("🖊️ Draw Line")
        if self.current_line:
            self.parent.position_plot.removeItem(self.current_line)
        self.current_line = None
        self.start_pos = None
        self.parent.position_plot.setCursor(Qt.ArrowCursor)
    
    def reset_polygon_drawing(self):
        """Reset polygon drawing state"""
        self.drawing_polygon = False
        self.parent.draw_polygon_btn.setChecked(False)
        self.parent.draw_polygon_btn.setText("⬡ Draw Polygon")
        if self.current_polygon:
            self.parent.position_plot.removeItem(self.current_polygon)
        self.current_polygon = None
        self.polygon_points = []
        self.parent.position_plot.setCursor(Qt.ArrowCursor)
    
    def cancel_drawing(self):
        """Cancel current drawing operation"""
        if self.drawing_line:
            self.reset_line_drawing()
        elif self.drawing_polygon:
            self.reset_polygon_drawing()
    
    def toggle_target_point_mode(self):
        """Toggle target point selection mode"""
        self.selecting_target = self.parent.target_point_btn.isChecked()
        if self.selecting_target:
            self.parent.target_point_btn.setText("🎯 Click on Map")
            self.parent.position_plot.setCursor(Qt.CrossCursor)
            self.disable_other_modes('selecting_target')
        else:
            self.parent.target_point_btn.setText("🎯 Set Target Point")
            self.parent.position_plot.setCursor(Qt.ArrowCursor)
    
    def disable_other_modes(self, except_mode=None):
        """Disable all other interactive modes except the specified one.
        
        This ensures mutual exclusivity between:
        - Create anchor mode (adding_anchor)
        - Delete anchor mode (deleting_anchor)
        - Move anchor mode (drag & drop - no explicit mode, just state)
        - Drawing modes (line, polygon)
        - Target selection mode
        - Trajectory drawing mode
        
        Args:
            except_mode: The mode to keep active (all others will be disabled)
        """
        # Track if we need to reset cursor (will be set by the caller for the active mode)
        should_reset_cursor = True
        
        def safe_set_checked(btn, checked):
            try:
                if btn is not None:
                    btn.setChecked(checked)
            except RuntimeError:
                pass

        def safe_set_text(btn, text):
            try:
                if btn is not None:
                    btn.setText(text)
            except RuntimeError:
                pass
                
        def safe_set_enabled(btn, enabled):
            try:
                if btn is not None:
                    btn.setEnabled(enabled)
            except RuntimeError:
                pass

        # Disable add anchor mode
        if except_mode != 'adding_anchor':
            was_active = self.adding_anchor
            self.adding_anchor = False
            if hasattr(self.parent, 'add_anchor_btn'):
                safe_set_checked(self.parent.add_anchor_btn, False)
                safe_set_text(self.parent.add_anchor_btn, "Click to Add")
                # Re-enable delete button if add mode was active
                if was_active and hasattr(self.parent, 'delete_anchor_btn'):
                    safe_set_enabled(self.parent.delete_anchor_btn, True)
        else:
            should_reset_cursor = False
        
        # Disable delete anchor mode
        if except_mode != 'deleting_anchor':
            was_active = self.deleting_anchor
            self.deleting_anchor = False
            if hasattr(self.parent, 'delete_anchor_btn'):
                safe_set_checked(self.parent.delete_anchor_btn, False)
                safe_set_text(self.parent.delete_anchor_btn, "Delete")
                # Re-enable add button if delete mode was active
                if was_active and hasattr(self.parent, 'add_anchor_btn'):
                     safe_set_enabled(self.parent.add_anchor_btn, True)
        else:
            should_reset_cursor = False
        
        # Disable line drawing mode
        if except_mode != 'drawing_line':
            was_active = self.drawing_line
            if self.drawing_line:
                # Clean up line preview
                if self.current_line:
                    self.parent.position_plot.removeItem(self.current_line)
                    self.current_line = None
                self.start_pos = None
            self.drawing_line = False
            if hasattr(self.parent, 'draw_mode_btn'):
                safe_set_checked(self.parent.draw_mode_btn, False)
                safe_set_text(self.parent.draw_mode_btn, "🖊️ Draw Line")
            # Re-enable anchor buttons if line mode was active
            if was_active:
                if hasattr(self.parent, 'add_anchor_btn'):
                    safe_set_enabled(self.parent.add_anchor_btn, True)
                if hasattr(self.parent, 'delete_anchor_btn'):
                    safe_set_enabled(self.parent.delete_anchor_btn, True)
        else:
            should_reset_cursor = False

        # Disable polygon drawing mode
        if except_mode != 'drawing_polygon':
            was_active = self.drawing_polygon
            if self.drawing_polygon:
                # Clean up polygon preview
                if self.current_polygon:
                    self.parent.position_plot.removeItem(self.current_polygon)
                    self.current_polygon = None
                self.polygon_points = []
            self.drawing_polygon = False
            if hasattr(self.parent, 'draw_polygon_btn'):
                safe_set_checked(self.parent.draw_polygon_btn, False)
                safe_set_text(self.parent.draw_polygon_btn, "⬡ Draw Polygon")
            # Re-enable anchor buttons if polygon mode was active
            if was_active:
                if hasattr(self.parent, 'add_anchor_btn'):
                    safe_set_enabled(self.parent.add_anchor_btn, True)
                if hasattr(self.parent, 'delete_anchor_btn'):
                    safe_set_enabled(self.parent.delete_anchor_btn, True)
        else:
            should_reset_cursor = False
            
        # Disable zone deletion mode
        if except_mode != 'deleting_zone':
            self.deleting_zone = False
            if hasattr(self.parent, 'nlos_widgets') and 'delete_mode_btn' in self.parent.nlos_widgets:
                safe_set_checked(self.parent.nlos_widgets['delete_mode_btn'], False)
                safe_set_text(self.parent.nlos_widgets['delete_mode_btn'], "🗑️ Delete")
        else:
            should_reset_cursor = False
        
        # Disable target selection mode
        if except_mode != 'selecting_target':
            self.selecting_target = False
            if hasattr(self.parent, 'target_point_btn'):
                safe_set_checked(self.parent.target_point_btn, False)
                safe_set_text(self.parent.target_point_btn, "🎯 Set Target Point")
        else:
            should_reset_cursor = False
        
        # Disable trajectory drawing mode
        if except_mode != 'drawing_trajectory':
            if self.drawing_trajectory:
                self.is_recording = False
                self.trajectory_points = []
            self.drawing_trajectory = False
            if hasattr(self.parent, 'draw_trajectory_btn'):
                safe_set_checked(self.parent.draw_trajectory_btn, False)
                safe_set_text(self.parent.draw_trajectory_btn, "Draw New Trajectory")
        else:
            should_reset_cursor = False
            
        # Disable moving zone picking mode
        if except_mode != 'picking_moving_trajectory':
            if self.picking_moving_trajectory:
                # Clean up preview
                if self.current_line:
                    self.parent.position_plot.removeItem(self.current_line)
                    self.current_line = None
                self.start_pos = None
                
            self.picking_moving_trajectory = False
            if hasattr(self.parent, 'nlos_widgets') and self.parent.nlos_widgets and 'add_moving_btn' in self.parent.nlos_widgets:
                try:
                    btn = self.parent.nlos_widgets['add_moving_btn']
                    if btn is not None:
                        btn.setChecked(False)
                        btn.setText("🏃 Moving Obstacle")
                except RuntimeError:
                    pass
        else:
            should_reset_cursor = False
        
        # Cancel any ongoing anchor drag (move mode)
        if self.is_dragging or self.dragging_anchor:
            self.finalize_anchor_drag()
        
        # Reset cursor to arrow if no mode is being activated
        if should_reset_cursor and except_mode is None:
            self.parent.position_plot.setCursor(Qt.ArrowCursor)
    
    def get_active_mode(self):
        """Get the currently active interaction mode, if any.
        
        Returns:
            str or None: Name of the active mode, or None if no mode is active
        """
        if self.adding_anchor:
            return 'adding_anchor'
        if self.deleting_anchor:
            return 'deleting_anchor'
        if self.deleting_zone:
            return 'deleting_zone'
        if self.drawing_line:
            return 'drawing_line'
        if self.drawing_polygon:
            return 'drawing_polygon'
        if self.selecting_target:
            return 'selecting_target'
        if self.drawing_trajectory:
            return 'drawing_trajectory'
        if self.picking_moving_trajectory:
            return 'picking_moving_trajectory'
        if self.is_dragging:
            return 'dragging'
        return None
    
    def is_any_mode_active(self):
        """Check if any interactive mode is currently active.
        
        Returns:
            bool: True if any mode is active, False otherwise
        """
        return self.get_active_mode() is not None
    
    def reset_all_modes(self):
        """Reset all interactive modes and restore default state.
        
        This method disables all modes and re-enables all buttons.
        Useful when cleaning the map or other operations that should
        reset the entire UI state.
        """
        # Disable all modes (pass None to disable everything)
        self.disable_other_modes(None)
        
        def safe_reset(btn, text):
            try:
                if btn is not None:
                    btn.setEnabled(True)
                    btn.setChecked(False)
                    btn.setText(text)
            except RuntimeError:
                pass
        
        # Re-enable all buttons
        if hasattr(self.parent, 'add_anchor_btn'):
            safe_reset(self.parent.add_anchor_btn, "Click to Add")
        
        if hasattr(self.parent, 'delete_anchor_btn'):
            safe_reset(self.parent.delete_anchor_btn, "Delete")
        
        if hasattr(self.parent, 'draw_mode_btn'):
            safe_reset(self.parent.draw_mode_btn, "🖊️ Draw Line")
        
        if hasattr(self.parent, 'draw_polygon_btn'):
            safe_reset(self.parent.draw_polygon_btn, "⬡ Draw Polygon")
        
        if hasattr(self.parent, 'target_point_btn'):
            safe_reset(self.parent.target_point_btn, "🎯 Set Target Point")
        
        if hasattr(self.parent, 'draw_trajectory_btn'):
            safe_reset(self.parent.draw_trajectory_btn, "Draw New Trajectory")
        
        # Reset cursor
        self.parent.position_plot.setCursor(Qt.ArrowCursor)
        
        # Re-enable plot panning (in case it was disabled during anchor drag)
        self.parent.position_plot.plotItem.vb.setMouseEnabled(x=True, y=True)

