
"""
AnchorSelectionManager Module
Handles multi-selection of anchors using rubber band selection (A key) and movement (D key)
"""
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QGraphicsRectItem
from src.core.uwb.uwb_devices import Anchor, Position

class AnchorSelectionManager:
    """Manages anchor selection logic and visual feedback"""
    
    def __init__(self, parent):
        self.parent = parent
        self.is_selection_mode = False  # Triggered by 'A' key
        self.is_moving_mode = False     # Triggered by 'S' key
        self.is_drag_selecting = False
        self.is_dragging_move = False
        self.moving_anchors = []
        self.selected_anchors = []      # Persist selection
        self.highlight_items = []       # Visual highlights
        
        self.start_pos = None
        self.current_rect = None
        self.selection_rect_item = None
        
        # Style for selection rectangle (Cyan for Anchors)
        self.selection_pen = pg.mkPen(color='#00BCD4', width=1, style=Qt.DashLine)
        self.selection_brush = pg.mkBrush(color=(0, 188, 212, 50))

    def handle_key_press(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_A:
            if not event.isAutoRepeat():
                self.is_selection_mode = True
                self.update_cursor()
                # Disable plot panning/zooming while selecting
                self.parent.position_plot.plotItem.vb.setMouseEnabled(x=False, y=False)
                
                # Disable other interactions
                if hasattr(self.parent.event_handler, 'disable_other_modes'):
                     pass
            return True
            
        elif event.key() == Qt.Key_D:
            # Enable move mode if A is also held (or independent if desired)
            if not event.isAutoRepeat():
                self.is_moving_mode = True
                self.update_cursor()
                # Auto-pause simulation
                self.pause_simulation()
            return True
            
        return False
        
    def handle_key_release(self, event):
        """Handle key release events"""
        if event.key() == Qt.Key_A:
            if not event.isAutoRepeat():
                self.exit_selection_mode()
            return True
            
        elif event.key() == Qt.Key_D:
            if not event.isAutoRepeat():
                self.is_moving_mode = False
                self.update_cursor()
            return True
            
        return False

    def update_cursor(self):
        """Update cursor based on current mode"""
        if self.is_selection_mode and self.is_moving_mode:
             self.parent.position_plot.setCursor(Qt.OpenHandCursor)
        elif self.is_selection_mode:
             self.parent.position_plot.setCursor(Qt.CrossCursor)
        else:
             self.parent.position_plot.setCursor(Qt.ArrowCursor)
        
    def handle_mouse_press(self, x, y, event):
        """Handle mouse press during selection/move mode"""
        if not self.is_selection_mode:
            return False
            
        if event.button() == Qt.LeftButton:
            self.start_pos = (x, y)
            
            # Check for Move Mode (A + S)
            if self.is_moving_mode:
                # If we have a previous selection, move THAT group
                if self.selected_anchors:
                    self.moving_anchors = self.selected_anchors
                else:
                    # Otherwise find anchor under cursor
                    anchor = self.find_anchor_at_point(x, y)
                    if anchor:
                        self.moving_anchors = [anchor]
                    else:
                        self.moving_anchors = []
                
                if self.moving_anchors:
                    self.is_dragging_move = True
                    self.parent.position_plot.setCursor(Qt.ClosedHandCursor)
                    return True
            else:
                # Default Selection Mode
                self.clear_highlights()
                self.selected_anchors = []
                
                self.is_drag_selecting = True
                
                # Create visual rectangle
                self.selection_rect_item = QGraphicsRectItem()
                self.selection_rect_item.setPen(self.selection_pen)
                self.selection_rect_item.setBrush(self.selection_brush)
                self.selection_rect_item.setZValue(1000) # Ensure it's on top
                self.parent.position_plot.addItem(self.selection_rect_item)
                return True
            
        return False
        
    def handle_mouse_move(self, x, y):
        """Handle mouse move during selection/move mode"""
        if not self.is_selection_mode:
            return False
            
        # Handle Moving Anchors
        if self.is_moving_mode and self.is_dragging_move and self.start_pos:
            dx = x - self.start_pos[0]
            dy = y - self.start_pos[1]
            
            for anchor in self.moving_anchors:
                self.move_anchor(anchor, dx, dy)
            
            self.start_pos = (x, y) # Update start pos for incremental delta
            self.update_anchor_visuals() # Redraw anchors
            self.update_highlights()
            return True
            
        # Handle Selection Rectangle
        if self.is_drag_selecting and self.start_pos and self.selection_rect_item:
            x0, y0 = self.start_pos
            
            # Create rect (handle all directions)
            left = min(x0, x)
            top = min(y0, y)
            width = abs(x - x0)
            height = abs(y - y0)
            
            self.current_rect = QRectF(left, top, width, height)
            self.selection_rect_item.setRect(self.current_rect)
            return True
            
        return False
        
    def handle_mouse_release(self, x, y, event):
        """Handle mouse release to finalize selection or move"""
        if not self.is_selection_mode:
            return False
            
        if event.button() == Qt.LeftButton:
            
            if self.is_dragging_move:
                self.is_dragging_move = False
                self.moving_anchors = [] if not self.selected_anchors else self.selected_anchors
                self.update_cursor()
                return True
                
            if self.is_drag_selecting:
                self.is_drag_selecting = False
                
                # Find items in rect
                if self.current_rect:
                    self.selected_anchors = self.find_anchors_in_rect(self.current_rect)
                    self.highlight_selection()
                
                self.cleanup_selection_visuals()
                return True
            
        return False

    def highlight_selection(self):
        """Draw bounding boxes around selected anchors"""
        self.clear_highlights()
        
        for anchor in self.selected_anchors:
            # Create a small rect item for highlight around the anchor
            # Anchors are points, so we make a small box around them
            size = 0.5 # 50cm box
            rect = QRectF(anchor.position.x - size/2, anchor.position.y - size/2, size, size)
                
            item = QGraphicsRectItem(rect)
            item.setPen(pg.mkPen('#FFFF00', width=2)) # Yellow highlight
            item.setBrush(pg.mkBrush(None))
            self.parent.position_plot.addItem(item)
            self.highlight_items.append((item, anchor))
            
    def update_highlights(self):
        """Update highlight positions"""
        size = 0.5
        for item, anchor in self.highlight_items:
             rect = QRectF(anchor.position.x - size/2, anchor.position.y - size/2, size, size)
             item.setRect(rect)

    def clear_highlights(self):
        """Remove highlight visual items"""
        for item, _ in self.highlight_items:
            self.parent.position_plot.removeItem(item)
        self.highlight_items = []

    def exit_selection_mode(self):
        """Exit selection mode and restore state"""
        self.is_selection_mode = False
        self.is_moving_mode = False 
        self.is_drag_selecting = False
        self.is_dragging_move = False
        self.selected_anchors = []
        self.cleanup_selection_visuals()
        self.clear_highlights()
        self.update_cursor()
        # Re-enable plot panning/zooming
        self.parent.position_plot.plotItem.vb.setMouseEnabled(x=True, y=True)

    def cleanup_selection_visuals(self):
        """Remove selection rectangle"""
        if self.selection_rect_item:
            self.parent.position_plot.removeItem(self.selection_rect_item)
            self.selection_rect_item = None
        self.current_rect = None
        self.start_pos = None

    def pause_simulation(self):
        """Pause simulation if running"""
        # Check if parent has pause button and it's not checked (Running)
        if hasattr(self.parent, 'pause_button') and self.parent.pause_button:
            if not self.parent.pause_button.isChecked():
                self.parent.pause_button.setChecked(True)
                # Manually trigger the toggle_pause logic since setChecked doesn't emit clicked
                if hasattr(self.parent, 'toggle_pause'):
                    self.parent.toggle_pause()

    def find_anchors_in_rect(self, rect):
        """Find anchors intersecting with the selection rectangle"""
        selected = []
        for anchor in self.parent.anchors:
            if rect.contains(anchor.position.x, anchor.position.y):
                selected.append(anchor)
        return selected
        
    def find_anchor_at_point(self, x, y):
        """Find the anchor closest to point (x, y)"""
        threshold = 0.5 # Hit radius
        closest = None
        min_dist = float('inf')
        
        for anchor in self.parent.anchors:
            dist = ((anchor.position.x - x)**2 + (anchor.position.y - y)**2)**0.5
            if dist < threshold and dist < min_dist:
                min_dist = dist
                closest = anchor
        return closest
        
    def move_anchor(self, anchor, dx, dy):
        """Translate an anchor by delta (dx, dy)"""
        anchor.position.x += dx
        anchor.position.y += dy
        
    def update_anchor_visuals(self):
        """Trigger redraw of anchors via PlotManager"""
        # We need to tell PlotManager to update. 
        # Localization_app usually calls this. We can access it via parent.
        if hasattr(self.parent, 'plot_manager'):
            self.parent.plot_manager.update_anchor_visualization(
                self.parent.position_plot, 
                self.parent.anchors, 
                self.parent.channel_conditions, 
                self.parent.tag
            )
            # Also update measurement lines if they exist
            if hasattr(self.parent.plot_manager, 'update_measurement_lines'):
                 self.parent.plot_manager.update_measurement_lines(
                     self.parent.position_plot,
                     self.parent.anchors,
                     self.parent.tag,
                     self.parent.channel_conditions
                 )
