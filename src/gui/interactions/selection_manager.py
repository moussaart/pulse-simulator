"""
SelectionManager Module
Handles multi-selection of obstacles using rubber band selection
"""
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QGraphicsRectItem
from src.core.uwb.Nlos_zones import NLOSZone, PolygonNLOSZone, MovingNLOSZone
from src.core.uwb.uwb_devices import Position

class SelectionManager:
    """Manages selection logic and visual feedback"""
    
    def __init__(self, parent):
        self.parent = parent
        self.is_selection_mode = False  # Triggered by 'O' key
        self.is_moving_mode = False     # Triggered by 'P' key
        self.is_drag_selecting = False
        self.is_dragging_move = False
        self.moving_zones = []
        self.selected_zones = []        # Persist selection
        self.highlight_items = []       # Visual highlights
        
        self.start_pos = None
        self.current_rect = None
        self.selection_rect_item = None
        self.selection_pen = pg.mkPen(color='#2196F3', width=1, style=Qt.DashLine)
        self.selection_brush = pg.mkBrush(color=(33, 150, 243, 50))

    def handle_key_press(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_O:
            if not event.isAutoRepeat():
                self.is_selection_mode = True
                self.update_cursor()
                # Disable plot panning/zooming while selecting
                self.parent.position_plot.plotItem.vb.setMouseEnabled(x=False, y=False)
                
                # Disable other interactions
                if hasattr(self.parent.event_handler, 'disable_other_modes'):
                     pass
            return True
            
        elif event.key() == Qt.Key_P:
            # Enable move mode if O is also held (or independent if desired, but request says "in same")
            if not event.isAutoRepeat():
                self.is_moving_mode = True
                self.update_cursor()
                # Auto-pause simulation
                self.pause_simulation()
            return True
            
        return False
        
    def handle_key_release(self, event):
        """Handle key release events"""
        if event.key() == Qt.Key_O:
            if not event.isAutoRepeat():
                self.exit_selection_mode()
            return True
            
        elif event.key() == Qt.Key_P:
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
            
            # Check for Move Mode (O + P)
            if self.is_moving_mode:
                # If we have a previous selection, move THAT group
                if self.selected_zones:
                    self.moving_zones = self.selected_zones
                else:
                    # Otherwise find zone under cursor
                    zone = self.find_zone_at_point(x, y)
                    if zone:
                        self.moving_zones = [zone]
                    else:
                        self.moving_zones = []
                
                if self.moving_zones:
                    self.is_dragging_move = True
                    self.parent.position_plot.setCursor(Qt.ClosedHandCursor)
                    return True
            else:
                # Default Selection Mode - Clear previous selection if starting new one?
                # Usually new selection clears old one, unless Shift is held? 
                # For simplicity, clear old selection on new drag start
                self.clear_highlights()
                self.selected_zones = []
                
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
            
        # Handle Moving Zones
        if self.is_moving_mode and self.is_dragging_move and self.start_pos:
            dx = x - self.start_pos[0]
            dy = y - self.start_pos[1]
            
            for zone in self.moving_zones:
                self.move_zone(zone, dx, dy)
            
            self.start_pos = (x, y) # Update start pos for incremental delta
            self.parent.nlos_manager.update_nlos_zones() # Redraw
            # Also update highlights to follow
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
                self.moving_zones = [] if not self.selected_zones else self.selected_zones
                # If we were moving a temporary selection (highlighted), keep it? 
                # If we moved just one item under cursor (no previous selection), moving_zones was [zone]
                # We should probably clear moving_zones but NOT selected_zones
                self.update_cursor()
                return True
                
            if self.is_drag_selecting:
                self.is_drag_selecting = False
                
                # Find items in rect
                if self.current_rect:
                    self.selected_zones = self.find_items_in_rect(self.current_rect)
                    self.highlight_selection()
                    
                    if self.selected_zones:
                        # Only open editor if we are NOT holding P (intent is selection/editing)
                        if not self.is_moving_mode:
                             self.parent.nlos_manager.edit_multiple_zones(self.selected_zones)
                
                self.cleanup_selection_visuals()
                
                # IMPORTANT: Do NOT exit selection mode here anymore, 
                # so the user can follow up with 'P' to move the selected items.
                return True
            
        return False

    def highlight_selection(self):
        """Draw bounding boxes around selected zones"""
        self.clear_highlights()
        
        for zone in self.selected_zones:
            # Create a rect item for highlight
            if isinstance(zone, NLOSZone):
                rect = QRectF(min(zone.x1, zone.x2), min(zone.y1, zone.y2), 
                              abs(zone.x1-zone.x2), abs(zone.y1-zone.y2))
            elif isinstance(zone, PolygonNLOSZone):
                xs = [p[0] for p in zone.points]
                ys = [p[1] for p in zone.points]
                rect = QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
            elif isinstance(zone, MovingNLOSZone):
                corners = zone.get_corners()
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                rect = QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
            else:
                continue
                
            item = QGraphicsRectItem(rect)
            item.setPen(pg.mkPen('#FFEB3B', width=2)) # Yellow highlight
            item.setBrush(pg.mkBrush(None))
            self.parent.position_plot.addItem(item)
            self.highlight_items.append((item, zone))
            
    def update_highlights(self):
        """Update highlight positions"""
        for item, zone in self.highlight_items:
             if isinstance(zone, NLOSZone):
                rect = QRectF(min(zone.x1, zone.x2), min(zone.y1, zone.y2), 
                              abs(zone.x1-zone.x2), abs(zone.y1-zone.y2))
             elif isinstance(zone, PolygonNLOSZone):
                xs = [p[0] for p in zone.points]
                ys = [p[1] for p in zone.points]
                rect = QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
             elif isinstance(zone, MovingNLOSZone):
                corners = zone.get_corners()
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                rect = QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
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
        self.selected_zones = []
        self.cleanup_selection_visuals()
        self.clear_highlights()
        self.update_cursor()
        # Re-enable plot panning/zooming
        self.parent.position_plot.plotItem.vb.setMouseEnabled(x=True, y=True)
        
    def find_zone_at_point(self, x, y):
        """Find the first zone containing the point (x, y)"""
        all_zones = (self.parent.channel_conditions.nlos_zones + 
                    self.parent.channel_conditions.moving_nlos_zones)
        # Check in reverse order (topmost first)
        for zone in reversed(all_zones):
             if hasattr(zone, 'contains_point'):
                 if zone.contains_point(Position(x, y)):
                     return zone
        return None
        
    def move_zone(self, zone, dx, dy):
        """Translate a zone by delta (dx, dy)"""
        if isinstance(zone, NLOSZone):
            zone.x1 += dx
            zone.x2 += dx
            zone.y1 += dy
            zone.y2 += dy
            
        elif isinstance(zone, PolygonNLOSZone):
            new_points = []
            for p in zone.points:
                new_points.append((p[0] + dx, p[1] + dy))
            zone.points = new_points
            
        elif isinstance(zone, MovingNLOSZone):
            # Move initial and final positions to shift the whole path
            zone.initial_pos = (zone.initial_pos[0] + dx, zone.initial_pos[1] + dy)
            zone.final_pos = (zone.final_pos[0] + dx, zone.final_pos[1] + dy)
            zone.update_position() # Update current pos based on new anchors


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


    def find_items_in_rect(self, rect):
        """Find NLOS zones intersecting with the selection rectangle"""
        selected_zones = []
        
        all_zones = (self.parent.channel_conditions.nlos_zones + 
                    self.parent.channel_conditions.moving_nlos_zones)
                    
        for zone in all_zones:
            if self.is_zone_in_rect(zone, rect):
                selected_zones.append(zone)
                
        return selected_zones
        
    def is_zone_in_rect(self, zone, rect):
        """Check if a zone intersects with the selection rectangle"""
        # Simplification: Check if bounding box overlaps or points inside
        # For precision, we can implement more complex intersection
        
        if isinstance(zone, NLOSZone):
            # Line segment check
            # Check if either endpoint is in rect
            p1_in = rect.contains(zone.x1, zone.y1)
            p2_in = rect.contains(zone.x2, zone.y2)
            
            # Also check if rect passes through line (line clipping)
            # For now, just checking points + easy inclusion is a good start
            # To be more robust, we could check if line intersects rect
            return p1_in or p2_in
            
        elif isinstance(zone, PolygonNLOSZone):
            # Check if any point is in rect
            for point in zone.points:
                if rect.contains(point[0], point[1]):
                    return True
            return False
            
        elif isinstance(zone, MovingNLOSZone):
            # Check corners
            corners = zone.get_corners()
            for point in corners:
                if rect.contains(point[0], point[1]):
                    return True
            return False
            
        return False
