"""
NLOS Manager Module
Handles NLOS zone creation, editing, and visualization
"""
import numpy as np
import pyqtgraph as pg
import cv2
from PyQt5.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtCore import Qt
from src.core.uwb.channel_model import NLOSZone, PolygonNLOSZone, PathLossParams, MovingNLOSZone
from src.core.uwb.uwb_devices import Position
from src.gui.windows.nlos_config_window import NLOSConfigWindow
from src.gui.windows.image_import_window import ImageImportWindow
from src.gui.windows.moving_nlos_window import MovingNLOSWindow


class NLOSManager:
    """Manages NLOS zones and their visualization"""
    
    def __init__(self, parent):
        self.parent = parent
        self.zone_colors = []
        self.zone_items = {}  # Map zone objects to their plot items

    
    def create_line_zone(self, start_pos, end_pos):
        """Create NLOS zone from line segment"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Calculate normal vector (perpendicular)
        line_width = 0.1
        nx = -dy/length * line_width
        ny = dx/length * line_width
        
        # Create rectangle corners
        corners = [
            (x1 + nx, y1 + ny),
            (x2 + nx, y2 + ny),
            (x2 - nx, y2 - ny),
            (x1 - nx, y1 - ny),
            (x1 + nx, y1 + ny)  # Close the polygon
        ]
        
        # Create polygon NLOS zone
        zone = PolygonNLOSZone(
            points=corners,
            error_bias=0.05,
            noise_factor=1.5,
            path_loss_params=PathLossParams(
                path_loss_exponent=2.5,
                reference_loss_db=-43.0,
                shadow_fading_std=4.0
            )
        )
        
        # Use red color for new zones
        red_color = [255, 0, 0]
        self.zone_colors.append((zone, red_color))
        self.parent.channel_conditions.nlos_zones.append(zone)
        
        # Update visualization
        self.update_nlos_zones()
    
    def create_polygon_from_lines(self, polygon_points):
        """Create polygon NLOS zone from drawn points"""
        if len(polygon_points) < 2:
            return
        
        width = 0.1
        left_points = []
        right_points = []
        
        # Process each line segment with miter joints
        for i in range(len(polygon_points)):
            p1 = polygon_points[i]
            
            # Get previous, current and next directions
            if i > 0:
                prev_dx = p1[0] - polygon_points[i-1][0]
                prev_dy = p1[1] - polygon_points[i-1][1]
                prev_len = np.sqrt(prev_dx*prev_dx + prev_dy*prev_dy)
                if prev_len > 0:
                    prev_dx, prev_dy = prev_dx/prev_len, prev_dy/prev_len
                else:
                    prev_dx = prev_dy = 0
            else:
                prev_dx = prev_dy = 0
            
            if i < len(polygon_points)-1:
                next_dx = polygon_points[i+1][0] - p1[0]
                next_dy = polygon_points[i+1][1] - p1[1]
                next_len = np.sqrt(next_dx*next_dx + next_dy*next_dy)
                if next_len > 0:
                    next_dx, next_dy = next_dx/next_len, next_dy/next_len
                else:
                    next_dx = next_dy = 0
            else:
                next_dx = next_dy = 0
            
            # Calculate miter vector
            if i == 0:  # First point
                nx = -next_dy
                ny = next_dx
                miter_x, miter_y = nx, ny
            elif i == len(polygon_points)-1:  # Last point
                nx = -prev_dy
                ny = prev_dx
                miter_x, miter_y = nx, ny
            else:  # Middle points
                avg_dx = (prev_dx + next_dx) / 2
                avg_dy = (prev_dy + next_dy) / 2
                avg_len = np.sqrt(avg_dx*avg_dx + avg_dy*avg_dy)
                
                if avg_len > 0.0001:
                    miter_x = -avg_dy / avg_len
                    miter_y = avg_dx / avg_len
                    
                    angle = np.arccos(prev_dx*next_dx + prev_dy*next_dy)
                    if angle < np.pi * 0.1:
                        miter_len = 1.0 / np.sin(angle/2)
                        miter_x *= miter_len
                        miter_y *= miter_len
                else:
                    miter_x = -prev_dy
                    miter_y = prev_dx
            
            # Add offset points
            left_points.append((
                p1[0] + miter_x * width,
                p1[1] + miter_y * width
            ))
            right_points.insert(0, (
                p1[0] - miter_x * width,
                p1[1] - miter_y * width
            ))
        
        # Combine points to form complete polygon
        complete_polygon = left_points + right_points + [left_points[0]]
        
        # Create polygon NLOS zone
        zone = PolygonNLOSZone(
            points=complete_polygon,
            error_bias=0.05,
            noise_factor=1.5,
            path_loss_params=PathLossParams(
                path_loss_exponent=2.5,
                reference_loss_db=-43.0,
                shadow_fading_std=4.0
            )
        )
        
        # Use red color for new zones
        red_color = [255, 0, 0]
        self.zone_colors.append((zone, red_color))
        self.parent.channel_conditions.nlos_zones.append(zone)
        
        # Update visualization
        self.update_nlos_zones()
    
    def update_nlos_zones(self):
        """Update visualization of all NLOS zones"""
        # Remove existing NLOS zone visualizations
        for item in self.parent.position_plot.items():
            if isinstance(item, pg.PlotDataItem) and hasattr(item, 'nlos_zone'):
                self.parent.position_plot.removeItem(item)
        
        self.zone_items.clear()
        
        # Redraw all NLOS zones with their specific colors
        all_zones = (self.parent.channel_conditions.nlos_zones + 
                    self.parent.channel_conditions.moving_nlos_zones)
        
        for zone in all_zones:
            # Get corners based on zone type
            if isinstance(zone, NLOSZone):
                corners = [(zone.x1, zone.y1), (zone.x2, zone.y1),
                          (zone.x2, zone.y2), (zone.x1, zone.y2),
                          (zone.x1, zone.y1)]
            elif isinstance(zone, MovingNLOSZone):
                corners = zone.get_corners()
            else:  # PolygonNLOSZone
                corners = zone.points
                if corners and corners[0] != corners[-1]:
                    corners = list(corners) + [corners[0]]
            
            x_values, y_values = zip(*corners)
            
            # Get color for this zone
            color = self.get_zone_color(zone)
            
            # Create zone visualization with color
            zone_item = pg.PlotDataItem(
                x_values, y_values,
                fillLevel=0,
                brush=pg.mkBrush(color[0], color[1], color[2], 50),
                pen=pg.mkPen(color[0], color[1], color[2], 255)
            )
            zone_item.nlos_zone = True
            self.parent.position_plot.addItem(zone_item)
            self.zone_items[zone] = zone_item
        
        # Make sure anchor and tag points are on top
        if hasattr(self.parent, 'plot_manager'):
            self.parent.plot_manager.update_anchor_visualization(
                self.parent.position_plot,
                self.parent.anchors,
                self.parent.channel_conditions,
                self.parent.tag
            )

    def update_moving_visualizations(self):
        """Efficiently update visualization of moving NLOS zones"""
        for zone in self.parent.channel_conditions.moving_nlos_zones:
            if zone in self.zone_items:
                corners = zone.get_corners()
                x_values, y_values = zip(*corners)
                self.zone_items[zone].setData(x_values, y_values)
            else:
                # If for some reason it's missing (shouldn't happen if update_nlos_zones called correctly)
                self.update_nlos_zones()
                return

    def toggle_moving_zone_placement(self):
        """Toggle interactive moving zone placement mode"""
        if self.parent.nlos_widgets['add_moving_btn'].isChecked():
            # Activate mode
            self.parent.event_handler.disable_other_modes('picking_moving_trajectory')
            self.parent.event_handler.picking_moving_trajectory = True
            self.parent.nlos_widgets['add_moving_btn'].setText("Click Start Point")
            self.parent.position_plot.setCursor(Qt.CrossCursor)
        else:
            # Deactivate mode
            self.parent.event_handler.picking_moving_trajectory = False
            self.parent.nlos_widgets['add_moving_btn'].setText("🏃 Moving Obstacle")
            self.parent.position_plot.setCursor(Qt.ArrowCursor)
    
    def get_zone_color(self, zone):
        """Get color for a specific zone"""
        for z, c in self.zone_colors:
            if z is zone:
                return c
            if isinstance(z, NLOSZone) and isinstance(zone, NLOSZone):
                if (z.x1 == zone.x1 and z.y1 == zone.y1 and 
                    z.x2 == zone.x2 and z.y2 == zone.y2):
                    return c
            elif isinstance(z, PolygonNLOSZone) and isinstance(zone, PolygonNLOSZone):
                if z.points == zone.points:
                    return c
        
        # Try to get color from matching configuration
        params = {
            'path_loss_exponent': zone.path_loss_params.path_loss_exponent,
            'shadow_fading_std': zone.path_loss_params.shadow_fading_std,
            'error_bias': zone.error_bias,
            'noise_factor': zone.noise_factor
        }
        color = self.parent.nlos_config_manager.get_color_for_parameters(params)
        if color is None:
            color = [255, 0, 0]  # Default red color
        self.zone_colors.append((zone, color))
        return color
    
    def edit_multiple_zones(self, zones):
        """Open dialog to edit multiple NLOS zones at once"""
        if not zones:
            return
            
        # Use first zone for initial values and color
        first_zone = zones[0]
        current_color = self.get_zone_color(first_zone)
        
        # Pass loaded configs from parent app
        configs = getattr(self.parent, 'loaded_configs', {})
        
        # Pass list of zones to config window
        from src.gui.windows.nlos_config_window import NLOSConfigWindow
        dialog = NLOSConfigWindow(zones, self.parent.nlos_config_manager, current_color, self.parent, loaded_configs=configs)
        
        if dialog.exec_() == QDialog.Accepted:
            # Update colors for all zones if changed
            # Params update is handled inside dialog
            for zone in zones:
                self.update_zone_color(zone, dialog.current_color)
            
            self.update_nlos_zones()

    def edit_nlos_zone(self, zone):
        """Open dialog to edit NLOS zone parameters"""
        current_color = self.get_zone_color(zone)
        
        # Pass loaded configs from parent app
        configs = getattr(self.parent, 'loaded_configs', {})
        
        dialog = NLOSConfigWindow(zone, self.parent.nlos_config_manager, current_color, self.parent, loaded_configs=configs)
        if dialog.exec_() == QDialog.Accepted:
            # Update zone parameters
            zone.path_loss_params.path_loss_exponent = dialog.pl_exp.value()
            zone.path_loss_params.shadow_fading_std = dialog.shadow_std.value()
            zone.error_bias = dialog.error_bias.value()
            zone.noise_factor = dialog.noise_factor.value()
            
            # Update zone color
            self.update_zone_color(zone, dialog.current_color)
            
            # Update visualization
            self.update_nlos_zones()
    
    def update_zone_color(self, zone, new_color):
        """Update color for a specific zone"""
        for i, (z, _) in enumerate(self.zone_colors):
            if z is zone:
                self.zone_colors[i] = (zone, new_color)
                break
            if isinstance(z, NLOSZone) and isinstance(zone, NLOSZone):
                if (z.x1 == zone.x1 and z.y1 == zone.y1 and 
                    z.x2 == zone.x2 and z.y2 == zone.y2):
                    self.zone_colors[i] = (zone, new_color)
                    break
            elif isinstance(z, PolygonNLOSZone) and isinstance(zone, PolygonNLOSZone):
                if z.points == zone.points:
                    self.zone_colors[i] = (zone, new_color)
                    break
    
    def apply_preset_to_zone(self, zone, config_name):
        """Apply a saved configuration preset to a zone"""
        config, color = self.parent.nlos_config_manager.get_config(config_name)
        if config:
            # Update zone parameters
            zone.path_loss_params.path_loss_exponent = config["path_loss_exponent"]
            zone.path_loss_params.shadow_fading_std = config["shadow_fading_std"]
            zone.error_bias = config["error_bias"]
            zone.noise_factor = config["noise_factor"]
            
            # Update zone color
            self.update_zone_color(zone, color)
            
            # Update visualization
            self.update_nlos_zones()
    
    def clear_nlos_zones(self):
        """Clear all NLOS regions"""
        self.parent.channel_conditions.nlos_zones.clear()
        self.parent.channel_conditions.moving_nlos_zones.clear()
        self.zone_colors.clear()
        self.zone_items.clear()
        self.update_nlos_zones()

    def delete_zone_at(self, x, y):
        """Delete an NLOS zone at the given coordinates"""
        point = Position(x, y)
        
        # Check standard/polygon zones
        for i, zone in enumerate(self.parent.channel_conditions.nlos_zones):
            is_in_zone = False
            if isinstance(zone, NLOSZone):
                # Add small buffer for easier clicking on lines/thin walls
                margin = 0.2
                is_in_zone = (min(zone.x1, zone.x2) - margin <= x <= max(zone.x1, zone.x2) + margin and 
                              min(zone.y1, zone.y2) - margin <= y <= max(zone.y1, zone.y2) + margin)
            else:  # PolygonNLOSZone
                # Use strict containment for polygons
                is_in_zone = zone.contains_point(point)
                
            if is_in_zone:
                # Remove from list
                self.parent.channel_conditions.nlos_zones.pop(i)
                # Remove from colors list
                self._remove_from_colors(zone)
                # Update visual
                self.update_nlos_zones()
                return

        # Check moving zones
        for i, zone in enumerate(self.parent.channel_conditions.moving_nlos_zones):
            corners = zone.get_corners()
            # Simple bounding box check first
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                # Precise check - treat as polygon
                poly = PolygonNLOSZone(corners) # Temporary wrapper for check
                if poly.contains_point(point):
                    self.parent.channel_conditions.moving_nlos_zones.pop(i)
                    self._remove_from_colors(zone)
                    self.update_nlos_zones()
                    return

    def _remove_from_colors(self, zone):
        """Helper to remove zone from color mapping"""
        for i, (z, _) in enumerate(self.zone_colors):
            if z is zone:
                self.zone_colors.pop(i)
                break
    
    def import_from_image(self):
        """Import NLOS zones from a floor plan image using interactive window"""
        # Create and show the configuration window
        dialog = ImageImportWindow(self.parent)
        if dialog.exec_() != QDialog.Accepted:
            return
            
        # Get data from dialog
        data = dialog.get_data()
        
        segments_px = data['segments_px']
        if not segments_px:
            QMessageBox.warning(self.parent, "Import Failed", "No wall segments detected or selected.")
            return

        img_w_px, img_h_px = data['image_size']
        target_w, target_h = data['target_size']
        off_x, off_y = data['offset']
        wall_width = data['wall_width']
        nlos_params = data['nlos_params']
        
        # Calculate scale factors
        scale_x = target_w / img_w_px
        scale_y = target_h / img_h_px
        
        zones_created = 0
        
        for p1_px, p2_px in segments_px:
            x1_px, y1_px = p1_px
            x2_px, y2_px = p2_px
            
            # Convert to simulation coordinates
            # Image: origin top-left, y down
            # Sim: origin at offset, y up
            sim_x1 = x1_px * scale_x + off_x
            sim_y1 = (img_h_px - y1_px) * scale_y + off_y
            sim_x2 = x2_px * scale_x + off_x
            sim_y2 = (img_h_px - y2_px) * scale_y + off_y
            
            # Create the wall
            self._create_wall_from_points_with_params(
                (sim_x1, sim_y1),
                (sim_x2, sim_y2),
                wall_width,
                nlos_params
            )
            zones_created += 1
            
        # Update visualization
        self.update_nlos_zones()
        
        # Show summary
        QMessageBox.information(
            self.parent, "Import Complete",
            f"Successfully imported {zones_created} wall segments."
        )

    def add_moving_zone_dialog(self, initial_pos=None, final_pos=None):
        """Open dialog to add a moving NLOS zone"""
        # Pass loaded configs from parent app
        configs = getattr(self.parent, 'loaded_configs', {})
        
        if initial_pos and final_pos:
            dialog = MovingNLOSWindow(self.parent, initial_pos=initial_pos, final_pos=final_pos, loaded_configs=configs)
        else:
            dialog = MovingNLOSWindow(self.parent, loaded_configs=configs)
            
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            
            from src.core.uwb.uwb_types import PathLossParams, SVModelParams
            
            # Construct params objects
            pl_params = PathLossParams(
                path_loss_exponent=data['path_loss_exp'],
                reference_loss_db=-43.0,
                shadow_fading_std=data['shadow_std'],
                frequency_decay_factor=data['freq_decay']
            )
            
            sv_params = SVModelParams(
                cluster_decay=data['cluster_decay'],
                ray_decay=data['ray_decay'],
                rms_delay_spread=data['rms_delay'],
                path_loss_exponent=data['path_loss_exp'],
                shadow_fading_std=data['shadow_std']
            )
            
            # Add the zone to channel conditions
            self.parent.channel_conditions.add_moving_nlos_zone(
                initial_position=data['initial_pos'],
                final_position=data['final_pos'],
                shape_type=data['shape_type'],
                speed=data['speed'],
                error_bias=data['error_bias'],
                noise_factor=data['noise_factor'],
                size=data['size'],
                width=data['width'],
                height=data['height'],
                rotation_speed=data['rotation_speed'],
                path_loss_params=pl_params,
                sv_params=sv_params,
                rms_delay_spread=data['rms_delay']
            )
            
            # Get the newly added zone (it's the last one)
            new_zone = self.parent.channel_conditions.moving_nlos_zones[-1]
            
            # Store its color
            self.zone_colors.append((new_zone, data['color']))
            
            # Update visualization
            self.update_nlos_zones()

    def _create_wall_from_points_with_params(self, start, end, width, params):
        """Create a thin rectangular wall with specific NLOS params"""
        x1, y1 = start
        x2, y2 = end
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
            
        nx = -dy/length * width
        ny = dx/length * width
        
        corners = [
            (x1 + nx, y1 + ny),
            (x2 + nx, y2 + ny),
            (x2 - nx, y2 - ny),
            (x1 - nx, y1 - ny),
            (x1 + nx, y1 + ny)
        ]
        
        zone = PolygonNLOSZone(
            points=corners,
            error_bias=params['error_bias'],
            noise_factor=params['noise_factor'],
            path_loss_params=PathLossParams(
                path_loss_exponent=params['path_loss_exp'],
                reference_loss_db=-43.0,
                shadow_fading_std=params['shadow_std']
            )
        )
        
        red_color = [255, 0, 0]
        self.zone_colors.append((zone, red_color))
        self.parent.channel_conditions.nlos_zones.append(zone)
    
    def _skeletonize(self, binary_img):
        """Skeletonize a binary image to get single-pixel-wide lines"""
        skeleton = np.zeros(binary_img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        img = binary_img.copy()
        while True:
            # Erode the image
            eroded = cv2.erode(img, element)
            # Open the eroded image (erode then dilate)
            temp = cv2.dilate(eroded, element)
            # Subtract to get the skeleton pixels
            temp = cv2.subtract(img, temp)
            # Add to skeleton
            skeleton = cv2.bitwise_or(skeleton, temp)
            # Update for next iteration
            img = eroded.copy()
            
            # Stop when image is empty
            if cv2.countNonZero(img) == 0:
                break
        
        return skeleton
    
    def _merge_collinear_segments(self, lines, max_distance):
        """Merge nearby collinear line segments"""
        if not lines:
            return []
        
        # Convert to list of line segments
        segments = list(lines)
        merged = []
        used = set()
        
        for i, (p1, p2) in enumerate(segments):
            if i in used:
                continue
            
            # Start with this segment
            start = p1
            end = p2
            used.add(i)
            
            # Try to extend by merging with collinear segments
            merged_any = True
            while merged_any:
                merged_any = False
                for j, (p3, p4) in enumerate(segments):
                    if j in used:
                        continue
                    
                    # Check if segments are collinear and close
                    if self._can_merge_segments((start, end), (p3, p4), max_distance):
                        # Merge by extending to the farthest endpoints
                        all_points = [start, end, p3, p4]
                        # Find the two points that are farthest apart
                        max_dist = 0
                        best_pair = (start, end)
                        for pi in all_points:
                            for pj in all_points:
                                dist = np.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2)
                                if dist > max_dist:
                                    max_dist = dist
                                    best_pair = (pi, pj)
                        start, end = best_pair
                        used.add(j)
                        merged_any = True
            
            merged.append((start, end))
        
        return merged
    
    def _can_merge_segments(self, seg1, seg2, max_distance):
        """Check if two segments can be merged (collinear and close)"""
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        # Calculate angles of both segments
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)
        
        # Normalize angles to [0, pi]
        angle1 = angle1 % np.pi
        angle2 = angle2 % np.pi
        
        # Check if angles are similar (within 10 degrees)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > np.pi/18:  # 10 degrees
            return False
        
        # Check if endpoints are close
        endpoints = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        min_dist = float('inf')
        for i in range(2):
            for j in range(2, 4):
                dist = np.sqrt((endpoints[i][0] - endpoints[j][0])**2 + 
                              (endpoints[i][1] - endpoints[j][1])**2)
                min_dist = min(min_dist, dist)
        
        return min_dist < max_distance
    
    def _create_wall_from_points(self, start, end, width):
        """Create a thin rectangular wall from two points"""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Calculate normal vector (perpendicular)
        nx = -dy/length * width
        ny = dx/length * width
        
        # Create rectangle corners
        corners = [
            (x1 + nx, y1 + ny),
            (x2 + nx, y2 + ny),
            (x2 - nx, y2 - ny),
            (x1 - nx, y1 - ny),
            (x1 + nx, y1 + ny)  # Close the polygon
        ]
        
        # Create polygon NLOS zone
        zone = PolygonNLOSZone(
            points=corners,
            error_bias=0.05,
            noise_factor=1.5,
            path_loss_params=PathLossParams(
                path_loss_exponent=2.5,
                reference_loss_db=-43.0,
                shadow_fading_std=4.0
            )
        )
        
        # Use red color
        red_color = [255, 0, 0]
        self.zone_colors.append((zone, red_color))
        self.parent.channel_conditions.nlos_zones.append(zone)

