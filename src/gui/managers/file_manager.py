"""
File Manager Module
Handles file operations (save/load map configurations)
"""
import json
from PyQt5.QtWidgets import QFileDialog
from src.core.uwb.uwb_devices import Anchor, Position
from src.core.uwb.channel_model import NLOSZone, PolygonNLOSZone, PathLossParams, MovingNLOSZone


class FileManager:
    """Manages file operations for map configurations"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def save_map_config(self):
        """Save current map configuration to file"""
        # Create configuration dictionary with null-safe widget access
        algorithm = 'Trilateration'
        movement_pattern = 'Circular'
        movement_speed = 1.0
        timestep_ms = 5
        
        try:
            if self.parent.algo_combo:
                algorithm = self.parent.algo_combo.currentText()
        except RuntimeError:
            pass
            
        try:
            if self.parent.pattern_combo:
                movement_pattern = self.parent.pattern_combo.currentText()
        except RuntimeError:
            pass
            
        try:
            if self.parent.speed_slider:
                movement_speed = self.parent.speed_slider.value() / 10.0
        except RuntimeError:
            pass
            
        try:
            if self.parent.timestep_slider:
                timestep_ms = self.parent.timestep_slider.value()
        except RuntimeError:
            pass

        try:
            self.parent.renormalize_anchors()
        except AttributeError:
             pass

        config = {
            'anchors': self.serialize_anchors(),
            'nlos_zones': self.serialize_nlos_zones(),
            'algorithm': algorithm,
            'movement_pattern': movement_pattern,
            'movement_speed': movement_speed,
            'timestep_ms': timestep_ms,
        }
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Save Map Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            # Add .json extension if not present
            if not file_path.endswith('.json'):
                file_path += '.json'
            
            # Save to file
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
            except Exception as e:
                self.parent.show_error_message("Error", f"Failed to save map: {str(e)}")
    
    def load_map_config(self):
        """Load map configuration from file"""
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Load Map Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Reset all interaction modes first
                if hasattr(self.parent, 'event_handler'):
                    self.parent.event_handler.reset_all_modes()
                
                # Clear current configuration
                self.parent.anchors.clear()
                self.parent.anchors.clear()
                self.parent.channel_conditions.nlos_zones.clear()
                self.parent.channel_conditions.moving_nlos_zones.clear()
                self.parent.nlos_manager.zone_colors.clear()
                if hasattr(self.parent.nlos_manager, 'zone_items'):
                    self.parent.nlos_manager.zone_items.clear()
                
                # Load anchors
                self.load_anchors(config['anchors'])
                
                # Load NLOS zones
                self.load_nlos_zones(config['nlos_zones'])
                
                # Load other settings (null-safe widget access)
                try:
                    if self.parent.algo_combo:
                        self.parent.algo_combo.setCurrentText(config['algorithm'])
                except RuntimeError:
                    pass
                
                try:
                    if self.parent.pattern_combo:
                        self.parent.pattern_combo.setCurrentText(config['movement_pattern'])
                except RuntimeError:
                    pass
                
                # Convert speed from m/s to slider value
                try:
                    if self.parent.speed_slider:
                        speed_slider_value = int(config['movement_speed'] * 10)
                        self.parent.speed_slider.setValue(speed_slider_value)
                except RuntimeError:
                    pass
                
                # Update visualization
                self.parent.renormalize_anchors()
                self.parent.nlos_manager.update_nlos_zones()
                
                if self.parent.distance_plots_window is not None:
                    try:
                        self.parent.distance_plots_window.update_anchors(self.parent.anchors)
                    except RuntimeError:
                        self.parent.distance_plots_window = None
                
                if 'timestep_ms' in config:
                    try:
                        if self.parent.timestep_slider:
                            self.parent.timestep_slider.setValue(config['timestep_ms'])
                    except RuntimeError:
                        pass
                else:
                    try:
                        if self.parent.timestep_slider:
                            self.parent.timestep_slider.setValue(5)
                    except RuntimeError:
                        pass
                    
            except Exception as e:
                self.parent.show_error_message("Error", f"Failed to load map: {str(e)}")
    
    def serialize_anchors(self):
        """Serialize anchors to list of dictionaries"""
        return [
            {
                'id': anchor.id,
                'x': anchor.position.x,
                'y': anchor.position.y
            } for anchor in self.parent.anchors
        ]
    
    def serialize_nlos_zones(self):
        """Serialize NLOS zones to list of dictionaries"""
        zones = []
        # Combine all zones for serialization
        all_zones_with_colors = self.parent.nlos_manager.zone_colors
        
        for zone, color in all_zones_with_colors:
            # Common parameters
            params = {
                'path_loss_exponent': zone.path_loss_params.path_loss_exponent,
                'shadow_fading_std': zone.path_loss_params.shadow_fading_std,
                'frequency_decay_factor': getattr(zone.path_loss_params, 'frequency_decay_factor', 1.0),
                'error_bias': zone.error_bias,
                'noise_factor': zone.noise_factor,
                'rms_delay_spread': getattr(zone, 'rms_delay_spread', 15e-9)
            }
            
            # S-V Params
            if hasattr(zone, 'sv_params') and zone.sv_params:
                params['sv_params'] = {
                    'cluster_decay': zone.sv_params.cluster_decay,
                    'ray_decay': zone.sv_params.ray_decay,
                    'cluster_arrival_rate': zone.sv_params.cluster_arrival_rate,
                    'ray_arrival_rate': zone.sv_params.ray_arrival_rate
                }
            
            if isinstance(zone, MovingNLOSZone):
                zone_data = {
                    'type': 'moving',
                    'initial_pos': zone.initial_pos,
                    'final_pos': zone.final_pos,
                    'shape_type': zone.shape_type,
                    'speed': zone.speed,
                    'rotation_speed': getattr(zone, 'rotation_speed', 0.5),
                    'size': zone.size,
                    'width': zone.width,
                    'height': zone.height,
                    'color': color,
                    'parameters': params
                }
            elif isinstance(zone, NLOSZone):
                zone_data = {
                    'type': 'rectangle',
                    'x1': zone.x1,
                    'y1': zone.y1,
                    'x2': zone.x2,
                    'y2': zone.y2,
                    'color': color,
                    'parameters': params
                }
            else:  # PolygonNLOSZone
                zone_data = {
                    'type': 'polygon',
                    'points': zone.points,
                    'color': color,
                    'parameters': params
                }
            zones.append(zone_data)
        return zones
    
    def load_anchors(self, anchors_data):
        """Load anchors from serialized data"""
        for anchor_data in anchors_data:
            anchor = Anchor(Position(anchor_data['x'], anchor_data['y']))
            anchor.id = anchor_data['id']
            self.parent.anchors.append(anchor)
    
    def load_nlos_zones(self, zones_data):
        """Load NLOS zones from serialized data"""
        from src.core.uwb.uwb_types import SVModelParams
        
        for zone_data in zones_data:
            zone_type = zone_data.get('type', 'rectangle')
            params_data = zone_data['parameters']
            
            # Reconstruct PathLossParams
            pl_params = PathLossParams(
                path_loss_exponent=params_data['path_loss_exponent'],
                shadow_fading_std=params_data['shadow_fading_std'],
                frequency_decay_factor=params_data.get('frequency_decay_factor', 1.0)
            )
            
            # Reconstruct SVModelParams
            sv_params = None
            if 'sv_params' in params_data:
                svp = params_data['sv_params']
                sv_params = SVModelParams(
                    cluster_decay=svp.get('cluster_decay', 60.0),
                    ray_decay=svp.get('ray_decay', 20.0),
                    cluster_arrival_rate=svp.get('cluster_arrival_rate', 2.0),
                    ray_arrival_rate=svp.get('ray_arrival_rate', 0.5),
                    path_loss_exponent=params_data['path_loss_exponent'],
                    shadow_fading_std=params_data['shadow_fading_std']
                )
            
            rms_delay = params_data.get('rms_delay_spread', 15e-9)
            
            if zone_type == 'moving':
                zone = MovingNLOSZone(
                    initial_position=tuple(zone_data['initial_pos']),
                    final_position=tuple(zone_data['final_pos']),
                    shape_type=zone_data['shape_type'],
                    speed=zone_data['speed'],
                    error_bias=params_data['error_bias'],
                    noise_factor=params_data['noise_factor'],
                    path_loss_params=pl_params,
                    sv_params=sv_params,
                    rms_delay_spread=rms_delay,
                    width=zone_data.get('width', 1.0),
                    height=zone_data.get('height', 1.0),
                    size=zone_data.get('size', 1.0),
                    rotation_speed=zone_data.get('rotation_speed', 0.5)
                )
                self.parent.channel_conditions.moving_nlos_zones.append(zone)
                
            elif zone_type == 'rectangle':
                zone = NLOSZone(
                    zone_data['x1'], zone_data['y1'],
                    zone_data['x2'], zone_data['y2'],
                    error_bias=params_data['error_bias'],
                    noise_factor=params_data['noise_factor'],
                    path_loss_params=pl_params,
                    sv_params=sv_params,
                    rms_delay_spread=rms_delay
                )
                self.parent.channel_conditions.nlos_zones.append(zone)
                
            else:  # polygon
                zone = PolygonNLOSZone(
                    points=zone_data['points'],
                    error_bias=params_data['error_bias'],
                    noise_factor=params_data['noise_factor'],
                    path_loss_params=pl_params,
                    sv_params=sv_params,
                    rms_delay_spread=rms_delay
                )
                self.parent.channel_conditions.nlos_zones.append(zone)
            
            self.parent.nlos_manager.zone_colors.append((zone, zone_data['color']))
    
    def clean_map(self):
        """Clean the map by removing all anchors and NLOS zones"""
        # Reset all interaction modes first
        if hasattr(self.parent, 'event_handler'):
            self.parent.event_handler.reset_all_modes()
        
        # Clear anchors (keeping minimum 3)
        self.parent.anchors = [
            Anchor(Position(-8, -8)),
            Anchor(Position(8, -8)),
            Anchor(Position(0, 8))
        ]
        
        # Clear NLOS zones
        # Clear NLOS zones
        self.parent.channel_conditions.nlos_zones.clear()
        self.parent.channel_conditions.moving_nlos_zones.clear()
        self.parent.nlos_manager.zone_colors.clear()
        if hasattr(self.parent.nlos_manager, 'zone_items'):
            self.parent.nlos_manager.zone_items.clear()
        
        # Update visualizations
        self.parent.plot_manager.update_anchor_visualization(
            self.parent.position_plot,
            self.parent.anchors,
            self.parent.channel_conditions,
            self.parent.tag
        )
        self.parent.update_anchor_list()
        self.parent.nlos_manager.update_nlos_zones()
        
        if self.parent.distance_plots_window is not None:
            self.parent.distance_plots_window.update_anchors(self.parent.anchors)
        
        # Update trajectory plan
        if hasattr(self.parent, 'trajectory_manager'):
            self.parent.trajectory_manager.update_trajectory_plan()
        
        # Reset target point
        self.parent.point = (0.0, 0.0)
        self.parent.plot_items['target_point_marker'].setData([], [])

