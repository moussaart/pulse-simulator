from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .uwb_types import PathLossParams, SVModelParams, CM1_LOS_0_4M, CM2_NLOS_0_4M
from .uwb_devices import Position
import time
import numpy as np

@dataclass
class NLOSZone:
    """NLOS zone definition with IEEE 802.15.3a channel model parameters"""
    x1: float
    y1: float
    x2: float
    y2: float
    # NLOS bias (0.05m typical for indoor environments)
    error_bias: float = 0.05  # meters
    # Noise factor (1.5 typical for indoor NLOS)
    noise_factor: float = 1.5
    # Path loss parameters specific to NLOS
    path_loss_params: PathLossParams = field(default_factory=lambda: PathLossParams(
        path_loss_exponent=2.5,
        reference_loss_db=-43.0,
        shadow_fading_std=4.0,
        frequency_decay_factor=1.0 # Added explicit default
    ))
    # S-V model parameters for IEEE 802.15.3a compliant multipath modeling
    sv_params: SVModelParams = field(default_factory=lambda: CM2_NLOS_0_4M)
    # RMS delay spread for NLOS error calculation (seconds)
    rms_delay_spread: float = 15e-9  # 15 ns typical indoor NLOS

    def contains_point(self, point: Position) -> bool:
        return (self.x1 <= point.x <= self.x2 and 
                self.y1 <= point.y <= self.y2)

    def get_corners(self) -> List[Tuple[float, float]]:
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2),
            (self.x1, self.y1)
        ]

class PolygonNLOSZone:
    """Polygon-shaped NLOS zone with IEEE 802.15.3a channel model support"""
    def __init__(self, points: List[Tuple[float, float]], 
                 error_bias: float = 0.05,
                 noise_factor: float = 1.5,
                 path_loss_params: PathLossParams = None,
                 sv_params: SVModelParams = None,
                 rms_delay_spread: float = 15e-9):
        self.points = points
        self.error_bias = error_bias
        self.noise_factor = noise_factor
        self.path_loss_params = path_loss_params or PathLossParams(
            path_loss_exponent=2.5,
            reference_loss_db=-43.0,
            shadow_fading_std=4.0,
            frequency_decay_factor=1.0
        )
        # IEEE 802.15.3a S-V model parameters
        self.sv_params = sv_params or CM2_NLOS_0_4M
        self.rms_delay_spread = rms_delay_spread
        # No material properties stored here as per redesign

    def contains_point(self, point: Position) -> bool:
        x, y = point.x, point.y
        inside = False
        j = len(self.points) - 1
        
        for i in range(len(self.points)):
            if (((self.points[i][1] > y) != (self.points[j][1] > y)) and
                (x < (self.points[j][0] - self.points[i][0]) * (y - self.points[i][1]) /
                     (self.points[j][1] - self.points[i][1]) + self.points[i][0])):
                inside = not inside
            j = i
        return inside

    def get_corners(self) -> List[Tuple[float, float]]:
        return self.points + [self.points[0]]

class MovingNLOSZone:
    """Dynamic NLOS zone with movement capabilities and IEEE 802.15.3a channel model"""
    def __init__(self, 
                 initial_position: Tuple[float, float],
                 final_position: Tuple[float, float],
                 shape_type: str = "circle",
                 speed: float = 1.0,  # meters per second
                 error_bias: float = 0.05,
                 noise_factor: float = 1.5,
                 path_loss_params: PathLossParams = None,
                 sv_params: SVModelParams = None,
                 rms_delay_spread: float = 15e-9,
                 width: float = 1.0,   # For rectangles
                 height: float = 1.0,  # For rectangles
                 size: float = 1.0,    # For regular shapes (radius or side)
                 rotation_speed: float = 0.5 # radians per second
                 ):
        self.initial_pos = initial_position
        self.final_pos = final_position
        self.current_pos = initial_position
        self.shape_type = shape_type.lower()
        self.speed = speed
        self.error_bias = error_bias
        self.noise_factor = noise_factor
        self.path_loss_params = path_loss_params or PathLossParams(
            path_loss_exponent=2.5,
            reference_loss_db=-43.0,
            shadow_fading_std=4.0,
            frequency_decay_factor=1.0
        )
        # IEEE 802.15.3a S-V model parameters
        self.sv_params = sv_params or CM2_NLOS_0_4M
        self.rms_delay_spread = rms_delay_spread
        
        # Dimensions
        self.size = size
        self.width = width
        self.height = height
        
        # Rotation
        self.angle = 0.0  # Current angle for rotation
        self.rotation_speed = rotation_speed
        self.start_time = time.time()
        
        # Validate shape type
        valid_shapes = ["circle", "square", "rectangle", "triangle", "diamond", "hexagon"]
        if self.shape_type not in valid_shapes:
            raise ValueError(f"Invalid shape type. Must be one of: {valid_shapes}")

    def update_position(self, current_time: float = None):
        """Update the current position based on time and movement parameters"""
        if current_time is None:
            current_time = time.time()
        
        elapsed_time = current_time - self.start_time
        total_distance = np.sqrt((self.final_pos[0] - self.initial_pos[0])**2 + 
                               (self.final_pos[1] - self.initial_pos[1])**2)
        
        if total_distance > 0:
            # Calculate progress (0 to 1)
            # Use modulo to create a back-and-forth or continuous loop
            # Here we implement a ping-pong loop (0->1->0)
            cycle_distance = total_distance * 2
            distance_traveled = (self.speed * elapsed_time) % cycle_distance
            
            if distance_traveled <= total_distance:
                # Forward direction
                progress = distance_traveled / total_distance
            else:
                # Backward direction
                progress = 2.0 - (distance_traveled / total_distance)
        else:
            progress = 0.0
        
        # Linear interpolation between initial and final positions
        self.current_pos = (
            self.initial_pos[0] + (self.final_pos[0] - self.initial_pos[0]) * progress,
            self.initial_pos[1] + (self.final_pos[1] - self.initial_pos[1]) * progress
        )
        
        # Update rotation angle
        self.angle = (self.angle + self.rotation_speed * 0.1) % (2 * np.pi)

    def get_points(self) -> List[Tuple[float, float]]:
        """Get the points defining the current shape at current position with rotation"""
        x, y = self.current_pos
        points = []
        
        if self.shape_type == "circle":
            # Approximate circle with 20 points
            for i in range(20):
                theta = 2 * np.pi * i / 20
                # Rotation doesn't affect a circle visually unless textured, but we apply it for consistency
                theta += self.angle
                points.append((
                    x + self.size * np.cos(theta),
                    y + self.size * np.sin(theta)
                ))
            return points
            
        elif self.shape_type == "square":
            half_size = self.size / 2
            base_points = [
                (-half_size, -half_size),
                (half_size, -half_size),
                (half_size, half_size),
                (-half_size, half_size)
            ]
            
        elif self.shape_type == "rectangle":
            half_w = self.width / 2
            half_h = self.height / 2
            base_points = [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h)
            ]
            
        elif self.shape_type == "triangle":
            # Equilateral triangle
            base_points = [
                (0, self.size),
                (-self.size * np.cos(np.pi/6), -self.size * np.sin(np.pi/6)),
                (self.size * np.cos(np.pi/6), -self.size * np.sin(np.pi/6))
            ]
            
        elif self.shape_type == "diamond":
            base_points = [
                (0, self.size),
                (self.size, 0),
                (0, -self.size),
                (-self.size, 0)
            ]
            
        elif self.shape_type == "hexagon":
            base_points = []
            for i in range(6):
                theta = 2 * np.pi * i / 6
                base_points.append((
                    self.size * np.cos(theta),
                    self.size * np.sin(theta)
                ))
        
        else:
            return []

        # Apply rotation and translation to base points
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        
        for px, py in base_points:
            # Rotate
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            # Translate
            points.append((x + rx, y + ry))
            
        return points

    def contains_point(self, point: Position) -> bool:
        """Check if a point is inside the moving NLOS zone"""
        x, y = point.x, point.y
        points = self.get_points()
        
        # Use ray casting algorithm
        inside = False
        j = len(points) - 1
        
        for i in range(len(points)):
            if (((points[i][1] > y) != (points[j][1] > y)) and
                (x < (points[j][0] - points[i][0]) * (y - points[i][1]) /
                     (points[j][1] - points[i][1]) + points[i][0])):
                inside = not inside
            j = i
        return inside

    def get_corners(self) -> List[Tuple[float, float]]:
        """Get the corners of the current shape"""
        points = self.get_points()
        return points + [points[0]]  # Close the shape

    def set_movement_parameters(self, speed: float = None, rotation_speed: float = None):
        """Update movement parameters"""
        if speed is not None:
            self.speed = speed
        if rotation_speed is not None:
            self.rotation_speed = rotation_speed

    def set_shape_parameters(self, size: float = None, width: float = None, height: float = None):
        """Update shape parameters"""
        if size is not None:
            self.size = size
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
