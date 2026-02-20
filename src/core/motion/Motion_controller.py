import numpy as np
from src.core.uwb.uwb_devices import Tag, Position
from src.utils.resource_loader import get_data_path
import csv
import time
from datetime import datetime
from scipy.interpolate import CubicSpline, interp1d

class MotionController:
    # Add class variable to cache loaded trajectories
    _trajectory_cache = {}

    @staticmethod
    def update_tag_position(tag : Tag, movement_pattern : str, movement_speed : float, t : float , frequence = 1000 , point =(0,0)):
        prev_x = tag.position.x
        prev_y = tag.position.y
        prev_vx = tag.velocity.x
        prev_vy = tag.velocity.y
        
        # Initialize new_x and new_y with current position
        new_x = tag.position.x
        new_y = tag.position.y
        
        # Check if pattern is a custom trajectory
        if movement_pattern.startswith("Custom:"):
            # Get trajectory points from the pattern - now using cache
            trajectory_name = movement_pattern[7:]  # Remove "Custom:" prefix
            if trajectory_name not in MotionController._trajectory_cache:
                trajectory_points = MotionController.load_custom_trajectory(trajectory_name)
                if trajectory_points:
                    MotionController._trajectory_cache[trajectory_name] = trajectory_points
                else:
                    # Fallback to circular if trajectory loading fails
                    radius = 5
                    frequency = 0.5 * movement_speed
                    new_x = radius * np.cos(frequency * t)
                    new_y = radius * np.sin(frequency * t)
                    tag.position.x = new_x
                    tag.position.y = new_y
                    return
            
            trajectory_points = MotionController._trajectory_cache[trajectory_name]
            points = np.array(trajectory_points)
            
            # Calculate total path length
            diff = np.diff(points, axis=0)
            segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
            total_length = np.sum(segment_lengths)
            
            # Calculate cumulative distances along the path
            cumulative_distances = np.concatenate(([0], np.cumsum(segment_lengths)))
            
            # Remove duplicate points (where distance difference is 0)
            mask = np.concatenate(([True], np.diff(cumulative_distances) > 1e-10))
            points = points[mask]
            cumulative_distances = cumulative_distances[mask]
            
            # Calculate total time needed to complete the trajectory at given speed
            total_time = total_length / movement_speed
            
            # Create periodic time parameter
            current_distance = (t * movement_speed) % total_length
            
            # Add closing point if needed
            if not np.allclose(points[0], points[-1]):
                # Only add closing point if it would not create a duplicate
                if np.abs(cumulative_distances[-1] - total_length) > 1e-10:
                    points = np.vstack([points, points[0]])
                    cumulative_distances = np.append(cumulative_distances, total_length)
            
            # Create periodic interpolation based on distance
            x_interp = interp1d(cumulative_distances, points[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
            y_interp = interp1d(cumulative_distances, points[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            # Get interpolated position
            new_x = float(x_interp(current_distance))
            new_y = float(y_interp(current_distance))
            
            # Calculate velocities using finite differences
            dt_small = 0.01
            next_distance = (current_distance + movement_speed * dt_small) % total_length
            next_x = float(x_interp(next_distance))
            next_y = float(y_interp(next_distance))
            
            # Update velocities
            tag.velocity.x = (next_x - new_x) / dt_small
            tag.velocity.y = (next_y - new_y) / dt_small
        else:
            # Update position based on selected movement pattern
            if movement_pattern == "Circular":
                radius = 5
                frequency = 0.5 * movement_speed
                new_x = radius * np.cos(frequency * t)
                new_y = radius * np.sin(frequency * t)
            
            elif movement_pattern == "Figure 8":
                radius = 5
                frequency = 0.5 * movement_speed
                new_x = radius * np.cos(frequency * t)
                new_y = radius/2 * np.sin(2 * frequency * t)
            
            elif movement_pattern == "Square":
                side = 8
                period = (4 * side) / movement_speed
                t_in_cycle = t % period
                segment_duration = period / 4
                segment_progress = (t_in_cycle % segment_duration) / segment_duration

                if 0 <= t_in_cycle < segment_duration:  # Right edge, Y increasing from -side/2 to side/2
                    new_x = side / 2
                    new_y = -side / 2 + segment_progress * side
                elif segment_duration <= t_in_cycle < 2 * segment_duration:  # Top edge, X decreasing from side/2 to -side/2
                    new_x = side / 2 - segment_progress * side
                    new_y = side / 2
                elif 2 * segment_duration <= t_in_cycle < 3 * segment_duration:  # Left edge, Y decreasing from side/2 to -side/2
                    new_x = -side / 2
                    new_y = side / 2 - segment_progress * side
                else:  # Bottom edge, X increasing from -side/2 to side/2
                    new_x = -side / 2 + segment_progress * side
                    new_y = -side / 2
            
            elif movement_pattern == "Random Walk":
                step_size = 0.1 * movement_speed
                dx = np.random.normal(0, step_size)
                dy = np.random.normal(0, step_size)
                
                # Keep within bounds
                new_x = np.clip(tag.position.x + dx, -8, 8)
                new_y = np.clip(tag.position.y + dy, -8, 8)
            elif movement_pattern == "Foot Mounted":
                # Simple foot-mounted motion along rectangle perimeter with natural bounce
                rect_width, rect_height = 8.0, 6.0
                perimeter = 2 * (rect_width + rect_height)
                dist = (t * movement_speed) % perimeter
                bounce = 0.05 * abs(np.sin(8 * t))  # Natural walking bounce
                
                if dist < rect_width:  # Bottom edge
                    new_x = -rect_width/2 + dist
                    new_y = -rect_height/2 + bounce
                elif dist < (rect_width + rect_height):
                    new_x = rect_width/2 + bounce
                    new_y = -rect_height/2 + (dist - rect_width)
                elif dist < (2 * rect_width + rect_height):
                    new_x = rect_width/2 - (dist - (rect_width + rect_height))
                    new_y = rect_height/2 + bounce
                else:
                    new_x = -rect_width/2 + bounce
                    new_y = rect_height/2 - (dist - (2 * rect_width + rect_height))
            elif movement_pattern == "Fixed Point":
                # Stay at a fixed point (0,0)
                new_x = point[0]
                new_y = point[1]
                
                # Zero velocity for fixed point
                tag.velocity.x = 0
                tag.velocity.y = 0
        # Update tag position
        tag.position.x = new_x
        tag.position.y = new_y
        
        # Calculate dt (use fixed dt for stability)
        dt = 0.05  # 50ms update rate
        
        if not movement_pattern.startswith("Custom:"):
            # Calculate velocities for non-custom patterns
            tag.velocity.x = (new_x - prev_x) / dt
            tag.velocity.y = (new_y - prev_y) / dt
        
        # Calculate accelerations
        tag.acceleration.x = (tag.velocity.x - prev_vx) / dt
        tag.acceleration.y = (tag.velocity.y - prev_vy) / dt
        
        # Calculate orientation and angular velocity
        tag.orientation = np.arctan2(tag.velocity.y, tag.velocity.x)
        
        # Calculate angular velocity (change in orientation)
        speed = np.sqrt(tag.velocity.x**2 + tag.velocity.y**2)
        if speed > 0.01:  # Only update angular velocity when moving
            target_orientation = np.arctan2(tag.velocity.y, tag.velocity.x)
            angle_diff = np.arctan2(np.sin(target_orientation - tag.orientation), 
                                   np.cos(target_orientation - tag.orientation))
            tag.angular_velocity = angle_diff / dt
        else:
            tag.angular_velocity = 0
        
        # Update IMU data
        tag.update_imu(t)

    @staticmethod
    def save_custom_trajectory(name: str, points: list, sampling_freq: float = 50.0):
        """
        Save a custom trajectory to CSV file only
        
        Args:
            name: trajectory name
            points: list of [x,y] coordinates
            sampling_freq: sampling frequency in Hz (default 50Hz = 20ms period)
        """
        import os
        
        # Create trajectories directory if it doesn't exist
        traj_dir = get_data_path("data/trajectories")
        os.makedirs(traj_dir, exist_ok=True)
            
        # Save detailed trajectory to CSV with timestamps
        csv_path = os.path.join(traj_dir, f"{name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['timestamp', 'x', 'y', 'z'])
            
            # Calculate time step
            dt = 1.0 / sampling_freq
            
            # Write coordinates with timestamps
            for i, point in enumerate(points):
                timestamp = i * dt
                x, y = point
                z = 0  # Add z coordinate (0 for 2D trajectory)
                writer.writerow([f"{timestamp:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    @staticmethod
    def load_custom_trajectory(name: str) -> list:
        """Load a custom trajectory from CSV file"""
        import os
        
        # Load CSV (only supported format)
        traj_dir = get_data_path("data/trajectories")
        csv_path = os.path.join(traj_dir, f"{name}.csv")
        if os.path.exists(csv_path):
            points = []
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        points.append([float(row['x']), float(row['y'])])
                return points
            except Exception as e:
                print(f"Error loading trajectory {name}: {e}")
                return None
            
        return None

    @staticmethod
    def get_available_trajectories() -> list:
        """Get list of available custom trajectories from CSV files"""
        import os
        
        trajectories = set()
        traj_dir = get_data_path("data/trajectories")
        if os.path.exists(traj_dir):
            for file in os.listdir(traj_dir):
                # Only list CSV files
                if file.endswith(".csv"):
                    trajectories.add(file[:-4])  # Remove .csv extension
        return list(trajectories)

    @staticmethod
    def clear_trajectory_cache():
        """Clear the trajectory cache to free memory"""
        MotionController._trajectory_cache.clear()

    @staticmethod
    def delete_custom_trajectory(name: str) -> bool:
        """
        Delete a custom trajectory CSV file
        
        Args:
            name: trajectory name (without .csv extension)
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        import os
        traj_dir = get_data_path("data/trajectories")
        csv_path = os.path.join(traj_dir, f"{name}.csv")
        
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                # Clear from cache if present
                if name in MotionController._trajectory_cache:
                    del MotionController._trajectory_cache[name]
                return True
            except Exception as e:
                print(f"Error deleting trajectory {name}: {e}")
                return False
        return False
