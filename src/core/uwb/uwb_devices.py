import numpy as np
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING
from enum import Enum


if TYPE_CHECKING:
    from src.core.uwb.channel_model import ChannelConditions

class MessageType(Enum):
    POLL = 1
    RESPONSE = 2
    FINAL = 3

@dataclass
class UWBMessage:
    msg_type: MessageType
    tx_timestamp: float
    rx_timestamp: float = 0.0

@dataclass
class Position:
    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

class UWBDevice:
    def __init__(self, position: Position):
        self.position = position
        # Clock drift (ppm)
        self.clock_drift = np.random.normal(0, 2)  # ±2 ppm clock drift
        self.processing_delay = 0.000002  # 2 microseconds processing delay
        
    def get_time(self, true_time: float) -> float:
        # Convert true time to device time with clock drift
        return true_time * (1 + self.clock_drift * 1e-6)
    
    def process_message(self, msg: UWBMessage, true_time: float) -> float:
        # Simulate message processing delay
        return self.get_time(true_time + self.processing_delay)

class Anchor(UWBDevice):
    _next_id = 1  # Class variable to track next available ID
    
    def __init__(self, position: Position):
        super().__init__(position)
        self.id = f"A{Anchor._next_id}"  # Anchor IDs will be A1, A2, A3, etc.
        Anchor._next_id += 1
        self.poll_rx_timestamp = 0.0
        self.response_tx_timestamp = 0.0
        # Add timestamps for DS-TWR
        self.final_rx_timestamp = 0.0
        self.ds_response_tx_timestamp = 0.0
    
    def process_poll(self, msg: UWBMessage, true_time: float) -> UWBMessage:
        # Process poll message and send response
        self.poll_rx_timestamp = self.get_time(true_time)
        self.response_tx_timestamp = self.process_message(msg, true_time)
        
        return UWBMessage(
            msg_type=MessageType.RESPONSE,
            tx_timestamp=self.response_tx_timestamp,
            rx_timestamp=self.poll_rx_timestamp
        )

class IMUSimulator:
    """
    Realistic IMU simulator that generates sensor data from tag kinematics.
    Simulates accelerometer and gyroscope with realistic noise and bias characteristics.
    """
    def __init__(self, sample_rate=100):
        """
        Initialize IMU simulator with realistic sensor parameters.
        
        Args:
            sample_rate: IMU sampling rate in Hz (default: 100 Hz)
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Sensor noise parameters (consumer-grade IMU)
        self.acc_noise_std = 0.02        # Accelerometer noise (m/s²)
        self.gyro_noise_std = 0.001      # Gyroscope noise (rad/s)
        
        # Bias parameters
        self.acc_bias = np.random.normal(0, 0.05, 3)   # Accelerometer bias (m/s²)
        self.gyro_bias = np.random.normal(0, 0.01, 3)  # Gyroscope bias (rad/s)
        self.bias_instability = 1e-5     # Bias random walk coefficient
        
        # Gravity constant
        self.g = 9.81
        
        # Previous state for differentiation
        self.prev_position = None
        self.prev_velocity = None
        self.prev_orientation = None
        
    def reset(self):
        """Reset the simulator state"""
        self.prev_position = None
        self.prev_velocity = None
        self.prev_orientation = None
        self.acc_bias = np.random.normal(0, 0.05, 3)
        self.gyro_bias = np.random.normal(0, 0.01, 3)
        
    def generate_imu_data(self, position, orientation, dt):
        """
        Generate realistic IMU measurements from tag kinematics.
        
        Args:
            position: Position object with x, y, z coordinates
            orientation: Tag orientation in radians (yaw angle)
            dt: Time step since last measurement
            
        Returns:
            tuple: (acceleration[3], angular_velocity[3]) in body frame
        """
        current_pos = np.array([position.x, position.y, position.z])
        
        # Initialize on first call
        if self.prev_position is None:
            self.prev_position = current_pos
            self.prev_velocity = np.zeros(3)
            self.prev_orientation = orientation
            # Return stationary measurements
            acc_body = np.array([0, 0, self.g]) + self.acc_bias + np.random.normal(0, self.acc_noise_std, 3)
            gyro_body = self.gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
            return acc_body, gyro_body
        
        # Calculate velocity and acceleration in world frame
        velocity = (current_pos - self.prev_position) / dt
        acceleration = (velocity - self.prev_velocity) / dt
        
        # Calculate angular velocity (yaw rate for 2D motion)
        delta_yaw = orientation - self.prev_orientation
        # Normalize to [-pi, pi]
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
        yaw_rate = delta_yaw / dt
        
        # Transform acceleration to body frame
        cos_yaw = np.cos(orientation)
        sin_yaw = np.sin(orientation)
        R_world_to_body = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Accelerometer measures specific force (linear acceleration + gravity)
        gravity_world = np.array([0, 0, -self.g])
        specific_force_world = acceleration - gravity_world
        specific_force_body = R_world_to_body @ specific_force_world
        
        # Gyroscope measures angular velocity in body frame
        angular_velocity_body = np.array([0, 0, yaw_rate])  # Only yaw for 2D motion
        
        # Add sensor imperfections
        # Update bias (random walk)
        self.acc_bias += np.random.normal(0, self.bias_instability, 3)
        self.gyro_bias += np.random.normal(0, self.bias_instability / 10, 3)
        
        # Add noise and bias
        acc_measured = specific_force_body + self.acc_bias + np.random.normal(0, self.acc_noise_std, 3)
        gyro_measured = angular_velocity_body + self.gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
        
        # Update previous state
        self.prev_position = current_pos
        self.prev_velocity = velocity
        self.prev_orientation = orientation
        
        return acc_measured, gyro_measured


class IMUData:
    """Storage class for IMU measurements"""
    def __init__(self):
        self.acc_x = np.array([], dtype=np.float64)
        self.acc_y = np.array([], dtype=np.float64)
        self.acc_z = np.array([], dtype=np.float64)
        self.gyro_x = np.array([], dtype=np.float64)
        self.gyro_y = np.array([], dtype=np.float64)
        self.gyro_z = np.array([], dtype=np.float64)
        self.timestamps = np.array([], dtype=np.float64)
        
    def add_measurement(self, t, ax, ay, az, gx, gy, gz):
        """Add a new IMU measurement"""
        self.timestamps = np.append(self.timestamps, t)
        self.acc_x = np.append(self.acc_x, ax)
        self.acc_y = np.append(self.acc_y, ay)
        self.acc_z = np.append(self.acc_z, az)
        self.gyro_x = np.append(self.gyro_x, gx)
        self.gyro_y = np.append(self.gyro_y, gy)
        self.gyro_z = np.append(self.gyro_z, gz)
        
        # Keep only last 1000 measurements
        max_samples = 1000
        if len(self.timestamps) > max_samples:
            self.timestamps = self.timestamps[-max_samples:]
            self.acc_x = self.acc_x[-max_samples:]
            self.acc_y = self.acc_y[-max_samples:]
            self.acc_z = self.acc_z[-max_samples:]
            self.gyro_x = self.gyro_x[-max_samples:]
            self.gyro_y = self.gyro_y[-max_samples:]
            self.gyro_z = self.gyro_z[-max_samples:]

    def clear(self):
        """Clear all stored measurements"""
        self.acc_x = np.array([], dtype=np.float64)
        self.acc_y = np.array([], dtype=np.float64)
        self.acc_z = np.array([], dtype=np.float64)
        self.gyro_x = np.array([], dtype=np.float64)
        self.gyro_y = np.array([], dtype=np.float64)
        self.gyro_z = np.array([], dtype=np.float64)
        self.timestamps = np.array([], dtype=np.float64)
        
    def __str__(self):
        """Return string representation of IMU data"""
        return (f"IMU Data:\n"
                f"  Timestamps: {self.timestamps[-5:] if len(self.timestamps) > 0 else []}\n"
                f"  Accelerometer (x,y,z): \n"
                f"    x: {self.acc_x[-5:] if len(self.acc_x) > 0 else []}\n" 
                f"    y: {self.acc_y[-5:] if len(self.acc_y) > 0 else []}\n"
                f"    z: {self.acc_z[-5:] if len(self.acc_z) > 0 else []}\n"
                f"  Gyroscope (x,y,z): \n"
                f"    x: {self.gyro_x[-5:] if len(self.gyro_x) > 0 else []}\n"
                f"    y: {self.gyro_y[-5:] if len(self.gyro_y) > 0 else []}\n"
                f"    z: {self.gyro_z[-5:] if len(self.gyro_z) > 0 else []}")

    def __repr__(self):
        """Return string representation of IMU data object"""
        return f"IMUData(samples={len(self.timestamps)})"

class Tag(UWBDevice):
    _next_id = 1  # Class variable to track next available ID
    
    def __init__(self, position: Position):
        super().__init__(position)
        self.id = f"T{Tag._next_id}"  # Tag IDs will be T1, T2, T3, etc.
        Tag._next_id += 1
        
        # Motion state
        self.velocity = Position(0, 0)
        self.acceleration = Position(0, 0)
        self.orientation = 0  # heading in radians
        self.angular_velocity = 0  # rad/s
        
        # IMU simulator and data storage
        self.imu_simulator = IMUSimulator(sample_rate=100)  # 100 Hz sampling rate
        self.imu_data = IMUData()
        self.last_update_time = 0
        
        # TWR timestamps
        self.poll_tx_timestamp = 0.0
        self.final_tx_timestamp = 0.0
        self.response_rx_timestamp = 0.0
        # Add timestamps for DS-TWR
        self.final_rx_timestamp = 0.0
        self.ds_response_tx_timestamp = 0.0
    
    def move(self, dx: float, dy: float):
        self.position.x += dx
        self.position.y += dy 

    def update_imu(self, t: float):
        """
        Update IMU measurements using the IMUSimulator.
        Generates realistic accelerometer and gyroscope data from tag motion.
        
        Args:
            t: Current simulation time
        """
        # Calculate time step
        dt = max(t - self.last_update_time, 0.001) if self.last_update_time > 0 else 0.001
        self.last_update_time = t
        
        # Generate IMU measurements from current tag kinematics
        try:
            acc_measured, gyro_measured = self.imu_simulator.generate_imu_data(
                self.position, 
                self.orientation, 
                dt
            )
            
            # Store the measurement
            self.imu_data.add_measurement(
                t, 
                acc_measured[0], acc_measured[1], acc_measured[2],
                gyro_measured[0], gyro_measured[1], gyro_measured[2]
            )
            
        except Exception as e:
            print(f"IMU update error: {e}")
            # Safe fallback: stationary measurement
            self.imu_data.add_measurement(t, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)

    def measure_distance_with_logs(self, anchor: Anchor, channel_conditions: 'ChannelConditions', 
                                 true_time: float, mode: str = "SS-TWR") -> Tuple[float, list]:
        messages = []
        
        # Calculate true distance and check LOS condition
        true_distance = self.position.distance_to(anchor.position)
        is_los = channel_conditions.check_los_to_anchor(anchor.position, self.position)
        
        propagation_time = true_distance / channel_conditions.c
        # Use geometric ray-tracing based ranging if enabled, else fallback
        if getattr(channel_conditions, 'use_ray_tracing', False):
            final_distance, noise = channel_conditions.measure_distance_geometric(anchor.position, self.position)
        else:
            final_distance, noise = channel_conditions.measure_distance(true_distance, is_los, anchor_pos=anchor.position)
        
        if mode == "SS-TWR":
            # Single-Sided TWR
            messages.append(f"Using SS-TWR Protocol between {self.id} and {anchor.id}:")
            
            # 1. Send POLL message
            self.poll_tx_timestamp = self.get_time(true_time)
            poll_msg = UWBMessage(MessageType.POLL, self.poll_tx_timestamp)
            messages.append(f"[{true_time:.6f}] {self.id} -> {anchor.id}: POLL message (TX: {self.poll_tx_timestamp:.6f})")
            
            # 2. Anchor receives POLL and sends RESPONSE
            anchor_rx_time = true_time + propagation_time
            response_msg = anchor.process_poll(poll_msg, anchor_rx_time)
            messages.append(f"[{anchor_rx_time:.6f}] {anchor.id} <- {self.id}: POLL received (RX: {anchor.poll_rx_timestamp:.6f})")
            messages.append(f"[{anchor_rx_time + anchor.processing_delay:.6f}] {anchor.id} -> {self.id}: RESPONSE (TX: {response_msg.tx_timestamp:.6f})")
            
            # 3. Tag receives RESPONSE
            response_rx_time = anchor_rx_time + propagation_time + anchor.processing_delay
            self.response_rx_timestamp = self.get_time(response_rx_time)
            messages.append(f"[{response_rx_time:.6f}] {self.id} <- {anchor.id}: RESPONSE received (RX: {self.response_rx_timestamp:.6f})")
            
            # Calculate round trip time and distance
            round_trip_time = self.response_rx_timestamp - self.poll_tx_timestamp
            measured_distance = (round_trip_time * channel_conditions.c - 
                               anchor.processing_delay * channel_conditions.c) / 2
            
        else:  # DS-TWR
            messages.append(f"Using DS-TWR Protocol between {self.id} and {anchor.id}:")
            
            # 1. Send POLL message
            self.poll_tx_timestamp = self.get_time(true_time)
            poll_msg = UWBMessage(MessageType.POLL, self.poll_tx_timestamp)
            messages.append(f"[{true_time:.6f}] {self.id} -> {anchor.id}: POLL message (TX: {self.poll_tx_timestamp:.6f})")
            
            # 2. Anchor receives POLL and sends RESPONSE
            anchor_rx_time = true_time + propagation_time
            response_msg = anchor.process_poll(poll_msg, anchor_rx_time)
            messages.append(f"[{anchor_rx_time:.6f}] {anchor.id} <- {self.id}: POLL received (RX: {anchor.poll_rx_timestamp:.6f})")
            messages.append(f"[{anchor_rx_time + anchor.processing_delay:.6f}] {anchor.id} -> {self.id}: RESPONSE (TX: {response_msg.tx_timestamp:.6f})")
            
            # 3. Tag receives RESPONSE and sends FINAL
            response_rx_time = anchor_rx_time + propagation_time + anchor.processing_delay
            self.response_rx_timestamp = self.get_time(response_rx_time)
            self.final_tx_timestamp = self.process_message(response_msg, response_rx_time)
            messages.append(f"[{response_rx_time:.6f}] {self.id} <- {anchor.id}: RESPONSE received (RX: {self.response_rx_timestamp:.6f})")
            messages.append(f"[{response_rx_time + self.processing_delay:.6f}] {self.id} -> {anchor.id}: FINAL (TX: {self.final_tx_timestamp:.6f})")
            
            # 4. Anchor receives FINAL and sends DS-RESPONSE
            final_rx_time = response_rx_time + propagation_time + self.processing_delay
            anchor.final_rx_timestamp = anchor.get_time(final_rx_time)
            anchor.ds_response_tx_timestamp = anchor.process_message(UWBMessage(MessageType.FINAL, self.final_tx_timestamp), final_rx_time)
            messages.append(f"[{final_rx_time:.6f}] {anchor.id} <- {self.id}: FINAL received (RX: {anchor.final_rx_timestamp:.6f})")
            messages.append(f"[{final_rx_time + anchor.processing_delay:.6f}] {anchor.id} -> {self.id}: DS-RESPONSE (TX: {anchor.ds_response_tx_timestamp:.6f})")
            
            # 5. Tag receives DS-RESPONSE
            ds_response_rx_time = final_rx_time + propagation_time + anchor.processing_delay
            self.ds_response_rx_timestamp = self.get_time(ds_response_rx_time)
            messages.append(f"[{ds_response_rx_time:.6f}] {self.id} <- {anchor.id}: DS-RESPONSE received (RX: {self.ds_response_rx_timestamp:.6f})")
            
            # Calculate distance using corrected DS-TWR formula
            round_trip_1 = self.response_rx_timestamp - self.poll_tx_timestamp
            round_trip_2 = self.ds_response_rx_timestamp - self.final_tx_timestamp
            reply_time_1 = response_msg.tx_timestamp - anchor.poll_rx_timestamp
            reply_time_2 = self.final_tx_timestamp - self.response_rx_timestamp
            
            # Corrected DS-TWR formula
            tof = (round_trip_1 * round_trip_2 - reply_time_1 * reply_time_2) / \
                  (round_trip_1 + round_trip_2)
            measured_distance = (tof * channel_conditions.c) / 2
            
            messages.append(f"Round trip 1: {round_trip_1:.9f}s")
            messages.append(f"Round trip 2: {round_trip_2:.9f}s")
            messages.append(f"Reply time 1: {reply_time_1:.9f}s")
            messages.append(f"Reply time 2: {reply_time_2:.9f}s")
            messages.append(f"Calculated ToF: {tof:.9f}s")
        
        # Add measurement noise
        
        
        messages.append(f"Calculated distance: {final_distance:.3f}m")
        messages.append("-" * 50)  # Separator between measurements
        
        # In a real simulation, we expect the 'measured_distance' to match 'final_distance' (plus noise)
        # But 'final_distance' comes from channel_model.measure_distance_detailed (Physics)
        # 'measured_distance' comes from TWR logic (Protocol)
        # To connect them, we should pass the physics result.
        
        # For now, we return the physics result 'final_distance' as the primary measurement,
        # but technically TWR protocol should yield it. 
        # The 'channel_conditions.measure_distance' call at the top serves as the "Physics Oracle".
        
        # We need to return the FULL RangingResult if available.
        # Since 'measure_distance' returns (dist, std), we might need to check if it was 'measure_distance_detailed'
        
        # We need to re-call measure_distance_detailed to get the full object if we want CIR.
        # The previous call: 
        #   final_distance, noise = channel_conditions.measure_distance(true_distance, is_los)
        # wrapped it.
        
        # Let's direct call measure_distance_detailed if possible to get CIR
        try:
             full_result = channel_conditions.measure_distance_detailed(true_distance, is_los)
             # Use values from full result to ensure consistency
             final_distance = full_result.measured_distance
        except AttributeError:
             full_result = None
        
        return final_distance, messages, full_result

    def measure_distance(self, anchor: Anchor, channel_conditions: 'ChannelConditions', 
                        true_time: float) -> float:
        # Use the existing measure_distance_with_logs method
        distance, _, _ = self.measure_distance_with_logs(anchor, channel_conditions, true_time)
        return distance 