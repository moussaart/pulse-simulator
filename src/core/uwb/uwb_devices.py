import numpy as np
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING
from enum import Enum
from src.core.uwb.imu import IMUSimulator, IMUData


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

        self.imu_data_on = True
    
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
            full_result = None
        else:
            # Single call to measure_distance_detailed — avoids the previous
            # pattern of calling measure_distance + measure_distance_detailed which
            # effectively ran the entire channel model TWICE per anchor.
            try:
                full_result = channel_conditions.measure_distance_detailed(
                    true_distance, is_los, anchor_pos=anchor.position)
                final_distance = full_result.measured_distance
                noise = full_result.measurement_std
            except AttributeError:
                final_distance, noise = channel_conditions.measure_distance(
                    true_distance, is_los, anchor_pos=anchor.position)
                full_result = None
        
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
        
        return final_distance, messages, full_result

    def measure_distance(self, anchor: Anchor, channel_conditions: 'ChannelConditions', 
                        true_time: float) -> float:
        # Use the existing measure_distance_with_logs method
        distance, _, _ = self.measure_distance_with_logs(anchor, channel_conditions, true_time)
        return distance 