"""
Data Collector Module
Core data collection with synchronized timestamping for AI training.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Any
from collections import deque
import numpy as np
import copy


@dataclass
class ChannelLinkData:
    """Channel/EM data for a single tag-anchor link"""
    anchor_id: str
    is_los: bool
    snr_db: float
    snr_linear: float
    path_loss_db: float
    noise_std: float
    signal_quality: float
    channel_model: str = "saleh_valenzuela"
    # Channel model parameters
    center_frequency_hz: float = 6.5e9
    bandwidth_hz: float = 500e6
    path_loss_exponent: float = 2.0
    shadow_fading_std: float = 2.0
    noise_model: str = "gaussian"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'anchor_id': self.anchor_id,
            'is_los': self.is_los,
            'snr_db': self.snr_db,
            'snr_linear': self.snr_linear,
            'path_loss_db': self.path_loss_db,
            'noise_std': self.noise_std,
            'signal_quality': self.signal_quality,
            'channel_model': self.channel_model,
            'center_frequency_hz': self.center_frequency_hz,
            'bandwidth_hz': self.bandwidth_hz,
            'path_loss_exponent': self.path_loss_exponent,
            'shadow_fading_std': self.shadow_fading_std,
            'noise_model': self.noise_model
        }


@dataclass  
class FilterOutput:
    """Output data from a localization filter"""
    filter_name: str
    estimated_position: Tuple[float, float]
    estimation_error: float
    state_vector: Optional[np.ndarray] = None
    state_covariance: Optional[np.ndarray] = None
    measurement_covariance: Optional[np.ndarray] = None
    process_noise_covariance: Optional[np.ndarray] = None
    innovation: Optional[np.ndarray] = None
    # Additional filter-specific parameters
    filter_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        def _array_to_list(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr
            
        return {
            'filter_name': self.filter_name,
            'estimated_position': list(self.estimated_position),
            'estimation_error': self.estimation_error,
            'state_vector': _array_to_list(self.state_vector),
            'state_covariance': _array_to_list(self.state_covariance),
            'measurement_covariance': _array_to_list(self.measurement_covariance),
            'process_noise_covariance': _array_to_list(self.process_noise_covariance),
            'innovation': _array_to_list(self.innovation),
            'filter_params': self.filter_params
        }


@dataclass
class DataSample:
    """Single timestamped data sample containing all collected data"""
    timestamp: float
    
    # Ground truth geometry
    tag_position_gt: Tuple[float, float, float]
    anchor_positions: List[Tuple[float, float, float]]
    anchor_ids: List[str]
    
    # Measurements
    distances_measured: List[float]
    distances_true: List[float]
    measurement_errors: List[float]
    measurement_noise_stds: List[float]
    
    # LOS/NLOS labels
    los_conditions: List[bool]
    
    # Channel/EM data per link
    channel_data: List[ChannelLinkData] = field(default_factory=list)
    
    # Filter outputs
    filter_outputs: Dict[str, FilterOutput] = field(default_factory=dict)
    
    # IMU data (if available)
    imu_acceleration: Optional[Tuple[float, float, float]] = None
    imu_angular_velocity: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'tag_position_gt': list(self.tag_position_gt),
            'anchor_positions': [list(p) for p in self.anchor_positions],
            'anchor_ids': self.anchor_ids,
            'distances_measured': self.distances_measured,
            'distances_true': self.distances_true,
            'measurement_errors': self.measurement_errors,
            'measurement_noise_stds': self.measurement_noise_stds,
            'los_conditions': self.los_conditions,
            'channel_data': [cd.to_dict() for cd in self.channel_data],
            'filter_outputs': {k: v.to_dict() for k, v in self.filter_outputs.items()},
            'imu_acceleration': list(self.imu_acceleration) if self.imu_acceleration else None,
            'imu_angular_velocity': list(self.imu_angular_velocity) if self.imu_angular_velocity else None
        }
    
    def to_flat_dict(self) -> dict:
        """Convert to flattened dictionary for DataFrame creation"""
        flat = {
            'timestamp': self.timestamp,
            'tag_x': self.tag_position_gt[0],
            'tag_y': self.tag_position_gt[1],
            'tag_z': self.tag_position_gt[2],
        }
        
        # Add per-anchor data
        for i, anchor_id in enumerate(self.anchor_ids):
            flat[f'anchor_{anchor_id}_x'] = self.anchor_positions[i][0]
            flat[f'anchor_{anchor_id}_y'] = self.anchor_positions[i][1]
            flat[f'dist_meas_{anchor_id}'] = self.distances_measured[i]
            flat[f'dist_true_{anchor_id}'] = self.distances_true[i]
            flat[f'is_los_{anchor_id}'] = self.los_conditions[i]
            
            if i < len(self.channel_data):
                cd = self.channel_data[i]
                flat[f'snr_db_{anchor_id}'] = cd.snr_db
                flat[f'path_loss_db_{anchor_id}'] = cd.path_loss_db
                flat[f'signal_quality_{anchor_id}'] = cd.signal_quality
        
        # Add filter outputs
        for filter_name, fo in self.filter_outputs.items():
            prefix = filter_name.replace(' ', '_').lower()
            flat[f'{prefix}_est_x'] = fo.estimated_position[0]
            flat[f'{prefix}_est_y'] = fo.estimated_position[1]
            flat[f'{prefix}_error'] = fo.estimation_error
        
        # Add IMU data
        if self.imu_acceleration:
            flat['imu_acc_x'] = self.imu_acceleration[0]
            flat['imu_acc_y'] = self.imu_acceleration[1]
            flat['imu_acc_z'] = self.imu_acceleration[2]
        if self.imu_angular_velocity:
            flat['imu_gyro_x'] = self.imu_angular_velocity[0]
            flat['imu_gyro_y'] = self.imu_angular_velocity[1]
            flat['imu_gyro_z'] = self.imu_angular_velocity[2]
            
        return flat


class DataBuffer:
    """Ring buffer for storing data samples with configurable size"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._callbacks: List[Callable[[DataSample], None]] = []
        
    def add(self, sample: DataSample) -> None:
        """Add a sample to the buffer and notify callbacks"""
        self._buffer.append(sample)
        for callback in self._callbacks:
            try:
                callback(sample)
            except Exception as e:
                print(f"Callback error: {e}")
                
    def get_all(self) -> List[DataSample]:
        """Get all samples in the buffer"""
        return list(self._buffer)
    
    def get_latest(self, n: int = 1) -> List[DataSample]:
        """Get the n most recent samples"""
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]
    
    def get_by_time_range(self, start_time: float, end_time: float) -> List[DataSample]:
        """Get samples within a time range"""
        return [s for s in self._buffer if start_time <= s.timestamp <= end_time]
    
    def clear(self) -> None:
        """Clear all samples from the buffer"""
        self._buffer.clear()
        
    def register_callback(self, callback: Callable[[DataSample], None]) -> None:
        """Register a callback to be called when new samples are added"""
        self._callbacks.append(callback)
        
    def unregister_callback(self, callback: Callable[[DataSample], None]) -> None:
        """Unregister a callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            
    def __len__(self) -> int:
        return len(self._buffer)
    
    def __iter__(self):
        return iter(self._buffer)


class DataCollector:
    """
    Core data collector for AI training.
    Collects synchronized, timestamped data from the simulation.
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer = DataBuffer(max_size=buffer_size)
        self.is_collecting = False
        
        # Data selection flags
        self.collect_em_data = True
        self.collect_channel_data = True
        self.collect_snr = True
        self.collect_filter_outputs = True
        self.collect_ground_truth = True
        self.collect_imu_data = True
        
        # Adapters (set by TrainingDataAPI)
        self.channel_adapter = None
        self.filter_adapter = None
        self.geometry_adapter = None
        
    def configure(self, 
                  buffer_size: int = None,
                  em: bool = None,
                  channel: bool = None,
                  snr: bool = None,
                  filter_outputs: bool = None,
                  ground_truth: bool = None,
                  imu: bool = None) -> None:
        """Configure data collection settings"""
        if buffer_size is not None:
            self.buffer = DataBuffer(max_size=buffer_size)
        if em is not None:
            self.collect_em_data = em
        if channel is not None:
            self.collect_channel_data = channel
        if snr is not None:
            self.collect_snr = snr
        if filter_outputs is not None:
            self.collect_filter_outputs = filter_outputs
        if ground_truth is not None:
            self.collect_ground_truth = ground_truth
        if imu is not None:
            self.collect_imu_data = imu
            
    def start(self) -> None:
        """Start data collection"""
        self.is_collecting = True
        
    def stop(self) -> None:
        """Stop data collection"""
        self.is_collecting = False
        
    def reset(self) -> None:
        """Reset the buffer and stop collection"""
        self.stop()
        self.buffer.clear()
        
    def collect(self,
                timestamp: float,
                tag,
                anchors: List,
                measurements: List[float],
                channel_conditions,
                filter_state: Dict[str, Any] = None,
                estimated_pos: Tuple[float, float] = None,
                error: float = None,
                algorithm_name: str = None) -> Optional[DataSample]:
        """
        Collect a data sample from the current simulation state.
        
        Args:
            timestamp: Current simulation time
            tag: Tag object with position
            anchors: List of anchor objects
            measurements: Distance measurements
            channel_conditions: ChannelConditions object
            filter_state: Dictionary with filter state (state, P, R, Q)
            estimated_pos: Estimated position from filter
            error: Estimation error
            algorithm_name: Name of the current algorithm
            
        Returns:
            DataSample if collection is enabled, None otherwise
        """
        if not self.is_collecting:
            return None
            
        # Get tag ground truth position
        tag_pos_gt = (tag.position.x, tag.position.y, getattr(tag.position, 'z', 0.0))
        
        # Get anchor positions and IDs
        anchor_positions = []
        anchor_ids = []
        for anchor in anchors:
            anchor_positions.append((
                anchor.position.x, 
                anchor.position.y, 
                getattr(anchor.position, 'z', 0.0)
            ))
            anchor_ids.append(anchor.id)
        
        # Calculate true distances and errors
        distances_true = []
        measurement_errors = []
        measurement_noise_stds = []
        los_conditions = []
        channel_data = []
        
        for i, anchor in enumerate(anchors):
            # True distance
            true_dist = anchor.position.distance_to(tag.position)
            distances_true.append(true_dist)
            
            # Measurement error
            if i < len(measurements):
                meas_err = measurements[i] - true_dist
                measurement_errors.append(meas_err)
            else:
                measurement_errors.append(0.0)
                
            # Check LOS condition
            is_los = channel_conditions.check_los_to_anchor(anchor.position, tag.position)
            los_conditions.append(is_los)
            
            # Collect channel data if enabled
            if self.collect_channel_data and self.channel_adapter:
                link_data = self.channel_adapter.extract_link_data(
                    channel_conditions, anchor, tag)
                channel_data.append(link_data)
                measurement_noise_stds.append(link_data.noise_std)
            else:
                measurement_noise_stds.append(0.0)
        
        # Collect filter outputs if enabled
        filter_outputs = {}
        if self.collect_filter_outputs and filter_state and self.filter_adapter:
            if estimated_pos and error is not None:
                filter_output = self.filter_adapter.capture_state(
                    algorithm_name or "Unknown",
                    estimated_pos,
                    error,
                    filter_state.get('state'),
                    filter_state.get('P'),
                    filter_state.get('R'),
                    filter_state.get('Q')
                )
                filter_outputs[algorithm_name or "current"] = filter_output
        
        # Collect IMU data if enabled
        imu_acc = None
        imu_gyro = None
        if self.collect_imu_data and hasattr(tag, 'imu_data') and tag.imu_data:
            imu_data = tag.imu_data
            if len(imu_data.acc_x) > 0:
                imu_acc = (
                    float(imu_data.acc_x[-1]),
                    float(imu_data.acc_y[-1]),
                    float(imu_data.acc_z[-1])
                )
            if len(imu_data.gyro_x) > 0:
                imu_gyro = (
                    float(imu_data.gyro_x[-1]),
                    float(imu_data.gyro_y[-1]),
                    float(imu_data.gyro_z[-1])
                )
        
        # Create data sample
        sample = DataSample(
            timestamp=timestamp,
            tag_position_gt=tag_pos_gt,
            anchor_positions=anchor_positions,
            anchor_ids=anchor_ids,
            distances_measured=list(measurements),
            distances_true=distances_true,
            measurement_errors=measurement_errors,
            measurement_noise_stds=measurement_noise_stds,
            los_conditions=los_conditions,
            channel_data=channel_data,
            filter_outputs=filter_outputs,
            imu_acceleration=imu_acc,
            imu_angular_velocity=imu_gyro
        )
        
        # Add to buffer
        self.buffer.add(sample)
        
        return sample
    
    def get_latest_sample(self) -> Optional[DataSample]:
        """Get the most recent sample"""
        samples = self.buffer.get_latest(1)
        return samples[0] if samples else None
    
    def get_all_samples(self) -> List[DataSample]:
        """Get all collected samples"""
        return self.buffer.get_all()
    
    def get_synchronized_data(self, start_time: float = None, end_time: float = None):
        """
        Get synchronized data as a list of flat dictionaries.
        Can be easily converted to pandas DataFrame.
        """
        samples = self.buffer.get_all()
        
        if start_time is not None:
            samples = [s for s in samples if s.timestamp >= start_time]
        if end_time is not None:
            samples = [s for s in samples if s.timestamp <= end_time]
            
        return [s.to_flat_dict() for s in samples]
    
    def stream_to_callback(self, callback: Callable[[DataSample], None]) -> None:
        """Register a callback for streaming data to external systems"""
        self.buffer.register_callback(callback)
        
    def stop_streaming(self, callback: Callable[[DataSample], None]) -> None:
        """Stop streaming to a callback"""
        self.buffer.unregister_callback(callback)
