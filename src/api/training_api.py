"""
Training Data API
Main facade for AI training data collection from the UWB simulator.
"""
from typing import List, Dict, Optional, Callable, Any
import numpy as np

from src.api.collectors.data_collector import DataCollector, DataSample, DataBuffer
from src.api.adapters.channel_adapter import ChannelDataAdapter
from src.api.adapters.filter_adapter import FilterDataAdapter
from src.api.adapters.geometry_adapter import GeometryDataAdapter
from src.api.export.data_exporter import DataExporter


class TrainingDataAPI:
    """
    Simple API for AI training data collection from UWB simulator.
    
    This is the main entry point for all data collection operations.
    Provides simple, direct access to simulation data for AI training.
    
    Usage:
        api = TrainingDataAPI()
        api.select_data(channel=True, filter_outputs=True, ground_truth=True)
        api.enable_collection()
        
        # ... run simulation ...
        
        api.export_to_file("training_data.npz")
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Initialize the Training Data API.
        
        Args:
            buffer_size: Maximum number of samples to keep in memory
        """
        # Core components
        self._collector = DataCollector(buffer_size=buffer_size)
        self._exporter = DataExporter()
        
        # Adapters
        self._channel_adapter = ChannelDataAdapter()
        self._filter_adapter = FilterDataAdapter()
        self._geometry_adapter = GeometryDataAdapter()
        
        # Connect adapters to collector
        self._collector.channel_adapter = self._channel_adapter
        self._collector.filter_adapter = self._filter_adapter
        self._collector.geometry_adapter = self._geometry_adapter
        
        # Streaming callbacks
        self._stream_callbacks: List[Callable[[DataSample], None]] = []
        
    # ==================== Configuration ====================
    
    def configure(self,
                  buffer_size: int = None,
                  **kwargs) -> 'TrainingDataAPI':
        """
        Configure API settings.
        
        Args:
            buffer_size: Maximum buffer size
            **kwargs: Additional collector configuration
            
        Returns:
            self for method chaining
        """
        if buffer_size is not None:
            self._collector.buffer = DataBuffer(max_size=buffer_size)
        self._collector.configure(**kwargs)
        return self
    
    def select_data(self,
                    em: bool = True,
                    channel: bool = True,
                    snr: bool = True,
                    filter_outputs: bool = True,
                    ground_truth: bool = True,
                    imu: bool = True) -> 'TrainingDataAPI':
        """
        Select which data categories to collect.
        
        Args:
            em: Collect EM signal data
            channel: Collect channel model parameters
            snr: Collect SNR data
            filter_outputs: Collect filter estimates and covariances
            ground_truth: Collect ground truth positions
            imu: Collect IMU data
            
        Returns:
            self for method chaining
        """
        self._collector.configure(
            em=em,
            channel=channel,
            snr=snr,
            filter_outputs=filter_outputs,
            ground_truth=ground_truth,
            imu=imu
        )
        return self
    
    # ==================== Collection Control ====================
    
    def enable_collection(self) -> 'TrainingDataAPI':
        """
        Enable data collection.
        
        Returns:
            self for method chaining
        """
        self._collector.start()
        return self
    
    def disable_collection(self) -> 'TrainingDataAPI':
        """
        Disable data collection.
        
        Returns:
            self for method chaining
        """
        self._collector.stop()
        return self
    
    def reset(self) -> 'TrainingDataAPI':
        """
        Reset collector and clear all data.
        
        Returns:
            self for method chaining
        """
        self._collector.reset()
        return self
    
    @property
    def is_collecting(self) -> bool:
        """Check if data collection is currently active"""
        return self._collector.is_collecting
    
    # ==================== Data Collection Hook ====================
    
    def collect_sample(self,
                       timestamp: float,
                       tag,
                       anchors: List,
                       measurements: List[float],
                       channel_conditions,
                       filter_state: Dict[str, Any] = None,
                       estimated_pos: tuple = None,
                       error: float = None,
                       algorithm_name: str = None) -> Optional[DataSample]:
        """
        Collect a data sample from current simulation state.
        Called by SimulationManager during each update.
        
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
        return self._collector.collect(
            timestamp=timestamp,
            tag=tag,
            anchors=anchors,
            measurements=measurements,
            channel_conditions=channel_conditions,
            filter_state=filter_state,
            estimated_pos=estimated_pos,
            error=error,
            algorithm_name=algorithm_name
        )
    
    # ==================== Data Access ====================
    
    def get_latest_sample(self) -> Optional[DataSample]:
        """Get the most recent collected sample"""
        return self._collector.get_latest_sample()
    
    def get_buffer(self) -> List[DataSample]:
        """Get all samples in the buffer"""
        return self._collector.get_all_samples()
    
    def get_sample_count(self) -> int:
        """Get number of collected samples"""
        return len(self._collector.buffer)
    
    def get_synchronized_data(self, 
                               start_time: float = None, 
                               end_time: float = None) -> List[dict]:
        """
        Get synchronized, time-stamped data as flat dictionaries.
        
        Can be easily converted to pandas DataFrame:
            df = pd.DataFrame(api.get_synchronized_data())
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of flat dictionaries, one per sample
        """
        return self._collector.get_synchronized_data(start_time, end_time)
    
    # ==================== Export ====================
    
    def export_to_file(self, path: str, format: str = "npz") -> None:
        """
        Export collected data to file.
        
        Args:
            path: Output file path
            format: Export format - 'json', 'csv', 'npz' (default)
        """
        samples = self._collector.get_all_samples()
        
        if format.lower() == 'json':
            self._exporter.to_json(samples, path)
        elif format.lower() == 'csv':
            self._exporter.to_csv(samples, path)
        elif format.lower() == 'npz':
            self._exporter.to_npz(samples, path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json', 'csv', or 'npz'")
    
    def export_for_pytorch(self, path: str) -> None:
        """
        Export data in PyTorch-ready format.
        
        Creates separate .npy files for features, anchor positions, and targets.
        
        Args:
            path: Directory path for output files
        """
        samples = self._collector.get_all_samples()
        self._exporter.to_pytorch_dataset(samples, path)
    
    def export_for_tensorflow(self, path: str) -> None:
        """
        Export data in TensorFlow TFRecord format.
        
        Args:
            path: Output file path (.tfrecord)
        """
        samples = self._collector.get_all_samples()
        self._exporter.to_tensorflow_tfrecord(samples, path)
    
    # ==================== Streaming ====================
    
    def stream_to_callback(self, callback: Callable[[DataSample], None]) -> None:
        """
        Register a callback for streaming data.
        Useful for online/reinforcement learning.
        
        Args:
            callback: Function that receives each new DataSample
        """
        self._collector.stream_to_callback(callback)
        self._stream_callbacks.append(callback)
    
    def stop_streaming(self, callback: Callable[[DataSample], None] = None) -> None:
        """
        Stop streaming to a callback or all callbacks.
        
        Args:
            callback: Specific callback to stop, or None to stop all
        """
        if callback is not None:
            self._collector.stop_streaming(callback)
            if callback in self._stream_callbacks:
                self._stream_callbacks.remove(callback)
        else:
            for cb in self._stream_callbacks:
                self._collector.stop_streaming(cb)
            self._stream_callbacks.clear()
    
    # ==================== Adapter Access ====================
    
    @property
    def channel_adapter(self) -> ChannelDataAdapter:
        """Access to channel data adapter for custom extraction"""
        return self._channel_adapter
    
    @property
    def filter_adapter(self) -> FilterDataAdapter:
        """Access to filter data adapter for custom filters"""
        return self._filter_adapter
    
    @property
    def geometry_adapter(self) -> GeometryDataAdapter:
        """Access to geometry data adapter"""
        return self._geometry_adapter
    
    # ==================== Convenience Methods ====================
    
    def get_channel_summary(self, channel_conditions, tag, anchors) -> List[dict]:
        """
        Get a summary of channel conditions for all anchors.
        
        Returns:
            List of channel data dictionaries per anchor
        """
        summaries = []
        for anchor in anchors:
            link_data = self._channel_adapter.extract_link_data(
                channel_conditions, anchor, tag)
            summaries.append(link_data.to_dict())
        return summaries
    
    def get_geometry_summary(self, tag, anchors, measurements=None) -> dict:
        """
        Get a summary of current geometry.
        
        Returns:
            Dictionary with geometry data
        """
        return self._geometry_adapter.get_geometry_summary(tag, anchors, measurements)
    
    def get_statistics(self) -> dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        samples = self._collector.get_all_samples()
        
        if not samples:
            return {'sample_count': 0}
        
        # Calculate basic statistics
        errors = []
        snr_values = []
        los_count = 0
        nlos_count = 0
        
        for s in samples:
            for fo in s.filter_outputs.values():
                errors.append(fo.estimation_error)
            for cd in s.channel_data:
                snr_values.append(cd.snr_db)
            los_count += sum(s.los_conditions)
            nlos_count += len(s.los_conditions) - sum(s.los_conditions)
        
        stats = {
            'sample_count': len(samples),
            'time_range': (samples[0].timestamp, samples[-1].timestamp),
            'duration': samples[-1].timestamp - samples[0].timestamp,
        }
        
        if errors:
            stats['mean_error'] = float(np.mean(errors))
            stats['std_error'] = float(np.std(errors))
            stats['max_error'] = float(np.max(errors))
        
        if snr_values:
            stats['mean_snr_db'] = float(np.mean(snr_values))
            stats['min_snr_db'] = float(np.min(snr_values))
        
        stats['los_percentage'] = los_count / (los_count + nlos_count) * 100 if (los_count + nlos_count) > 0 else 0
        
        return stats
