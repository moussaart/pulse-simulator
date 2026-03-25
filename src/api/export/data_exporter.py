"""
Data Exporter
Export collected data to various formats for AI training.
"""
import json
import os
from typing import List, Generator, Callable, Any
from pathlib import Path
import numpy as np

from src.api.collectors.data_collector import DataSample, DataBuffer


class DataExporter:
    """
    Export collected data to various formats.
    Supports JSON, CSV, NPZ, and AI framework specific formats.
    """
    
    def to_json(self, samples: List[DataSample], path: str) -> None:
        """
        Export samples to JSON format.
        
        Args:
            samples: List of DataSample objects
            path: Output file path
        """
        data = [s.to_dict() for s in samples]
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
            
        print(f"Exported {len(samples)} samples to {path}")
    
    def to_csv(self, samples: List[DataSample], path: str) -> None:
        """
        Export samples to CSV format (flattened).
        
        Args:
            samples: List of DataSample objects
            path: Output file path
        """
        if not samples:
            print("No samples to export")
            return
            
        # Convert to flat dictionaries
        flat_data = [s.to_flat_dict() for s in samples]
        
        # Get all keys
        all_keys = set()
        for d in flat_data:
            all_keys.update(d.keys())
        all_keys = sorted(all_keys)
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            # Write header
            f.write(','.join(all_keys) + '\n')
            
            # Write data rows
            for d in flat_data:
                row = []
                for key in all_keys:
                    val = d.get(key, '')
                    if val is None:
                        val = ''
                    elif isinstance(val, bool):
                        val = '1' if val else '0'
                    row.append(str(val))
                f.write(','.join(row) + '\n')
                
        print(f"Exported {len(samples)} samples to {path}")
    
    def to_npz(self, samples: List[DataSample], path: str) -> None:
        """
        Export samples to NumPy NPZ format.
        Efficient format for ML training.
        
        Args:
            samples: List of DataSample objects
            path: Output file path
        """
        if not samples:
            print("No samples to export")
            return
            
        # Prepare arrays
        timestamps = np.array([s.timestamp for s in samples])
        
        # Tag positions (ground truth)
        tag_positions = np.array([s.tag_position_gt for s in samples])
        
        # Get consistent anchor count
        n_anchors = max(len(s.anchor_ids) for s in samples)
        
        # Anchor positions
        anchor_positions = np.zeros((len(samples), n_anchors, 3))
        for i, s in enumerate(samples):
            for j, pos in enumerate(s.anchor_positions):
                if j < n_anchors:
                    anchor_positions[i, j] = pos
        
        # Measurements
        distances_measured = np.zeros((len(samples), n_anchors))
        distances_true = np.zeros((len(samples), n_anchors))
        los_conditions = np.zeros((len(samples), n_anchors), dtype=bool)
        
        for i, s in enumerate(samples):
            for j, dist in enumerate(s.distances_measured):
                if j < n_anchors:
                    distances_measured[i, j] = dist
            for j, dist in enumerate(s.distances_true):
                if j < n_anchors:
                    distances_true[i, j] = dist
            for j, los in enumerate(s.los_conditions):
                if j < n_anchors:
                    los_conditions[i, j] = los
        
        # Channel data (SNR, path loss per anchor)
        snr_db = np.zeros((len(samples), n_anchors))
        path_loss_db = np.zeros((len(samples), n_anchors))
        signal_quality = np.zeros((len(samples), n_anchors))
        
        for i, s in enumerate(samples):
            for j, cd in enumerate(s.channel_data):
                if j < n_anchors:
                    snr_db[i, j] = cd.snr_db
                    path_loss_db[i, j] = cd.path_loss_db
                    signal_quality[i, j] = cd.signal_quality
        
        # Filter outputs (collect from all samples)
        filter_names = set()
        for s in samples:
            filter_names.update(s.filter_outputs.keys())
        
        filter_data = {}
        for fname in filter_names:
            safe_name = fname.replace(' ', '_').lower()
            est_positions = []
            errors = []
            state_covs = []
            
            for s in samples:
                if fname in s.filter_outputs:
                    fo = s.filter_outputs[fname]
                    est_positions.append(fo.estimated_position)
                    errors.append(fo.estimation_error)
                    if fo.state_covariance is not None:
                        state_covs.append(fo.state_covariance)
                else:
                    est_positions.append((np.nan, np.nan))
                    errors.append(np.nan)
            
            filter_data[f'{safe_name}_estimated_positions'] = np.array(est_positions)
            filter_data[f'{safe_name}_errors'] = np.array(errors)
            if state_covs:
                filter_data[f'{safe_name}_state_covariances'] = np.array(state_covs)
        
        # IMU data
        imu_acc = np.array([
            s.imu_acceleration if s.imu_acceleration else (np.nan, np.nan, np.nan)
            for s in samples
        ])
        imu_gyro = np.array([
            s.imu_angular_velocity if s.imu_angular_velocity else (np.nan, np.nan, np.nan)
            for s in samples
        ])
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save all arrays
        np.savez(
            path,
            timestamps=timestamps,
            tag_positions=tag_positions,
            anchor_positions=anchor_positions,
            distances_measured=distances_measured,
            distances_true=distances_true,
            los_conditions=los_conditions,
            snr_db=snr_db,
            path_loss_db=path_loss_db,
            signal_quality=signal_quality,
            imu_acceleration=imu_acc,
            imu_angular_velocity=imu_gyro,
            **filter_data
        )
        
        print(f"Exported {len(samples)} samples to {path}")
    
    def to_pytorch_dataset(self, samples: List[DataSample], path: str) -> None:
        """
        Export samples in a format ready for PyTorch DataLoader.
        Creates separate .npy files for inputs and targets.
        
        Args:
            samples: List of DataSample objects
            path: Directory path for output files
        """
        if not samples:
            print("No samples to export")
            return
            
        # Create output directory
        Path(path).mkdir(parents=True, exist_ok=True)
        
        n_samples = len(samples)
        n_anchors = max(len(s.anchor_ids) for s in samples)
        
        # Input features: distances, LOS conditions, SNR, signal quality
        # Shape: (n_samples, n_anchors, n_features)
        n_features = 4  # distance, is_los, snr, signal_quality
        X = np.zeros((n_samples, n_anchors, n_features))
        
        for i, s in enumerate(samples):
            for j in range(min(len(s.distances_measured), n_anchors)):
                X[i, j, 0] = s.distances_measured[j]
                X[i, j, 1] = 1.0 if s.los_conditions[j] else 0.0
                if j < len(s.channel_data):
                    X[i, j, 2] = s.channel_data[j].snr_db
                    X[i, j, 3] = s.channel_data[j].signal_quality
        
        # Anchor positions as additional input
        anchor_pos = np.zeros((n_samples, n_anchors, 3))
        for i, s in enumerate(samples):
            for j, pos in enumerate(s.anchor_positions):
                if j < n_anchors:
                    anchor_pos[i, j] = pos
        
        # Target: ground truth tag position
        y = np.array([s.tag_position_gt[:2] for s in samples])  # Only x, y
        
        # Save arrays
        np.save(os.path.join(path, 'features.npy'), X)
        np.save(os.path.join(path, 'anchor_positions.npy'), anchor_pos)
        np.save(os.path.join(path, 'targets.npy'), y)
        
        # Save metadata
        metadata = {
            'n_samples': n_samples,
            'n_anchors': n_anchors,
            'n_features': n_features,
            'feature_names': ['distance', 'is_los', 'snr_db', 'signal_quality']
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Exported PyTorch dataset to {path}")
    
    def create_streaming_generator(self, 
                                    buffer: DataBuffer) -> Generator[DataSample, None, None]:
        """
        Create a generator that yields samples as they are added to the buffer.
        Useful for online learning.
        
        Args:
            buffer: DataBuffer instance
            
        Yields:
            DataSample objects as they are collected
        """
        last_idx = 0
        while True:
            current_len = len(buffer)
            if current_len > last_idx:
                samples = buffer.get_all()[last_idx:current_len]
                for sample in samples:
                    yield sample
                last_idx = current_len
    
    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
