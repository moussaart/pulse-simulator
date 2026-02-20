"""
Simulation Recorder Module
Handles snapshot-based recording of simulation states for timeline playback.
Optimized for memory efficiency using circular buffers and NumPy arrays.
"""
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


@dataclass
class SimulationSnapshot:
    """Represents a single point in simulation time"""
    timestamp: float
    tag_position: Tuple[float, float]
    estimated_position: Tuple[float, float]
    error: float
    anchor_states: List[Dict[str, Any]] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'tag_position': self.tag_position,
            'estimated_position': self.estimated_position,
            'error': self.error,
            'anchor_states': self.anchor_states,
            'measurements': self.measurements
        }


class SimulationRecorder:
    """
    Records simulation states for timeline playback.
    Uses circular buffer for memory-efficient storage in infinite mode.
    """
    
    def __init__(self, max_duration: float = 60.0, snapshot_interval: int = 5):
        """
        Initialize the simulation recorder.
        
        Args:
            max_duration: Maximum duration of history to keep (seconds).
                         For infinite simulations, older data is discarded.
            snapshot_interval: Record every Nth frame (1 = every frame, 5 = every 5th)
        """
        self.max_duration = max_duration
        self.snapshot_interval = snapshot_interval
        self.frame_count = 0
        
        # Use standard deque without maxlen for time-based pruning
        self.snapshots: deque = deque()
        
        # For efficient binary search on timestamps
        self._timestamps: deque = deque()
        
        # Recording state
        self.is_recording = True
        self.simulation_ended = False
        
    def record_snapshot(self, 
                       timestamp: float,
                       tag_position: Tuple[float, float],
                       estimated_position: Tuple[float, float],
                       error: float,
                       anchors: list = None,
                       channel_conditions = None,
                       measurements: list = None) -> bool:
        """
        Record a simulation snapshot.
        
        Args:
            timestamp: Current simulation time
            tag_position: (x, y) of true tag position
            estimated_position: (x, y) of estimated position
            error: Current localization error
            anchors: List of anchor objects (optional)
            channel_conditions: Channel conditions for LOS state (optional)
            measurements: List of measured distances corresponding to anchors (optional)
            
        Returns:
            True if snapshot was recorded, False if skipped due to interval
        """
        if not self.is_recording:
            return False
            
        self.frame_count += 1
        
        # Only record at specified interval
        if self.frame_count % self.snapshot_interval != 0:
            return False
        
        # Build anchor states and measurement dict
        anchor_states = []
        measurement_dict = {}
        
        if anchors:
            # Create mapping from anchor ID to measurement if provided
            # Assuming measurements list corresponds to anchors list order
            anchor_measurements = {}
            if measurements and len(measurements) == len(anchors):
                for i, meas in enumerate(measurements):
                    if anchors[i].id:
                        anchor_measurements[anchors[i].id] = meas

            for i, anchor in enumerate(anchors):
                anchor_state = {
                    'id': anchor.id,
                    'position': (anchor.position.x, anchor.position.y)
                }
                # Add LOS state if channel conditions available
                if channel_conditions and hasattr(anchor, 'position'):
                    try:
                        from src.core.uwb.uwb_devices import Position
                        tag_pos = Position(tag_position[0], tag_position[1])
                        is_los = channel_conditions.check_los_to_anchor(
                            anchor.position, tag_pos)
                        anchor_state['is_los'] = is_los
                    except:
                        anchor_state['is_los'] = True
                
                anchor_states.append(anchor_state)
                
                # Store measurement for this anchor if available
                if anchor.id in anchor_measurements:
                     measurement_dict[anchor.id] = anchor_measurements[anchor.id]
        
        # Create and store snapshot
        snapshot = SimulationSnapshot(
            timestamp=timestamp,
            tag_position=tag_position,
            estimated_position=estimated_position,
            error=error,
            anchor_states=anchor_states,
            measurements=measurement_dict
        )
        
        self.snapshots.append(snapshot)
        self._timestamps.append(timestamp)
        
        # Prune old data if duration limit is set
        self._prune_history()
        
        return True
    
    def get_snapshot_at_time(self, target_time: float) -> Optional[SimulationSnapshot]:
        """
        Get the snapshot closest to the target time.
        Uses NumPy searchsorted for efficient O(log n) lookup.
        
        Args:
            target_time: Time to retrieve snapshot for
            
        Returns:
            Closest snapshot or None if no snapshots exist
        """
        if not self.snapshots:
            return None
        
        # Vectorized binary search using NumPy
        import numpy as np
        timestamps = np.array(self._timestamps)
        
        if target_time <= timestamps[0]:
            return self.snapshots[0]
        if target_time >= timestamps[-1]:
            return self.snapshots[-1]
        
        # np.searchsorted finds insertion point in O(log n)
        idx = int(np.searchsorted(timestamps, target_time))
        
        # Check which neighbor is closer
        if idx > 0 and abs(timestamps[idx] - target_time) > abs(timestamps[idx-1] - target_time):
            idx -= 1
        
        return self.snapshots[idx]
    
    def get_snapshots_up_to_time(self, target_time: float, max_points: int = None) -> List[SimulationSnapshot]:
        """
        Get all snapshots up to and including target time.
        Useful for reconstructing trajectory history.
        
        Args:
            target_time: Maximum time to include
            max_points: If set, only return the last N snapshots ending at target_time
            
        Returns:
            List of snapshots from start to target_time
        """
        if not self.snapshots:
            return []
        
        result = []
        for snapshot in self.snapshots:
            if snapshot.timestamp <= target_time:
                result.append(snapshot)
            else:
                break
        
        if max_points is not None and len(result) > max_points:
            return result[-max_points:]
            
        return result
    
    def get_trajectory_up_to_time(self, target_time: float, max_points: int = None) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Get trajectory coordinates up to target time.
        
        Args:
            target_time: Time limit
            max_points: Max number of recent points to return
            
        Returns:
            (tag_x, tag_y, est_x, est_y) lists
        """
        snapshots = self.get_snapshots_up_to_time(target_time, max_points)
        
        tag_x = [s.tag_position[0] for s in snapshots]
        tag_y = [s.tag_position[1] for s in snapshots]
        est_x = [s.estimated_position[0] for s in snapshots]
        est_y = [s.estimated_position[1] for s in snapshots]
        
        return tag_x, tag_y, est_x, est_y
    
    @property
    def duration(self) -> float:
        """Get total recorded duration"""
        if not self._timestamps:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]
    
    @property
    def start_time(self) -> float:
        """Get start timestamp"""
        return self._timestamps[0] if self._timestamps else 0.0
    
    @property
    def end_time(self) -> float:
        """Get end timestamp"""
        return self._timestamps[-1] if self._timestamps else 0.0
    
    @property
    def snapshot_count(self) -> int:
        """Get number of stored snapshots"""
        return len(self.snapshots)
    
    def clear(self):
        """Clear all recorded data"""
        self.snapshots.clear()
        self._timestamps.clear()
        self.frame_count = 0
        self.simulation_ended = False
        self.is_recording = True
    
    def pause_recording(self):
        """Pause recording"""
        self.is_recording = False
    
    def resume_recording(self):
        """Resume recording"""
        self.is_recording = True
    
    def mark_simulation_ended(self):
        """Mark that simulation has ended (reached duration limit)"""
        self.simulation_ended = True
        self.is_recording = False
    def _prune_history(self):
        """Remove snapshots older than max_duration from current end time"""
        if self.max_duration is None or not self._timestamps:
            return
            
        current_time = self._timestamps[-1]
        cutoff_time = current_time - self.max_duration
        
        # Efficiently pop from left while timestamp < cutoff_time
        # We need at least one snapshot before cutoff to interpolate/render properly if needed?
        # Actually just keeping data within [current - duration, current] is standard.
        while self._timestamps and self._timestamps[0] < cutoff_time:
            self._timestamps.popleft()
            self.snapshots.popleft()
    
    def set_max_duration(self, duration: float):
        """
        Update max duration and prune buffer if needed.
        
        Args:
            duration: New max duration in seconds, or None for infinite
        """
        self.max_duration = duration
        if duration is not None:
            self._prune_history()
    
    def set_snapshot_interval(self, interval: int):
        """
        Update snapshot interval.
        
        Args:
            interval: New interval (1 = every frame)
        """
        self.snapshot_interval = max(1, interval)
