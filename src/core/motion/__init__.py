"""
Motion package: Handles motion control and related functionality.
"""
from .Motion_controller import MotionController
from .Trajectory_interface import TrajectoryInterface

__all__ = [
    'MotionController',
    'TrajectoryInterface'
]