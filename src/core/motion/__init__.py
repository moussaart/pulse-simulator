"""
Motion package: Handles motion control and related functionality.

``Trajectory_interface`` is a Qt dialog, so it is imported lazily (PEP 562).
Importing it eagerly made ``from src.core.motion import MotionController`` pull in
PyQt5, which broke headless training runs on machines without a display. GUI code
is unaffected: ``from src.core.motion import TrajectoryInterface`` still resolves.
"""
from .Motion_controller import MotionController

__all__ = [
    'MotionController',
    'TrajectoryInterface'
]


def __getattr__(name):
    if name == 'TrajectoryInterface':
        from .Trajectory_interface import TrajectoryInterface
        return TrajectoryInterface
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
