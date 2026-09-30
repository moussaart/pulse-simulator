"""
Localization package: Implements localization algorithms and custom message boxes.

The Qt message box is imported lazily (PEP 562). It used to be imported eagerly,
which meant that merely importing an algorithm pulled PyQt5 into the process and
made a headless training run fail on a machine with no display. GUI code keeps
working: ``from src.core.localization import CustomMessageBox`` still resolves.
"""
from .Localization_alghorthime import LocalizationAlgorthimes
from .Alghortimes_doc import Alghortimes_doc

__all__ = [
    'LocalizationAlgorthimes',
    'CustomMessageBox',
    'Alghortimes_doc'
]


def __getattr__(name):
    if name == 'CustomMessageBox':
        from .Boxs import CustomMessageBox
        return CustomMessageBox
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
