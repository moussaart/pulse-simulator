"""
GUI Panels Package: Contains UI panel components.
"""
from .control_panels import ControlPanelFactory
from .Plots import LocalizationErrorPlot
from .dockable_panel import DockablePanel, FloatingPanelWindow, PanelManager

__all__ = [
    'ControlPanelFactory',
    'LocalizationErrorPlot',
    'DockablePanel',
    'FloatingPanelWindow',
    'PanelManager',
]
