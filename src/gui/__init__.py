"""
GUI package: Contains all graphical user interface components organized in submodules.

Submodules:
- windows: Window and dialog components
- managers: Manager classes for different aspects (events, plot, simulation, etc.)
- panels: UI panel components
"""

# Import from windows submodule
from .windows import (
    DistancePlotsWindow,
    IMUWindow,
    IMUData,
    NLOSConfigWindow,
    NLOSConfigManager,
)

# Import from panels submodule
from .panels import (
    LocalizationErrorPlot,
    ControlPanelFactory,
)

# Import from managers submodule
from .managers import (
    EventHandler,
    TrajectoryManager,
    FileManager,
    NLOSManager,
    PlotManager,
    SimulationManager,
)

__all__ = [
    # Windows
    'DistancePlotsWindow',
    'IMUWindow',
    'IMUData',
    'NLOSConfigWindow',
    'NLOSConfigManager',
    # Panels
    'LocalizationErrorPlot',
    'ControlPanelFactory',
    # Managers
    'EventHandler',
    'TrajectoryManager',
    'FileManager',
    'NLOSManager',
    'PlotManager',
    'SimulationManager',
]