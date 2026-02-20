"""
Reusable widget library for the GUI.
Provides themed UI components to reduce duplication across windows and panels.
"""

from .group_box import ModernGroupBox
from .navigation import PlotGroupNavigation
from .sliders import TimeWindowSlider, LabeledSlider
from .buttons import ActionButton
from .plot_helpers import create_themed_plot
from .spin_box import create_themed_spinbox, create_labeled_spinbox
from .distance_plot_widget import DistancePlotWidget
from .cir_plot_widget import CIRPlotWidget
from .base_window import PersistentWindow

__all__ = [
    'ModernGroupBox',
    'PlotGroupNavigation',
    'TimeWindowSlider',
    'LabeledSlider',
    'ActionButton',
    'create_themed_plot',
    'create_themed_spinbox',
    'create_labeled_spinbox',
    'PersistentWindow',
    'DistancePlotWidget',
    'CIRPlotWidget',
]
