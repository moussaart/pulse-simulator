"""
ThemedSpinBox - Factory function for styled QDoubleSpinBox widgets.
Replaces duplicated spinbox creation + styling in NLOS windows and params windows.
"""
from PyQt5.QtWidgets import QDoubleSpinBox, QLabel
from src.gui.theme import COLORS


def create_themed_spinbox(min_val=0, max_val=100, step=1.0, default=0,
                          suffix="", decimals=2):
    """
    Create a themed QDoubleSpinBox with consistent dark styling.

    Args:
        min_val: Minimum value.
        max_val: Maximum value.
        step: Single step increment.
        default: Initial value.
        suffix: Suffix text (e.g. ' m', ' ns').
        decimals: Number of decimal places.

    Returns:
        A styled QDoubleSpinBox.
    """
    spin = QDoubleSpinBox()
    spin.setRange(min_val, max_val)
    spin.setSingleStep(step)
    spin.setValue(default)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(suffix)
    spin.setStyleSheet(f"""
        QDoubleSpinBox {{
            background-color: {COLORS['input_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 5px;
            color: {COLORS['text_bright']};
            min-width: 80px;
        }}
        QDoubleSpinBox:hover {{
            border-color: {COLORS['accent']};
        }}
        QDoubleSpinBox:focus {{
            border-color: {COLORS['accent']};
        }}
    """)
    return spin


def create_labeled_spinbox(label_text, min_val=0, max_val=100, step=1.0,
                           default=0, suffix="", decimals=2):
    """
    Create a themed label + spinbox pair.

    Returns:
        Tuple of (QLabel, QDoubleSpinBox).
    """
    label = QLabel(label_text)
    label.setStyleSheet(f"""
        QLabel {{
            color: {COLORS['text']};
            font-size: 12px;
            padding: 5px;
        }}
    """)
    spin = create_themed_spinbox(min_val, max_val, step, default, suffix, decimals)
    return label, spin
