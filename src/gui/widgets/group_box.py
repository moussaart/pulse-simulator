"""
ModernGroupBox - Centralized themed QGroupBox widget.
Replaces duplicated local definitions in nlos_config_window.py and moving_nlos_window.py.
"""
from PyQt5.QtWidgets import QGroupBox
from src.gui.theme import COLORS


class ModernGroupBox(QGroupBox):
    """A dark-themed QGroupBox with accent-colored title and rounded borders."""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['panel_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 20px;
                padding: 15px 10px 10px 10px;
                font-weight: bold;
                color: {COLORS['text']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {COLORS['accent']};
                background-color: transparent;
            }}
        """)
