"""
ActionButton - Themed QPushButton with variant support.
Replaces duplicated button styling across multiple window files.
"""
from PyQt5.QtWidgets import QPushButton
from src.gui.theme import COLORS


# Variant color presets
_VARIANTS = {
    'primary': {
        'bg': COLORS['primary'],
        'hover': COLORS['primary_hover'],
        'text': '#ffffff',
        'border': COLORS['primary'],
    },
    'secondary': {
        'bg': COLORS['secondary'],
        'hover': COLORS['secondary_hover'],
        'text': COLORS['text'],
        'border': COLORS['border'],
    },
    'success': {
        'bg': '#2da44e',
        'hover': '#238636',
        'text': '#ffffff',
        'border': '#2da44e',
    },
    'danger': {
        'bg': '#da3633',
        'hover': '#b62324',
        'text': '#ffffff',
        'border': '#da3633',
    },
}


class ActionButton(QPushButton):
    """
    A themed QPushButton supporting variant styles.
    
    Variants: 'primary', 'secondary', 'success', 'danger'.
    """

    def __init__(self, text, variant="secondary", parent=None):
        super().__init__(text, parent)
        self.set_variant(variant)

    def set_variant(self, variant):
        """Apply a variant style to the button."""
        v = _VARIANTS.get(variant, _VARIANTS['secondary'])
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {v['bg']};
                color: {v['text']};
                border: 1px solid {v['border']};
                padding: 6px 14px;
                border-radius: 3px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {v['hover']};
            }}
            QPushButton:disabled {{
                background-color: #2d2d30;
                color: #6e6e6e;
                border: 1px solid #2d2d30;
            }}
        """)
