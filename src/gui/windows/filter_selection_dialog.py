from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                           QLabel, QCheckBox, QPushButton, QWidget,
                           QScrollArea, QFrame, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPainter, QColor
from src.gui.theme import COLORS, MODERN_STYLESHEET
from src.gui.widgets import ActionButton

class FilterCheckBox(QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text']};
                font-size: 11px;
                spacing: 5px;
                padding: 6px;
                border-radius: 3px;
                background-color: {COLORS['widget_bg']};
            }}
            QCheckBox:hover {{
                background-color: {COLORS['secondary']};
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 2px;
                border: 1px solid {COLORS['accent']};
                background-color: {COLORS['background']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent']};
                border: none;
                image: url(resources/icons/check.png);
            }}
        """)

# CompactButton replaced by ActionButton from src.gui.widgets

class FilterSelectionDialog(QDialog):
    def __init__(self, available_filters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filters")
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['background']};
            }}
        """)
        self.setModal(True)
        
        self.available_filters = available_filters
        
        # Compact size
        self.setFixedSize(300, 400)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header with counter
        header_layout = QHBoxLayout()
        title_label = QLabel("Select Filters")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text']};
                font-size: 13px;
                font-weight: bold;
                padding: 2px;
            }}
        """)
        self.counter_label = QLabel("0 selected")
        self.counter_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text']};
                font-size: 11px;
                padding: 2px 8px;
                background: {COLORS['widget_bg']};
                border: 1px solid {COLORS['accent']};
                border-radius: 2px;
            }}
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.counter_label)
        layout.addLayout(header_layout)

        # Quick actions
        quick_actions = QHBoxLayout()
        quick_actions.setSpacing(4)
        select_all = ActionButton("Select All", variant="secondary")
        clear_all = ActionButton("Clear", variant="secondary")
        los_aekf_lambda = ActionButton("LOS-AEKF λ Compare", variant="secondary")
        
        select_all.clicked.connect(self.select_all)
        clear_all.clicked.connect(self.clear_all)
        los_aekf_lambda.clicked.connect(self.select_los_aekf_lambda)
        
        quick_actions.addWidget(select_all)
        quick_actions.addWidget(clear_all)
        quick_actions.addWidget(los_aekf_lambda)
        quick_actions.addStretch()
        layout.addLayout(quick_actions)

        # Filter grid container with background
        grid_container = QFrame()
        grid_container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['widget_bg']};
                border-radius: 4px;
                padding: 2px;
            }}
        """)
        grid_layout = QVBoxLayout(grid_container)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setSpacing(2)

        # Filter grid
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['widget_bg']};
            }}
        """)
        self.grid_layout = QGridLayout(scroll_widget)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)

        # Create checkboxes in a grid
        self.checkboxes = {}
        for i, filter_config in enumerate(available_filters):
            checkbox = FilterCheckBox(filter_config['display_name'])
            checkbox.stateChanged.connect(self.update_counter)
            self.checkboxes[filter_config['name']] = checkbox
            row = i // 2
            col = i % 2
            self.grid_layout.addWidget(checkbox, row, col)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {COLORS['accent']};
                background-color: {COLORS['widget_bg']};
                border-radius: 4px;
            }}
            QScrollBar:vertical {{
                border: none;
                width: 6px;
                margin: 4px;
                background: {COLORS['background']};
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['accent']};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        layout.addWidget(scroll)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        cancel_btn = CompactButton("Cancel")
        ok_btn = CompactButton("Apply", is_primary=True)
        
        cancel_btn.clicked.connect(self.reject)
        ok_btn.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

        # Flag for LOS-AEKF lambda comparison mode
        self.is_los_aekf_lambda_mode = False

    def update_counter(self):
        """Update the selection counter"""
        count = sum(1 for cb in self.checkboxes.values() if cb.isChecked())
        self.counter_label.setText(f"{count} selected")

    def select_all(self):
        self.is_los_aekf_lambda_mode = False
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def clear_all(self):
        self.is_los_aekf_lambda_mode = False
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def select_los_aekf_lambda(self):
        """Select only LOS_AEKF for lambda comparison mode"""
        self.is_los_aekf_lambda_mode = True
        for name, checkbox in self.checkboxes.items():
            checkbox.setChecked(name == 'LOS_AEKF')

    def get_selected_filters(self):
        if self.is_los_aekf_lambda_mode:
            # Create multiple LOS_AEKF configurations with different lambda values
            lambda_values = [1,  10,  50, 100, 500, 1000 , 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
            los_aekf_filters = []
            
            # Find the base LOS_AEKF configuration
            base_config = None
            for filter_config in self.available_filters:
                if filter_config['name'] == 'LOS_AEKF':
                    base_config = filter_config
                    break
            
            if base_config:
                for lambda_val in lambda_values:
                    # Create a new configuration for each lambda value
                    new_config = base_config.copy()
                    new_config['display_name'] = f"LOS-AEKF (λ={lambda_val})"
                    new_config['name'] = f"LOS_AEKF_{lambda_val}"
                    new_config['color'] = self.get_color_for_lambda(lambda_val)
                    new_config['nlos_factor'] = lambda_val
                    los_aekf_filters.append(new_config)
                
            return los_aekf_filters
        else:
            return [
                filter_config for filter_config in self.available_filters
                if self.checkboxes[filter_config['name']].isChecked()
            ]

    def get_color_for_lambda(self, lambda_val):
        """Generate a unique color for each lambda value"""
        colors = {
            1: '#FF0000',      # Red
            10: '#00FF00',     # Green
            50: '#0000FF',     # Blue
            100: '#FF00FF',    # Magenta
            500: '#F6A8F6FF',  # Light Purple
            1000: '#00FFFF',   # Cyan
            5000: '#800080',   # Purple
            10000: '#008000',  # Dark Green
            50000: '#FFA500',  # Orange
            100000: '#000080', # Navy Blue
            500000: '#FBFF00F4', # Brown
            1000000: '#CAD55FFF', # Gray
            5000000: '#CD9898FF'  # Black
        }
        return colors.get(lambda_val, '#FFFFFF')  # Default to white if lambda not in dict 