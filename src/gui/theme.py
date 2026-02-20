"""
GUI Theme definitions
"""

COLORS = {
    'background': '#1e1e1e',
    'panel_bg': '#252526',
    'text': '#cccccc',
    'text_bright': '#ffffff',
    'text_dim': '#858585',
    'accent': '#007acc',
    'border': '#3e3e42',
    'input_bg': '#3c3c3c',
    'success': '#89d185',
    'warning': '#dcdcaa',
    'error': '#f48771',
    'primary': '#007acc',
    'primary_light': '#3399ff',
    'primary_hover': '#0062a3',
    'primary_pressed': '#004c80',
    'secondary': '#3e3e42',
    'secondary_hover': '#4e4e52',
    'widget_bg': '#252526',
}

MODERN_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
    color: #cccccc;
}
QWidget {
    color: #cccccc;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QFrame {
    border: none;
}
QLabel {
    color: #cccccc;
}
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    color: #cccccc;
    border: 1px solid #3e3e42;
    border-radius: 2px;
    padding: 4px;
    selection-background-color: #264f78;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #007acc;
}
QComboBox {
    background-color: #3c3c3c;
    color: #cccccc;
    border: 1px solid #3e3e42;
    border-radius: 2px;
    padding: 4px;
}
QComboBox:hover {
    border: 1px solid #505050;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #252526;
    color: #cccccc;
    selection-background-color: #2a2d2e;
    selection-color: #ffffff;
    border: 1px solid #454545;
}
QPushButton {
    background-color: #3e3e42;
    color: #ffffff;
    border: 1px solid #3e3e42;
    border-radius: 2px;
    padding: 6px 12px;
}
QPushButton:hover {
    background-color: #4e4e52;
}
QPushButton:pressed {
    background-color: #2d2d30;
}
QPushButton:checked {
    background-color: #007acc;
    border: 1px solid #007acc;
}
QPushButton:disabled {
    background-color: #2d2d30;
    color: #6e6e6e;
    border: 1px solid #2d2d30;
}
QCheckBox {
    color: #cccccc;
    spacing: 5px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    background-color: #3c3c3c;
    border: 1px solid #505050;
    border-radius: 2px;
}
QCheckBox::indicator:hover {
    border: 1px solid #007acc;
}
QCheckBox::indicator:checked {
    background-color: #007acc;
    border: 1px solid #007acc;
}
QSlider::groove:horizontal {
    border: 1px solid #3e3e42;
    height: 4px;
    background: #3c3c3c;
    margin: 2px 0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #007acc;
    border: 1px solid #007acc;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #0062a3;
}
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 12px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #424242;
    min-height: 20px;
    border-radius: 0px;
}
QScrollBar::handle:vertical:hover {
    background: #4f4f4f;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #2d2d2d;
    color: #969696;
    border: none;
    padding: 8px 16px;
    margin-right: 1px;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-top: 2px solid #007acc;
}
QTabBar::tab:hover {
    background-color: #333333;
}
QGroupBox {
    border: 1px solid #3e3e42;
    border-radius: 4px;
    margin-top: 20px;
    font-weight: bold;
    background-color: #252526;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 3px;
    color: #e7e7e7;
    background-color: transparent; 
}
QMenuBar {
    background-color: #1e1e1e;
    color: #cccccc;
    border-bottom: 1px solid #3e3e42;
}
QMenuBar::item:selected {
    background-color: #3c3c3c;
}
QMenu {
    background-color: #252526;
    color: #cccccc;
    border: 1px solid #454545;
}
QMenu::item:selected {
    background-color: #094771;
}
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #3e3e42;
    color: #cccccc;
    font-family: 'Consolas', 'Courier New', monospace;
}
QMessageBox {
    background-color: #252526;
    color: #cccccc;
}
QMessageBox QLabel {
    color: #cccccc;
}
QDialog {
    background-color: #252526;
    color: #cccccc;
}
QFileDialog {
    background-color: #252526;
    color: #cccccc;
}
QFileDialog QListView, QFileDialog QTreeView {
    background-color: #1e1e1e;
    color: #cccccc;
    border: 1px solid #3e3e42;
}
"""

# Specialized styles for specific components to be used in Python code
BUTTON_PRIMARY_STYLE = """
QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 2px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0062a3;
}
QPushButton:pressed {
    background-color: #004c80;
}
"""

BUTTON_SECONDARY_STYLE = """
QPushButton {
    background-color: #3e3e42;
    color: white;
    border: 1px solid #3e3e42;
    border-radius: 2px;
    padding: 6px 12px;
}
QPushButton:hover {
    background-color: #4e4e52;
}
QPushButton:checked {
    background-color: #007acc;
    border-color: #007acc;
}
"""

PANEL_STYLE = """
QGroupBox {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 2px;
    margin-top: 10px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #007acc;
    font-weight: bold;
    background-color: transparent;
}
"""
