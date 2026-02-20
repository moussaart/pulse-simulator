from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QGroupBox, QComboBox, QPushButton, QLineEdit,
                            QWidget, QFormLayout, QDoubleSpinBox, QColorDialog, QSlider, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from src.gui.widgets import ModernGroupBox, ActionButton, create_themed_spinbox

# ModernGroupBox is now imported from src.gui.widgets

class MovingNLOSWindow(QDialog):
    def __init__(self, parent=None, initial_pos=None, final_pos=None, loaded_configs=None):
        super().__init__(parent)
        self.setWindowTitle("Add Moving NLOS Zone")
        self.loaded_configs = loaded_configs or {}
        self.setMinimumWidth(500)
        
        self.initial_pos = initial_pos if initial_pos else (0.0, 0.0)
        self.final_pos = final_pos if final_pos else (5.0, 5.0)
        
        # Modern dark theme with gradients
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #1E1E1E, stop:1 #2C2C2C);
            }
            QLabel {
                color: #FFFFFF;
                font-size: 12px;
            }
            QDoubleSpinBox {
                background-color: #363636;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 5px 10px;
                color: #FFFFFF;
                min-width: 80px;
                min-height: 25px;
                selection-background-color: #404040;
            }
            QDoubleSpinBox:hover {
                border-color: #4A4A4A;
                background-color: #404040;
            }
            QDoubleSpinBox:focus {
                border-color: #2196F3;
                background-color: #404040;
            }
            QComboBox {
                background-color: #363636;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 5px 10px;
                color: #FFFFFF;
                min-height: 25px;
            }
            QComboBox:hover {
                border-color: #4A4A4A;
                background-color: #404040;
            }
            QPushButton {
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                color: white;
            }
        """)
        
        self.current_color = [255, 165, 0]  # Default Orange for moving zones
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("Configure Moving Obstacle")
        header.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #FF9800, stop:1 #F57C00);
            border-radius: 8px;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # 1. Trajectory Configuration
        trajectory_group = ModernGroupBox("Trajectory & Movement")
        trajectory_layout = QGridLayout(trajectory_group)
        trajectory_layout.setSpacing(10)

        # Start Position
        trajectory_layout.addWidget(QLabel("Start Position (m):"), 0, 0)
        self.start_x = self.create_spinbox(-20, 20, 0.1, self.initial_pos[0])
        self.start_y = self.create_spinbox(-20, 20, 0.1, self.initial_pos[1])
        trajectory_layout.addWidget(QLabel("X:"), 0, 1)
        trajectory_layout.addWidget(self.start_x, 0, 2)
        trajectory_layout.addWidget(QLabel("Y:"), 0, 3)
        trajectory_layout.addWidget(self.start_y, 0, 4)

        # End Position
        trajectory_layout.addWidget(QLabel("End Position (m):"), 1, 0)
        self.end_x = self.create_spinbox(-20, 20, 0.1, self.final_pos[0])
        self.end_y = self.create_spinbox(-20, 20, 0.1, self.final_pos[1])
        trajectory_layout.addWidget(QLabel("X:"), 1, 1)
        trajectory_layout.addWidget(self.end_x, 1, 2)
        trajectory_layout.addWidget(QLabel("Y:"), 1, 3)
        trajectory_layout.addWidget(self.end_y, 1, 4)
        
        # Speed & Rotation
        trajectory_layout.addWidget(QLabel("Speed (m/s):"), 2, 0)
        self.speed = self.create_spinbox(0.1, 10.0, 0.1, 1.0)
        trajectory_layout.addWidget(self.speed, 2, 1, 1, 2)
        
        trajectory_layout.addWidget(QLabel("Rotation (rad/s):"), 2, 3)
        self.rotation = self.create_spinbox(0.0, 6.28, 0.1, 0.5)
        trajectory_layout.addWidget(self.rotation, 2, 4)

        layout.addWidget(trajectory_group)

        # 2. Shape Configuration
        shape_group = ModernGroupBox("Shape & Dimensions")
        shape_layout = QFormLayout(shape_group)
        
        self.shape_type = QComboBox()
        self.shape_type.addItems(["Circle", "Square", "Rectangle", "Triangle", "Diamond", "Hexagon"])
        self.shape_type.currentTextChanged.connect(self.update_dimension_fields)
        shape_layout.addRow("Shape Type:", self.shape_type)
        
        # Dimensions
        self.size_spin = self.create_spinbox(0.1, 5.0, 0.1, 1.0)
        self.width_spin = self.create_spinbox(0.1, 5.0, 0.1, 1.0)
        self.height_spin = self.create_spinbox(0.1, 5.0, 0.1, 1.0)
        
        self.size_label = QLabel("Size (Radius/Side):")
        self.width_label = QLabel("Width:")
        self.height_label = QLabel("Height:")
        
        shape_layout.addRow(self.size_label, self.size_spin)
        shape_layout.addRow(self.width_label, self.width_spin)
        shape_layout.addRow(self.height_label, self.height_spin)
        
        layout.addWidget(shape_group)
        self.update_dimension_fields(self.shape_type.currentText())

        # 3. Channel Configuration - Grouped like NLOSConfigWindow
        
        # Add Preset Selection
        if self.loaded_configs:
            preset_group = ModernGroupBox("Load Preset Configuration")
            preset_layout = QHBoxLayout(preset_group)
            
            self.config_combo = QComboBox()
            self.config_combo.addItems(["Select a preset..."] + sorted(list(self.loaded_configs.keys())))
            self.config_combo.currentTextChanged.connect(self.load_selected_config)
            
            preset_layout.addWidget(QLabel("Environment:"))
            preset_layout.addWidget(self.config_combo)
            layout.addWidget(preset_group)

        # 3a. Signal Propagation (Path Loss)
        link_group = ModernGroupBox("Signal Propagation (Path Loss)")
        link_layout = QGridLayout(link_group)
        link_layout.setSpacing(10)
        
        self.pl_exp = self.create_spinbox(0.0, 10.0, 0.1, 2.5)
        self.shadow_std = self.create_spinbox(0.0, 20.0, 0.1, 4.0)
        self.freq_decay = self.create_spinbox(0.0, 5.0, 0.1, 1.0)
        
        link_layout.addWidget(QLabel("Path Loss (n):"), 0, 0)
        link_layout.addWidget(self.pl_exp, 0, 1)
        link_layout.addWidget(QLabel("Shadowing (dB):"), 0, 2)
        link_layout.addWidget(self.shadow_std, 0, 3)
        link_layout.addWidget(QLabel("Freq Decay (κ):"), 1, 0)
        link_layout.addWidget(self.freq_decay, 1, 1)
        
        layout.addWidget(link_group)

        # 3b. Multipath (S-V Model)
        mp_group = ModernGroupBox("Multipath (S-V Model)")
        mp_layout = QGridLayout(mp_group)
        mp_layout.setSpacing(10)
        
        self.cluster_decay = self.create_spinbox(0.1, 100.0, 0.5, 5.5)  # ns
        self.ray_decay = self.create_spinbox(0.1, 100.0, 0.5, 6.7)      # ns
        self.rms_delay_spread = self.create_spinbox(1.0, 100.0, 1.0, 10.0) # ns
        
        mp_layout.addWidget(QLabel("Cluster Decay Γ (ns):"), 0, 0)
        mp_layout.addWidget(self.cluster_decay, 0, 1)
        mp_layout.addWidget(QLabel("Ray Decay γ (ns):"), 0, 2)
        mp_layout.addWidget(self.ray_decay, 0, 3)
        mp_layout.addWidget(QLabel("RMS Delay (ns):"), 1, 0)
        mp_layout.addWidget(self.rms_delay_spread, 1, 1)
        
        layout.addWidget(mp_group)

        # 3c. Error Model (NLOS Effects)
        err_group = ModernGroupBox("Error Model (NLOS Effects)")
        err_layout = QGridLayout(err_group)
        err_layout.setSpacing(10)
        
        self.noise_factor = self.create_spinbox(1.0, 20.0, 0.1, 1.5)
        self.error_bias = self.create_spinbox(0.0, 5.0, 0.01, 0.05)
        
        err_layout.addWidget(QLabel("Noise Factor:"), 0, 0)
        err_layout.addWidget(self.noise_factor, 0, 1)
        err_layout.addWidget(QLabel("Bias (m):"), 0, 2)
        err_layout.addWidget(self.error_bias, 0, 3)
        
        # Color Picker inside Error Model group or separate? Let's verify existing
        # Existing had a separate color picker row. Let's keep it near the bottom or in a "Visuals" group
        # but to save space, maybe just put it in the Error Model group or a separate small HBox
        
        # Color Picker row
        color_layout = QHBoxLayout()
        self.color_preview = QPushButton()
        self.color_preview.setFixedSize(40, 40)
        self.color_preview.clicked.connect(self.pick_color)
        self.update_color_preview()
        
        color_layout.addWidget(QLabel("Zone Color:"))
        color_layout.addWidget(self.color_preview)
        color_layout.addStretch()
        
        err_layout.addLayout(color_layout, 1, 0, 1, 4)
        
        layout.addWidget(err_group)

        # Buttons
        button_layout = QHBoxLayout()
        add_btn = ActionButton("Add Obstacle", variant="success")
        add_btn.clicked.connect(self.accept)
        
        cancel_btn = ActionButton("Cancel", variant="danger")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(add_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)

    def create_spinbox(self, min_val, max_val, step, val):
        """Legacy wrapper — delegates to create_themed_spinbox."""
        return create_themed_spinbox(min_val, max_val, step, val)

    def load_selected_config(self, config_name):
        """Load parameters from selected configuration"""
        if not config_name or config_name == "Select a preset..." or config_name not in self.loaded_configs:
            return
            
        env_config = self.loaded_configs[config_name]
        # Use NLOS params for the "NLOS case"
        params = env_config.nlos_params
        
        # Update GUI fields
        self.pl_exp.setValue(params.path_loss_exponent)
        self.shadow_std.setValue(params.shadow_fading_std)
        # freq_decay not in SVModelParams
        
        # Update S-V params (convert s to ns)
        self.cluster_decay.setValue(params.cluster_decay * 1e9)
        self.ray_decay.setValue(params.ray_decay * 1e9)
        self.rms_delay_spread.setValue(params.rms_delay_spread * 1e9)

    def update_dimension_fields(self, shape):
        is_rect = shape == "Rectangle"
        
        self.size_label.setVisible(not is_rect)
        self.size_spin.setVisible(not is_rect)
        
        self.width_label.setVisible(is_rect)
        self.width_spin.setVisible(is_rect)
        self.height_label.setVisible(is_rect)
        self.height_spin.setVisible(is_rect)

    def pick_color(self):
        color = QColorDialog.getColor(QColor(*self.current_color), self, "Select Color")
        if color.isValid():
            self.current_color = [color.red(), color.green(), color.blue()]
            self.update_color_preview()

    def update_color_preview(self):
        c = self.current_color
        self.color_preview.setStyleSheet(f"""
            background-color: rgb({c[0]}, {c[1]}, {c[2]});
            border: 2px solid #555;
            border-radius: 6px;
        """)

    def get_data(self):
        return {
            'initial_pos': (self.start_x.value(), self.start_y.value()),
            'final_pos': (self.end_x.value(), self.end_y.value()),
            'shape_type': self.shape_type.currentText(),
            'speed': self.speed.value(),
            'rotation_speed': self.rotation.value(),
            'size': self.size_spin.value(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            # Full Channel Params
            'path_loss_exp': self.pl_exp.value(),
            'shadow_std': self.shadow_std.value(),
            'freq_decay': self.freq_decay.value(),
            'cluster_decay': self.cluster_decay.value() * 1e-9, # convert ns to s
            'ray_decay': self.ray_decay.value() * 1e-9,       # convert ns to s
            'rms_delay': self.rms_delay_spread.value() * 1e-9,# convert ns to s
            'error_bias': self.error_bias.value(),
            'noise_factor': self.noise_factor.value(),
            'color': self.current_color
        }
