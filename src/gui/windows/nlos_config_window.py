from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QGroupBox, QComboBox, QPushButton, QLineEdit,
                            QWidget, QFormLayout, QDoubleSpinBox, QColorDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import json
import os
import random
import colorsys
from src.gui.theme import COLORS, MODERN_STYLESHEET
from src.gui.widgets import ModernGroupBox, ActionButton, create_themed_spinbox
from src.utils.resource_loader import get_data_path






class NLOSConfigManager:
    """Manages saved NLOS configurations and their associated colors"""
    def __init__(self):
        self.config_file = get_data_path("data/configs/nlos_configs.json")
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        self.saved_configs = self.load_configs()
        self.config_colors = {}  # Store colors for configurations

    def load_configs(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    # Convert stored configs to new format if needed
                    if isinstance(data, dict):
                        configs = {}
                        for name, config in data.items():
                            if isinstance(config, dict) and 'color' not in config:
                                color = self.generate_random_color()
                                configs[name] = {
                                    'parameters': config,
                                    'color': color
                                }
                            else:
                                configs[name] = config
                        return configs
            except:
                return {}
        return {}

    def save_configs(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.saved_configs, f, indent=4)

    def save_config(self, name, parameters, color=None):
        if color is None:
            color = self.generate_random_color()
        
        self.saved_configs[name] = {
            'parameters': parameters,
            'color': color
        }
        self.save_configs()

    def get_config(self, name):
        config_data = self.saved_configs.get(name)
        if config_data:
            return config_data['parameters'], config_data['color']
        return None, None

    def get_config_names(self):
        return list(self.saved_configs.keys())

    def generate_random_color(self):
        """Generate a random color that's visually distinct"""
        hue = random.random()
        saturation = 0.7 + random.random() * 0.3  # 0.7-1.0
        value = 0.5 + random.random() * 0.3  # 0.5-0.8
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return [int(x * 255) for x in rgb]

    def get_color_for_parameters(self, parameters):
        """Get color for a set of parameters, reusing existing colors for matching configs"""
        for name, config in self.saved_configs.items():
            stored_params = config['parameters']
            if (abs(stored_params['path_loss_exponent'] - parameters['path_loss_exponent']) < 0.01 and
                abs(stored_params['shadow_fading_std'] - parameters['shadow_fading_std']) < 0.01 and
                abs(stored_params['error_bias'] - parameters['error_bias']) < 0.01 and
                abs(stored_params['noise_factor'] - parameters['noise_factor']) < 0.01):
                return config['color']
        return None

    def delete_config(self, name):
        """Delete a named configuration"""
        if name in self.saved_configs:
            del self.saved_configs[name]
            self.save_configs()
            return True
        return False

    def get_config_dir(self):
        """Get the directory where configs are stored"""
        return os.path.dirname(self.config_file)



# ModernGroupBox is now imported from src.gui.widgets

class NLOSConfigWindow(QDialog):
    def __init__(self, zones, config_manager, current_color=None, parent=None, loaded_configs=None):
        super().__init__(parent)
        
        # Handle both single zone and list of zones
        if isinstance(zones, list):
            self.zones = zones
            self.zone = zones[0] if zones else None
            title = f"Edit {len(zones)} NLOS Zones"
        else:
            self.zones = [zones]
            self.zone = zones
            title = "Edit NLOS Zone Parameters"
            
        self.setWindowTitle(title)
        self.config_manager = config_manager
        self.loaded_configs = loaded_configs or {}
        self.setMinimumWidth(450)
        
        # Modern dark theme with gradients
        # Modern dark theme with gradients - utilizing global theme
        self.setStyleSheet(MODERN_STYLESHEET + f"""
            QDialog {{
                background-color: {COLORS['background']};
            }}
            QPushButton#colorButton {{
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
            }}
            QPushButton#colorButton:hover {{
                border-color: {COLORS['text_dim']};
            }}
        """)
        
        # Initialize color
        self.current_color = current_color if current_color else [33, 150, 243]  # Material Blue
        
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Header
        header = QLabel("NLOS Zone Parameters")
        header.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #2196F3, stop:1 #1976D2);
            border-radius: 8px;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # 1. Signal Propagation (Path Loss)
        link_group = ModernGroupBox("Signal Propagation (Path Loss)")
        link_layout = QFormLayout(link_group)
        link_layout.setSpacing(10)
        
        self.pl_exp = create_themed_spinbox(0.0, 10.0, 0.1, self.zone.path_loss_params.path_loss_exponent)
        self.shadow_std = create_themed_spinbox(0.0, 20.0, 0.1, self.zone.path_loss_params.shadow_fading_std)
        self.freq_decay = create_themed_spinbox(0.0, 5.0, 0.1, getattr(self.zone.path_loss_params, 'frequency_decay_factor', 1.0))
        
        self.add_parameter_row(link_layout, "Path Loss (n):", self.pl_exp, "Free Space = 2.0")
        self.add_parameter_row(link_layout, "Shadowing (dB):", self.shadow_std, "Std Dev σ")
        self.add_parameter_row(link_layout, "Freq Decay (κ):", self.freq_decay, "1.0 = Linear")
        
        layout.addWidget(link_group)

        # 2. Multipath (S-V Model)
        mp_group = ModernGroupBox("Multipath (S-V Model)")
        mp_layout = QFormLayout(mp_group)
        mp_layout.setSpacing(10)
        
        # Get S-V params
        sv_params = getattr(self.zone, 'sv_params', None)
        cluster_decay_ns = sv_params.cluster_decay * 1e9 if sv_params else 7.1
        ray_decay_ns = sv_params.ray_decay * 1e9 if sv_params else 4.3
        rms_delay_ns = getattr(self.zone, 'rms_delay_spread', 15e-9) * 1e9
        
        self.cluster_decay = create_themed_spinbox(0.1, 100.0, 0.5, cluster_decay_ns)
        self.ray_decay = create_themed_spinbox(0.1, 100.0, 0.5, ray_decay_ns)
        self.rms_delay_spread = create_themed_spinbox(1.0, 100.0, 1.0, rms_delay_ns)
        
        self.add_parameter_row(mp_layout, "Cluster Decay Γ (ns):", self.cluster_decay, "Time constant")
        self.add_parameter_row(mp_layout, "Ray Decay γ (ns):", self.ray_decay, "Time constant")
        self.add_parameter_row(mp_layout, "RMS Delay (ns):", self.rms_delay_spread, "Delay Spread")
        
        layout.addWidget(mp_group)

        # 3. Error Model (NLOS Effects)
        err_group = ModernGroupBox("Error Model (NLOS Effects)")
        err_layout = QFormLayout(err_group)
        err_layout.setSpacing(10)
        
        self.noise_factor = create_themed_spinbox(1.0, 20.0, 0.1, self.zone.noise_factor)
        self.error_bias = create_themed_spinbox(0.0, 5.0, 0.01, self.zone.error_bias)
        
        self.add_parameter_row(err_layout, "Noise Factor:", self.noise_factor, "Multiplier for σ")
        self.add_parameter_row(err_layout, "Bias (m):", self.error_bias, "Added distance")
        
        layout.addWidget(err_group)

        # Configuration section
        config_group = ModernGroupBox("Preset Configurations")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(15)
        
        # Add color picker and preset selector in a horizontal layout
        controls_layout = QHBoxLayout()
        
        # Color picker section
        color_layout = QVBoxLayout()
        color_label = QLabel("🎨 Zone Color")
        color_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        
        self.color_preview = QPushButton()
        self.color_preview.setObjectName("colorButton")
        self.color_preview.setFixedSize(60, 60)
        self.color_preview.clicked.connect(self.pick_color)
        self.update_color_preview(self.current_color)
        
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_preview)
        controls_layout.addLayout(color_layout)
        
        # Preset selector section
        preset_layout = QVBoxLayout()
        preset_label = QLabel("📋 Load Preset")
        preset_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        
        self.config_combo = QComboBox()
        
        # Combine saved configs and loaded CSV configs
        saved_names = self.config_manager.get_config_names() if self.config_manager else []
        loaded_names = list(self.loaded_configs.keys())
        all_names = sorted(list(set(saved_names + loaded_names)))
        
        self.config_combo.addItems(["Select a preset..."] + all_names)
        self.config_combo.currentTextChanged.connect(self.load_selected_config)
        
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.config_combo)
        controls_layout.addLayout(preset_layout)
        
        config_layout.addLayout(controls_layout)
        
        # Save preset button
        action_layout = QVBoxLayout()
        action_layout.setSpacing(8)
        
        # Row 1: Save (Main Action)
        save_button = ActionButton("💾 Save as New Preset", variant="primary")
        save_button.clicked.connect(self.save_current_config)
        action_layout.addWidget(save_button)
        
        # Row 2: Manage (Open Folder, Delete)
        manage_row = QHBoxLayout()
        manage_row.setSpacing(8)
        
        open_folder_button = ActionButton("📂 Open Folder", variant="secondary")
        open_folder_button.clicked.connect(self.open_config_folder)
        open_folder_button.setMinimumWidth(100)
        
        delete_button = ActionButton("🗑️ Delete Preset", variant="danger")
        delete_button.clicked.connect(self.delete_current_config)
        delete_button.setMinimumWidth(100)
        
        manage_row.addWidget(open_folder_button, 1) # Full width stretch
        manage_row.addWidget(delete_button, 1) # Full width stretch
        action_layout.addLayout(manage_row)
        
        config_layout.addLayout(action_layout)
        
        layout.addWidget(config_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        apply_button = ActionButton("Apply Changes", variant="success")
        cancel_button = ActionButton("Cancel", variant="danger")
        
        apply_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)

    def create_modern_spinbox(self, min_val, max_val, step, value):
        """Legacy wrapper - delegates to create_themed_spinbox."""
        return create_themed_spinbox(min_val, max_val, step, value)

    def add_parameter_row(self, layout, label_text, widget, range_text):
        label = QLabel(label_text)
        range_label = QLabel(range_text)
        range_label.setStyleSheet("color: #888888; font-size: 11px;")
        
        widget_layout = QHBoxLayout()
        widget_layout.addWidget(widget)
        widget_layout.addWidget(range_label)
        widget_layout.addStretch()
        
        layout.addRow(label, widget_layout)

    def update_color_preview(self, color):
        self.current_color = color
        style = f"""
            QPushButton {{
                background-color: rgb({color[0]}, {color[1]}, {color[2]});
                border: 2px solid #555555;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                border: 2px solid #666666;
            }}
        """
        self.color_preview.setStyleSheet(style)

    def pick_color(self):
        current = QColor(*self.current_color)
        color = QColorDialog.getColor(current, self, "Select Zone Color")
        if color.isValid():
            self.current_color = [color.red(), color.green(), color.blue()]
            self.update_color_preview(self.current_color)

    def save_current_config(self):
        if not self.config_manager:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Save Configuration")
        dialog.setStyleSheet("""
            QDialog { 
                background-color: #2C2C2C;
                border-radius: 8px;
            }
            QLabel { 
                color: #FFFFFF;
                font-size: 12px;
                padding: 5px;
            }
            QLineEdit {
                background-color: #363636;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px;
                color: #FFFFFF;
                min-width: 250px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #888888;
                background-color: #404040;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add icon and label with better styling
        label = QLabel("💼 Enter a name for this configuration preset:")
        label.setStyleSheet("""
            font-weight: bold;
            color: #E0E0E0;
            font-size: 13px;
        """)
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Indoor Office, Concrete Wall, etc.")

        # Add color preview in save dialog
        color_preview = QWidget()
        color_preview.setStyleSheet(f"""
            background-color: rgb({self.current_color[0]}, {self.current_color[1]}, {self.current_color[2]});
            border: 2px solid #555555;
            border-radius: 4px;
            min-height: 30px;
        """)

        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")

        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)

        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #c62828;
            }
        """)

        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)

        layout.addWidget(label)
        layout.addWidget(name_input)
        layout.addWidget(QLabel("Selected Color:"))
        layout.addWidget(color_preview)
        layout.addLayout(button_layout)

        save_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted and name_input.text():
            name = name_input.text()
            config = {
                "path_loss_exponent": self.pl_exp.value(),
                "shadow_fading_std": self.shadow_std.value(),
                "frequency_decay_factor": self.freq_decay.value(),
                # NLOS Params
                "error_bias": self.error_bias.value(),
                "noise_factor": self.noise_factor.value(),
                # IEEE 802.15.3a S-V model parameters
                "rms_delay_spread_ns": self.rms_delay_spread.value(),
                "cluster_decay_ns": self.cluster_decay.value(),
                "ray_decay_ns": self.ray_decay.value()
            }
            
            self.config_manager.save_config(name, config, self.current_color)
            
            if name not in [self.config_combo.itemText(i) for i in range(self.config_combo.count())]:
                self.config_combo.addItem(name)

    def delete_current_config(self, display_success=True):
        """Delete the currently selected configuration"""
        current_name = self.config_combo.currentText()
        if not current_name or current_name == "Select a preset...":
            return
            
        if current_name in self.loaded_configs:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Cannot Delete", 
                              f"'{current_name}' is a built-in configuration (loaded from CSV) and cannot be deleted here.")
            return

        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, "Confirm Deletion", 
                                   f"Are you sure you want to delete the preset '{current_name}'?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                   
        if reply == QMessageBox.Yes:
            if self.config_manager.delete_config(current_name):
                # Remove from combo box
                index = self.config_combo.findText(current_name)
                if index >= 0:
                    self.config_combo.removeItem(index)
                
                # Reset selection
                self.config_combo.setCurrentIndex(0)
                
                if display_success:
                    QMessageBox.information(self, "Success", f"Configuration '{current_name}' deleted.")

    def open_config_folder(self):
        """Open the folder containing configuration files"""
        if self.config_manager:
            folder = self.config_manager.get_config_dir()
            if os.path.exists(folder):
                os.startfile(folder)
            else:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", "Configuration folder does not exist.")

    def load_selected_config(self, config_name):
        if not config_name or config_name == "Select a preset...":
            return

        # 1. Check loaded CSV configs first
        if config_name in self.loaded_configs:
            env_config = self.loaded_configs[config_name]
            # Use NLOS params for the "NLOS case"
            params = env_config.nlos_params
            
            # Update GUI fields
            self.pl_exp.setValue(params.path_loss_exponent)
            self.shadow_std.setValue(params.shadow_fading_std)
            # freq_decay not in SVModelParams, keep current or default
            
            # Update S-V params (convert s to ns)
            self.cluster_decay.setValue(params.cluster_decay * 1e9)
            self.ray_decay.setValue(params.ray_decay * 1e9)
            self.rms_delay_spread.setValue(params.rms_delay_spread * 1e9)
            
            # Configs don't have bias/noise, so maybe leave them or set defaults?
            # User wants "nlos case", implying these channel parameters are key.
            return

        # 2. Check saved user configs
        if self.config_manager:
            config, color = self.config_manager.get_config(config_name)
            if config:
                self.pl_exp.setValue(config.get("path_loss_exponent", 2.5))
                self.shadow_std.setValue(config.get("shadow_fading_std", 4.0))
                self.freq_decay.setValue(config.get("frequency_decay_factor", 1.0))
                
                self.error_bias.setValue(config.get("error_bias", 0.05))
                self.noise_factor.setValue(config.get("noise_factor", 1.5))
                
                # Load S-V model parameters if present
                if "rms_delay_spread_ns" in config:
                    self.rms_delay_spread.setValue(config["rms_delay_spread_ns"])
                if "cluster_decay_ns" in config:
                    self.cluster_decay.setValue(config["cluster_decay_ns"])
                if "ray_decay_ns" in config:
                    self.ray_decay.setValue(config["ray_decay_ns"])
                if color:
                    self.update_color_preview(color)

    def accept(self):
        """Apply changes to all zone objects"""
        for zone in self.zones:
            # Update Path Loss Params
            zone.path_loss_params.path_loss_exponent = self.pl_exp.value()
            zone.path_loss_params.shadow_fading_std = self.shadow_std.value()
            if hasattr(zone.path_loss_params, 'frequency_decay_factor'):
                zone.path_loss_params.frequency_decay_factor = self.freq_decay.value()
            
            # Update NLOS Params
            zone.error_bias = self.error_bias.value()
            zone.noise_factor = self.noise_factor.value()
            
            # Update S-V Params
            if hasattr(zone, 'sv_params'):
                zone.sv_params.cluster_decay = self.cluster_decay.value() * 1e-9
                zone.sv_params.ray_decay = self.ray_decay.value() * 1e-9
                
            if hasattr(zone, 'rms_delay_spread'):
                zone.rms_delay_spread = self.rms_delay_spread.value() * 1e-9
                
                
        super().accept() 