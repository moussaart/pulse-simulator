from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, 
                             QGroupBox,  QPushButton, QDialog,
                            QDoubleSpinBox, QLineEdit )
from PyQt5.QtCore import Qt

import numpy as np
from src.core.uwb.uwb_devices import Tag, Position
from src.core.motion import MotionController


class TrajectoryInterface():
    
    def __init__(self):
        super().__init__()
        self.drawing_trajectory = False
        self.trajectory_points = []
        self.is_recording = False
        self.trajectory_speed_rate = 50
        self.trajectory_preview = None
        self.trajectory_speed_rate_input = None
        self.trajectory_speed_rate_label = None
        self.trajectory_speed_rate_layout = None
        self.trajectory_speed_rate_group = None
        
    
    @staticmethod
    def toggle_trajectory_drawing(self):
        """Toggle trajectory drawing mode"""
        if not hasattr(self, 'drawing_trajectory'):
            self.drawing_trajectory = False
            self.trajectory_points = []
            self.is_recording = False  # Add recording state
        
        self.drawing_trajectory = not self.drawing_trajectory
        
        if self.drawing_trajectory:
            # Create dialog for speed rate selection
            speed_dialog = QDialog(self)
            speed_dialog.setWindowTitle("Set Trajectory Speed")
            speed_dialog.setStyleSheet("""
                QDialog {
                    background-color: #2d2d2d;
                    min-width: 300px;
                }
                QLabel {
                    color: #ffffff;
                    font-size: 12px;
                    padding: 5px;
                }
                QDoubleSpinBox {
                    background-color: #363636;
                    border: 1px solid #404040;
                    border-radius: 4px;
                    padding: 8px;
                    color: white;
                    font-size: 12px;
                    margin: 5px;
                }
                QPushButton {
                    background-color: #2196F3;
                    border-radius: 4px;
                    padding: 8px 16px;
                    color: white;
                    font-weight: bold;
                    margin: 5px;
                }
            """)
            
            layout = QVBoxLayout(speed_dialog)
            
            # Add speed rate input
            speed_label = QLabel("Set trajectory speed rate (points/second):")
            layout.addWidget(speed_label)
            
            speed_input = QDoubleSpinBox()
            speed_input.setRange(1, 100)  # Allow speeds from 1 to 100 points/second
            speed_input.setValue(50)  # Default value
            speed_input.setSingleStep(1)
            layout.addWidget(speed_input)
            
            # Add OK button
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(speed_dialog.accept)
            layout.addWidget(ok_button)
            
            if speed_dialog.exec_() == QDialog.Accepted:
                self.trajectory_speed_rate = speed_input.value()
                # Start drawing mode
                self.draw_trajectory_btn.setText("Click to Start Recording")
                self.position_plot.setCursor(Qt.CrossCursor)
                self.trajectory_points = []
                self.is_recording = False
                
                # Create preview line
                self.trajectory_preview = self.position_plot.plot([], [], pen='y')
                
                # Pause simulation
                self.pause_button.setChecked(True)
                self.toggle_pause()
            else:
                self.drawing_trajectory = False
                return
        else:
            # Finish drawing mode
            self.draw_trajectory_btn.setText("Draw New Trajectory")
            self.position_plot.setCursor(Qt.ArrowCursor)
            self.is_recording = False
            
            # Remove preview line
            self.position_plot.removeItem(self.trajectory_preview)
            
            # Save trajectory if we have points
            if len(self.trajectory_points) > 1:
                self.save_trajectory_dialog()

    @staticmethod
    def save_trajectory_dialog(self):
        """Show enhanced dialog to save the drawn trajectory"""
        # Create custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Custom Trajectory")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
                padding: 5px;
            }
            QLineEdit {
                background-color: #363636;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                color: white;
                font-size: 12px;
                margin: 5px;
            }
            QPushButton {
                background-color: #2196F3;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton#cancelButton {
                background-color: #666666;
            }
            QPushButton#cancelButton:hover {
                background-color: #808080;
            }
            QGroupBox {
                border: 1px solid #404040;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                color: white;
            }
        """)

        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add trajectory info group
        info_group = QGroupBox("Trajectory Information")
        info_layout = QVBoxLayout()
        
        # Add stats using the custom speed rate
        sampling_freq = self.trajectory_speed_rate  # Use the custom speed rate
        duration = len(self.trajectory_points) / sampling_freq
        distance = sum(np.sqrt(np.sum(np.diff(np.array(self.trajectory_points), axis=0)**2, axis=1)))
        
        stats_text = QLabel(
            f"Points: {len(self.trajectory_points)}\n"
            f"Duration: {duration:.2f} seconds\n"
            f"Sampling Rate: {sampling_freq:.1f} Hz\n"
            f"Total Distance: {distance:.2f} m"
        )
        info_layout.addWidget(stats_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Add name input
        name_group = QGroupBox("Trajectory Name")
        name_layout = QVBoxLayout()
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter trajectory name...")
        name_layout.addWidget(name_input)
        
        # Add name validation
        name_status = QLabel("")
        name_status.setStyleSheet("color: #ff5555;")  # Red for errors
        name_layout.addWidget(name_status)
        
        name_group.setLayout(name_layout)
        layout.addWidget(name_group)
        
        # Add buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("cancelButton")
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
        
        # Add validation and save logic
        def validate_name():
            name = name_input.text().strip()
            if not name:
                name_status.setText("Name cannot be empty")
                save_button.setEnabled(False)
                return False
            
            # Check for invalid characters
            invalid_chars = '<>:"/\\|?*'
            if any(c in name for c in invalid_chars):
                name_status.setText(f"Name cannot contain: {invalid_chars}")
                save_button.setEnabled(False)
                return False
            
            # Check if name already exists
            if f"Custom:{name}" in self.pattern_combo.currentText():
                name_status.setText("This name already exists")
                save_button.setEnabled(False)
                return False
            
            name_status.setText("")
            save_button.setEnabled(True)
            return True
        
        name_input.textChanged.connect(validate_name)
        
        def save_trajectory():
            if validate_name():
                name = name_input.text().strip()
                
                # Save trajectory
                MotionController.save_custom_trajectory(
                    name, 
                    self.trajectory_points,
                    sampling_freq=sampling_freq
                )
                
                # Update patterns list
                self.update_trajectory_patterns()
                
                # Select the new pattern
                self.pattern_combo.setCurrentText(f"Custom:{name}")
                
                dialog.accept()
        
        save_button.clicked.connect(save_trajectory)
        cancel_button.clicked.connect(dialog.reject)
        
        # Initial validation
        validate_name()
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Show success message
            self.show_info_message(
                "Trajectory Saved",
                f"Successfully saved trajectory '{name_input.text()}' with:\n"
                f"- {len(self.trajectory_points)} points\n"
                f"- {sampling_freq:.1f} Hz sampling rate\n"
                f"- {duration:.2f} seconds duration\n"
                f"- {distance:.2f} m total distance"
            )

    @staticmethod
    def update_trajectory_plan(self):
        """Update the visualization of the planned trajectory"""
        try:
            x_points = []
            y_points = []
            
            if self.movement_pattern.startswith("Custom:"):
                # For custom trajectories, plot the entire path
                trajectory_name = self.movement_pattern.split(":", 1)[1]
                trajectory_points = MotionController.load_custom_trajectory(trajectory_name)
                if trajectory_points:
                    # Add first point at the end to close the loop
                    trajectory_points.append(trajectory_points[0])
                    x_points = [p[0] for p in trajectory_points]
                    y_points = [p[1] for p in trajectory_points]
            else:
                # For built-in patterns, generate points using MotionController
                t_points = np.linspace(0, 20, 500)  # 20 seconds of trajectory
                temp_tag = Tag(Position(0, 0))
                
                # Generate trajectory points using MotionController
                for t in t_points:
                    MotionController.update_tag_position(temp_tag, self.movement_pattern, 1.0, t)
                    x_points.append(temp_tag.position.x)
                    y_points.append(temp_tag.position.y)
            
            # Update trajectory plan plot
            self.trajectory_plan.setData(x_points, y_points)
            
        except Exception as e:
            print(f"Error updating trajectory plan: {e}")
