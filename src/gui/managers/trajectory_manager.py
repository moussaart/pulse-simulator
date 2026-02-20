"""
Trajectory Manager Module
Handles custom trajectory creation and management
"""
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QGroupBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from src.core.motion import MotionController
from src.core.uwb.uwb_devices import Tag, Position


class TrajectoryManager:
    """Manages trajectory drawing and custom trajectory operations"""
    
    def __init__(self, parent):
        self.parent = parent
        self.trajectory_speed_rate = 50  # Default points per second
    
    def update_trajectory_plan(self):
        """Update the visualization of the planned trajectory"""
        # Prevent recursion by checking if we're already updating
        if hasattr(self, '_updating_trajectory') and self._updating_trajectory:
            return
        
        try:
            self._updating_trajectory = True
            x_points = []
            y_points = []
            
            if self.parent.movement_pattern.startswith("Custom:"):
                # For custom trajectories, plot the entire path
                trajectory_name = self.parent.movement_pattern.split(":", 1)[1]
                trajectory_points = MotionController.load_custom_trajectory(trajectory_name)
                if trajectory_points:
                    trajectory_points.append(trajectory_points[0])
                    x_points = [p[0] for p in trajectory_points]
                    y_points = [p[1] for p in trajectory_points]
            else:
                # For built-in patterns, generate points
                side = 8
                period = (4 * side) / self.parent.movement_speed
                t_points = np.linspace(0, period, 500)
                temp_tag = Tag(Position(0, 0))
                
                for t in t_points:
                    MotionController.update_tag_position(
                        tag=temp_tag,
                        movement_pattern=self.parent.movement_pattern,
                        movement_speed=self.parent.movement_speed,
                        t=t,
                        frequence=1/self.parent.dt if hasattr(self.parent, 'dt') and self.parent.dt > 0 else 200,
                        point=self.parent.point
                    )
                    x_points.append(temp_tag.position.x)
                    y_points.append(temp_tag.position.y)
            
            # Update trajectory plan plot
            if hasattr(self.parent, 'plot_items') and 'trajectory_plan' in self.parent.plot_items:
                self.parent.plot_items['trajectory_plan'].setData(x_points, y_points)
            
        except Exception as e:
            print(f"Error updating trajectory plan: {e}")
        finally:
            self._updating_trajectory = False
    
    def update_trajectory_patterns(self):
        """Update the pattern combo box with available custom trajectories"""
        # Skip if panel not opened yet
        if self.parent.pattern_combo is None:
            return
            
        current_pattern = self.parent.pattern_combo.currentText()
        
        # Get built-in patterns
        patterns = ["Circular", "Figure 8", "Square", "Random Walk", "Fixed Point", "Foot Mounted"]
        
        # Add custom trajectories
        custom_trajectories = MotionController.get_available_trajectories()
        for trajectory in custom_trajectories:
            patterns.append(f"Custom:{trajectory}")
        
        # Block signals to prevent infinite recursion
        self.parent.pattern_combo.blockSignals(True)
        
        # Update combo box
        self.parent.pattern_combo.clear()
        self.parent.pattern_combo.addItems(patterns)
        
        # Restore previous selection if it exists
        index = self.parent.pattern_combo.findText(current_pattern)
        if index >= 0:
            self.parent.pattern_combo.setCurrentIndex(index)
        
        # Unblock signals
        self.parent.pattern_combo.blockSignals(False)
    
    def save_trajectory_dialog(self):
        """Show dialog to save the drawn trajectory"""
        dialog = QDialog(self.parent)
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
        
        layout = QVBoxLayout(dialog)
        
        # Add trajectory info group
        info_group = QGroupBox("Trajectory Information")
        info_layout = QVBoxLayout()
        
        # Calculate stats
        sampling_freq = self.trajectory_speed_rate
        duration = len(self.parent.event_handler.trajectory_points) / sampling_freq
        distance = sum(np.sqrt(np.sum(np.diff(np.array(self.parent.event_handler.trajectory_points), axis=0)**2, axis=1)))
        
        stats_text = QLabel(
            f"Points: {len(self.parent.event_handler.trajectory_points)}\n"
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
        
        name_status = QLabel("")
        name_status.setStyleSheet("color: #ff5555;")
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
        
        # Validation and save logic
        def validate_name():
            name = name_input.text().strip()
            if not name:
                name_status.setText("Name cannot be empty")
                save_button.setEnabled(False)
                return False
            
            invalid_chars = '<>:"/\\|?*'
            if any(c in name for c in invalid_chars):
                name_status.setText(f"Name cannot contain: {invalid_chars}")
                save_button.setEnabled(False)
                return False
            
            if f"Custom:{name}" in [self.parent.pattern_combo.itemText(i) 
                                   for i in range(self.parent.pattern_combo.count())]:
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
                    self.parent.event_handler.trajectory_points,
                    sampling_freq=sampling_freq
                )
                
                # Update patterns list
                self.update_trajectory_patterns()
                
                # Select the new pattern
                self.parent.pattern_combo.setCurrentText(f"Custom:{name}")
                
                # Clean up drawing path
                if hasattr(self.parent, 'trajectory_preview'):
                    self.parent.position_plot.removeItem(self.parent.trajectory_preview)
                    self.parent.trajectory_preview = None
                
                # Update trajectory plan
                self.update_trajectory_plan()
                
                dialog.accept()
        
        save_button.clicked.connect(save_trajectory)
        cancel_button.clicked.connect(dialog.reject)
        
        # Initial validation
        validate_name()
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Show success message
            from src.core.localization import CustomMessageBox
            success_dialog = CustomMessageBox(
                "Trajectory Saved",
                f"Successfully saved trajectory '{name_input.text()}' with:\n"
                f"- {len(self.parent.event_handler.trajectory_points)} points\n"
                f"- {sampling_freq:.1f} Hz sampling rate\n"
                f"- {duration:.2f} seconds duration\n"
                f"- {distance:.2f} m total distance",
                "info",
                self.parent
            )
            success_dialog.exec_()
            
            # Resume simulation if it was paused
            if self.parent.is_paused:
                self.parent.pause_button.setChecked(False)
                self.parent.toggle_pause()
    
    def toggle_trajectory_drawing(self):
        """Toggle trajectory drawing mode (mutually exclusive with other modes)"""
        if not hasattr(self.parent.event_handler, 'drawing_trajectory'):
            self.parent.event_handler.drawing_trajectory = False
            self.parent.event_handler.trajectory_points = []
            self.parent.event_handler.is_recording = False
        
        is_activating = not self.parent.event_handler.drawing_trajectory
        
        if is_activating:
            # First disable all other modes
            self.parent.event_handler.disable_other_modes('drawing_trajectory')
            self.parent.event_handler.drawing_trajectory = True
            # Create dialog for speed rate selection
            speed_dialog = self.create_speed_dialog()
            
            if speed_dialog.exec_() == QDialog.Accepted:
                # Start drawing mode
                self.parent.draw_trajectory_btn.setText("Click to Start Recording")
                self.parent.position_plot.setCursor(Qt.CrossCursor)
                self.parent.event_handler.trajectory_points = []
                self.parent.event_handler.is_recording = False
                
                # Create preview line
                self.parent.trajectory_preview = self.parent.position_plot.plot([], [], pen='y')
                
                # Pause simulation
                self.parent.pause_button.setChecked(True)
                self.parent.toggle_pause()
            else:
                self.parent.event_handler.drawing_trajectory = False
        else:
            # Finish drawing mode
            self.parent.draw_trajectory_btn.setText("Draw New Trajectory")
            self.parent.position_plot.setCursor(Qt.ArrowCursor)
            self.parent.event_handler.is_recording = False
            
            # Remove preview line
            if hasattr(self.parent, 'trajectory_preview'):
                self.parent.position_plot.removeItem(self.parent.trajectory_preview)
                self.parent.trajectory_preview = None
            
            # Resume simulation if paused and no points recorded
            if self.parent.is_paused and len(self.parent.event_handler.trajectory_points) <= 1:
                self.parent.pause_button.setChecked(False)
                self.parent.toggle_pause()
            
            # Save trajectory if we have enough points
            if len(self.parent.event_handler.trajectory_points) > 1:
                self.save_trajectory_dialog()
    
    def create_speed_dialog(self):
        """Create dialog for trajectory speed rate selection"""
        speed_dialog = QDialog(self.parent)
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
        
        speed_label = QLabel("Set trajectory speed rate (points/second):")
        layout.addWidget(speed_label)
        
        speed_input = QDoubleSpinBox()
        speed_input.setRange(1, 1000)
        speed_input.setValue(50)
        speed_input.setSingleStep(1)
        layout.addWidget(speed_input)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(lambda: (
            setattr(self, 'trajectory_speed_rate', speed_input.value()),
            speed_dialog.accept()
        ))
        layout.addWidget(ok_button)
        
        return speed_dialog
    
    def import_csv_trajectory(self):
        """Import a trajectory from a CSV file"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import csv
        import os
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Import Trajectory CSV",
            "",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
            
        try:
            points = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                
                # Verify headers
                if not reader.fieldnames or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
                    QMessageBox.warning(self.parent, "Invalid CSV", 
                                      "CSV file must have 'x' and 'y' columns.")
                    return
                
                for row in reader:
                    try:
                        x = float(row['x'])
                        y = float(row['y'])
                        points.append([x, y])
                    except ValueError:
                        continue
            
            if not points:
                QMessageBox.warning(self.parent, "Empty Trajectory", 
                                  "No valid points found in CSV file.")
                return
                
            # Get filename as trajectory name
            name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Save it as a custom trajectory
            # Use default 50Hz if not specified in file (we don't parse timestamp yet for freq)
            MotionController.save_custom_trajectory(name, points, sampling_freq=50.0)
            
            # Update patterns list
            self.update_trajectory_patterns()
            
            # Select the new pattern
            pattern_name = f"Custom:{name}"
            index = self.parent.pattern_combo.findText(pattern_name)
            if index >= 0:
                self.parent.pattern_combo.setCurrentIndex(index)
            
            # Update trajectory plan visualization
            self.update_trajectory_plan()
            
            # Show success message
            from src.core.localization import CustomMessageBox
            success_dialog = CustomMessageBox(
                "Trajectory Imported",
                f"Successfully imported '{name}' with {len(points)} points.",
                "info",
                self.parent
            )
            success_dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Import Error", f"Failed to import trajectory: {str(e)}")

    def delete_custom_trajectory_dialog(self):
        """Show confirmation dialog to delete selected custom trajectory"""
        current_pattern = self.parent.pattern_combo.currentText()
        
        if not current_pattern.startswith("Custom:"):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self.parent, "Cannot Delete", 
                              "Only custom trajectories can be deleted.")
            return

        trajectory_name = current_pattern[7:]  # Remove "Custom:" prefix
        
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(self.parent, "Confirm Deletion", 
                                   f"Are you sure you want to delete the trajectory '{trajectory_name}'?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                   
        if reply == QMessageBox.Yes:
            if MotionController.delete_custom_trajectory(trajectory_name):
                # Update patterns list
                self.update_trajectory_patterns()
                
                # Reset to default
                self.parent.pattern_combo.setCurrentIndex(0)
                
                QMessageBox.information(self.parent, "Success", f"Trajectory '{trajectory_name}' deleted.")
            else:
                QMessageBox.critical(self.parent, "Error", f"Failed to delete trajectory '{trajectory_name}'.")

    def open_trajectory_folder(self):
        """Open the folder containing custom trajectories"""
        from src.utils.resource_loader import get_data_path
        import os
        
        folder = get_data_path("data/trajectories")
        os.makedirs(folder, exist_ok=True)
        
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self.parent, "Error", "Trajectory folder does not exist.")

