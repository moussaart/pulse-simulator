from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QTextEdit, QPushButton, QCheckBox, 
                             QMessageBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import os
import re

from src.gui.utils.code_editor import CodeEditor
from src.gui.utils.python_highlighter import PythonHighlighter
from src.gui.widgets import ActionButton

class AlgorithmCreationWindow(QDialog):
    """
    Window for creating new custom localization algorithms.
    Provides a code editor with templates based on selected features.
    """
    
    def __init__(self, parent=None, algorithms_dir=None):
        super().__init__(parent)
        self.algorithms_dir = algorithms_dir
        self.setWindowTitle("Create New Algorithm")
        self.resize(1000, 700)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 20px;
                font-weight: bold;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QLineEdit {
                background-color: #252526;
                border: 1px solid #3e3e42;
                color: #d4d4d4;
                padding: 4px;
                border-radius: 2px;
            }
            QLineEdit:focus {
                border: 1px solid #007acc;
            }
            QTextEdit, QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
            }
            QPushButton {
                background-color: #3e3e42;
                color: #ffffff;
                border: 1px solid #3e3e42;
                padding: 6px 12px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QCheckBox {
                color: #d4d4d4;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QLabel {
                color: #d4d4d4;
            }
        """)
        
        self.init_ui()
        self.generate_template()  # Initial template
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # --- Top Section: Metadata ---
        meta_group = QGroupBox("Algorithm Details")
        meta_layout = QGridLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Weighted Centroid v2")
        self.name_input.textChanged.connect(self.update_filename_preview)
        self.name_input.textChanged.connect(self.update_code_name)
        
        self.filename_preview = QLabel("Filename: custom_algorithm.py")
        self.filename_preview.setStyleSheet("color: gray; font-style: italic;")
        
        meta_layout.addWidget(QLabel("Algorithm Name:"), 0, 0)
        meta_layout.addWidget(self.name_input, 0, 1)
        meta_layout.addWidget(self.filename_preview, 1, 1)
        
        meta_group.setLayout(meta_layout)
        layout.addWidget(meta_group)
        
        # --- Middle Section: Options & Editor ---
        middle_layout = QHBoxLayout()
        
        # Options Panel
        options_group = QGroupBox("Features")
        options_layout = QVBoxLayout()
        
        self.use_imu_cb = QCheckBox("Use IMU Data")
        self.use_imu_cb.setToolTip("Include IMU acceleration data in the update loop")
        self.use_imu_cb.stateChanged.connect(self.generate_template)
        
        self.use_state_cb = QCheckBox("Track State (Kalman)")
        self.use_state_cb.setToolTip("Include state and covariance in inputs/outputs")
        self.use_state_cb.stateChanged.connect(self.generate_template)
        
        self.use_nlos_cb = QCheckBox("Handle NLOS Info")
        self.use_nlos_cb.setToolTip("Include LOS/NLOS status for anchors")
        self.use_nlos_cb.stateChanged.connect(self.generate_template)
        
        options_layout.addWidget(self.use_imu_cb)
        options_layout.addWidget(self.use_state_cb)
        options_layout.addWidget(self.use_nlos_cb)
        options_layout.addStretch()
        
        options_group.setLayout(options_layout)
        middle_layout.addWidget(options_group, 1)
        
        # Code Editor
        editor_group = QGroupBox("Code Editor")
        editor_layout = QVBoxLayout()
        editor_layout.setContentsMargins(0, 10, 0, 0)
        
        # Use custom CodeEditor
        self.code_editor = CodeEditor()
        self.code_editor.setFont(QFont("Consolas", 11))
        
        # Apply syntax highlighter
        self.highlighter = PythonHighlighter(self.code_editor.document())
        
        editor_layout.addWidget(self.code_editor)
        editor_group.setLayout(editor_layout)
        middle_layout.addWidget(editor_group, 3)
        
        layout.addLayout(middle_layout)
        
        # --- Bottom Section: Actions ---
        btn_layout = QHBoxLayout()
        
        save_btn = ActionButton("💾 Save Algorithm", variant="success")
        save_btn.clicked.connect(self.save_algorithm)
        
        cancel_btn = ActionButton("Cancel", variant="secondary")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
    def update_filename_preview(self):
        name = self.name_input.text()
        safe_name = self._sanitize_filename(name)
        if not safe_name:
            safe_name = "custom_algorithm"
        self.filename_preview.setText(f"Filename: {safe_name}.py")

    def update_code_name(self):
        name = self.name_input.text() or "My Custom Algorithm"
        class_name = "".join(x.title() for x in self._sanitize_filename(name).split('_')) + "Algorithm"
        
        current_code = self.code_editor.toPlainText()
        
        # Update class definition
        new_code = re.sub(
            r'class\s+\w+\(BaseLocalizationAlgorithm\):',
            f'class {class_name}(BaseLocalizationAlgorithm):',
            current_code
        )
        
        # Update name property return value
        new_code = re.sub(
            r'def name\(self\) -> str:\s+return "[^"]+"',
            f'def name(self) -> str:\n        return "{name}"',
            new_code
        )
        
        # Only update if changed to avoid cursor jumping
        if new_code != current_code:
            cursor = self.code_editor.textCursor()
            scroll = self.code_editor.verticalScrollBar().value()
            
            self.code_editor.setPlainText(new_code)
            
            self.code_editor.setTextCursor(cursor)
            self.code_editor.verticalScrollBar().setValue(scroll)
        
    def _sanitize_filename(self, name):
        # Convert to snake_case-ish
        s = name.lower()
        s = re.sub(r'[^a-z0-9_]', '_', s)
        s = re.sub(r'_+', '_', s)
        s = s.strip('_')
        return s
        
    def generate_template(self):
        name = self.name_input.text() or "My Custom Algorithm"
        class_name = "".join(x.title() for x in self._sanitize_filename(name).split('_')) + "Algorithm"
        
        use_imu = self.use_imu_cb.isChecked()
        use_state = self.use_state_cb.isChecked()
        use_nlos = self.use_nlos_cb.isChecked()
        
        code = [
            "import numpy as np",
            "from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput",
            "",
            f"class {class_name}(BaseLocalizationAlgorithm):",
            "    \"\"\"",
            f"    Implementation of {name}",
            "    \"\"\"",
            "    ",
            "    @property",
            "    def name(self) -> str:",
            f"        return \"{name}\"",
            "",
            "    def initialize(self) -> None:",
            "        # Reset any internal state here",
            "        pass",
            "",
            "    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:",
            "        # Unpack inputs",
            "        measurements = input_data.measurements",
            "        anchors = input_data.anchors",
            "        dt = input_data.dt",
            ""
        ]
        
        if use_state:
            code.extend([
                "        state = input_data.state",
                "        covariance = input_data.covariance",
                "        previous_state = input_data.previous_state",
                "        previous_covariance = input_data.previous_covariance",
                "        Q = input_data.Q",
                "        R = input_data.R",
                "        initialized = input_data.initialized",
                "",
                "        if not initialized:",
                "            # Initialize state if needed",
                "            state = np.zeros(4) # x, y, vx, vy",
                "            covariance = np.eye(4)",
                "            previous_state = np.zeros(4)",
                "            previous_covariance = np.eye(4)",
                "            Q = np.eye(4) * 0.1",
                "            R = np.eye(4) * 0.1",
                "            initialized = True",
                ""
            ])
            
        if use_imu:
            code.extend([
                "        # IMU Data (if available)",
                "        if input_data.imu_data_on and input_data.accel is not None:",
                "            accel = input_data.accel  # [ax, ay, az]",
                "            gyro = input_data.gyro    # [gx, gy, gz]",
                ""
            ])
            
        if use_nlos:
            code.extend([
                "        # NLOS Status (0=LOS, 1=NLOS)",
                "        is_los = input_data.is_los",
                ""
            ])
            
        code.extend([
            "        # TODO: Implement your position estimation logic here",
            "        # Example: Just use the previous position or (0,0)",
            "        x, y = 0.0, 0.0",
            "        if input_data.tag.position:",
            "            x, y = input_data.tag.position.x, input_data.tag.position.y",
            "",
            "        # Return result",
            "        return AlgorithmOutput(",
            "            position=(float(x), float(y)),",
            "            state=state if 'state' in locals() else input_data.state,",
            "            covariance=covariance if 'covariance' in locals() else input_data.covariance,",
            "            initialized=initialized if 'initialized' in locals() else input_data.initialized,",
            "            previous_state=previous_state if 'previous_state' in locals() else input_data.previous_state,",
            "            previous_covariance=previous_covariance if 'previous_covariance' in locals() else input_data.previous_covariance,",
            "            Q=Q if 'Q' in locals() else input_data.Q,",
            "            R=R if 'R' in locals() else input_data.R",
            "        )"
        ])
        
        self.code_editor.setPlainText("\n".join(code))
        
    def save_algorithm(self):
        name = self.name_input.text()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Please provide a name for the algorithm.")
            return
            
        filename = self._sanitize_filename(name) + ".py"
        
        if not self.algorithms_dir or not os.path.exists(self.algorithms_dir):
            QMessageBox.critical(self, "Error", f"Algorithms directory not found: {self.algorithms_dir}")
            return
            
        file_path = os.path.join(self.algorithms_dir, filename)
        
        if os.path.exists(file_path):
            reply = QMessageBox.question(self, "File Exists", 
                                       f"Algorithm file '{filename}' already exists. Overwrite?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
                
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.code_editor.toPlainText())
            
            QMessageBox.information(self, "Success", f"Algorithm saved to {filename}")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
