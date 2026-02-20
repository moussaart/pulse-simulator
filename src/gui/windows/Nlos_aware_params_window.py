from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider, 
                            QDoubleSpinBox, QHBoxLayout, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from src.gui.widgets import create_labeled_spinbox

class NLOSAwareParamsWindow(QWidget):
    # Update signal to include new parameters
    params_changed = pyqtSignal(float, float, float, float, bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NLOS-Aware AEKF Parameters")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Alpha parameter
        self.alpha_label, self.alpha_spin = create_labeled_spinbox(
            "Alpha:", min_val=0, max_val=1, default=0.5, step=0.1
        )
        self.alpha_spin.valueChanged.connect(self.on_params_changed)
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(self.alpha_label)
        alpha_layout.addWidget(self.alpha_spin)
        layout.addLayout(alpha_layout)
        
        # Beta parameter
        self.beta_label, self.beta_spin = create_labeled_spinbox(
            "Beta:", min_val=0, max_val=1, default=0.5, step=0.1
        )
        self.beta_spin.valueChanged.connect(self.on_params_changed)
        beta_layout = QHBoxLayout()
        beta_layout.addWidget(self.beta_label)
        beta_layout.addWidget(self.beta_spin)
        layout.addLayout(beta_layout)
        
        # NLOS Factor parameter
        self.nlos_label, self.nlos_spin = create_labeled_spinbox(
            "NLOS Factor:", min_val=1, max_val=1000, default=100, step=10
        )
        self.nlos_spin.valueChanged.connect(self.on_params_changed)
        nlos_layout = QHBoxLayout()
        nlos_layout.addWidget(self.nlos_label)
        nlos_layout.addWidget(self.nlos_spin)
        layout.addLayout(nlos_layout)
        
        # Error Probability parameter
        self.error_prob_label, self.error_prob_spin = create_labeled_spinbox(
            "Error Probability:", min_val=0, max_val=1, default=0.1, step=0.05
        )
        self.error_prob_spin.valueChanged.connect(self.on_params_changed)
        error_layout = QHBoxLayout()
        error_layout.addWidget(self.error_prob_label)
        error_layout.addWidget(self.error_prob_spin)
        layout.addLayout(error_layout)

        # Add Imperfections checkbox
        imperfections_layout = QHBoxLayout()
        self.imperfections_check = QCheckBox("Add Imperfections")
        self.imperfections_check.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 12px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background-color: #363636;
                border: 1px solid #404040;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
            }
            QCheckBox::indicator:hover {
                border-color: #2196F3;
            }
        """)
        self.imperfections_check.stateChanged.connect(self.on_params_changed)
        imperfections_layout.addWidget(self.imperfections_check)
        layout.addLayout(imperfections_layout)
        
        # Set window style
        self.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
            }
        """)
        
        # Set fixed size
        self.setFixedSize(300, 220)
    
    def on_params_changed(self):
        """Emit signal when any parameter changes"""
        self.params_changed.emit(
            self.alpha_spin.value(),
            self.beta_spin.value(),
            self.nlos_spin.value(),
            self.error_prob_spin.value(),
            self.imperfections_check.isChecked()
        )
    
    def get_params(self):
        """Return current parameter values"""
        return (
            self.alpha_spin.value(),
            self.beta_spin.value(),
            self.nlos_spin.value(),
            self.error_prob_spin.value(),
            self.imperfections_check.isChecked()
        ) 