from PyQt5.QtWidgets import ( QVBoxLayout, QHBoxLayout, QLabel, QPushButton,  QDialog)
# Add this after the imports and before other class definitions
class CustomMessageBox(QDialog):
    """Custom styled message box for better visual appearance"""
    def __init__(self, title, message, icon_type="info", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                border: 2px solid #3c3c3c;
                border-radius: 10px;
            }
            QLabel {
                color: white;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton#CancelButton {
                background-color: #757575;
            }
            QPushButton#CancelButton:hover {
                background-color: #616161;
            }
        """)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Create header with icon and title
        header_layout = QHBoxLayout()
        
        # Set icon based on type
        icon_label = QLabel()
        if icon_type == "error":
            icon_text = "❌"
            icon_color = "#f44336"
        elif icon_type == "warning":
            icon_text = "⚠️"
            icon_color = "#ffc107"
        elif icon_type == "question":
            icon_text = "❓"
            icon_color = "#2196F3"
        else:  # info
            icon_text = "ℹ️"
            icon_color = "#2196F3"
            
        icon_label.setText(icon_text)
        icon_label.setStyleSheet(f"""
            QLabel {{
                color: {icon_color};
                font-size: 24px;
                padding: 10px;
            }}
        """)
        header_layout.addWidget(icon_label)
        
        # Message text
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        header_layout.addWidget(msg_label, 1)
        layout.addLayout(header_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        if icon_type == "question":
            # Yes/No buttons for questions
            yes_btn = QPushButton("Yes")
            no_btn = QPushButton("No")
            no_btn.setObjectName("CancelButton")
            yes_btn.clicked.connect(self.accept)
            no_btn.clicked.connect(self.reject)
            button_layout.addWidget(yes_btn)
            button_layout.addWidget(no_btn)
        else:
            # OK button for other types
            ok_btn = QPushButton("OK")
            ok_btn.clicked.connect(self.accept)
            button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)