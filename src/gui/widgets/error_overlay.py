"""
PULSE Simulation — Error Overlay Widget

A translucent overlay that appears on top of the simulation view when a
runtime error is detected.  It shows a user-friendly message, an optional
collapsible technical-detail section, and Restart / Dismiss buttons.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor, QFont


class ErrorOverlayWidget(QWidget):
    """Full-window overlay that displays simulation errors."""

    # Emitted when the user clicks the Restart button
    restart_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ErrorOverlay")
        self._setup_ui()
        self.hide()  # Hidden by default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_error(self, category: str, user_message: str, technical_details: str = ""):
        """Display the overlay with the given error information."""
        self._category_label.setText(f"⚠️  {category}")
        self._message_label.setText(user_message)
        if technical_details:
            self._details_text.setPlainText(technical_details)
            self._details_toggle.setVisible(True)
        else:
            self._details_text.clear()
            self._details_toggle.setVisible(False)
        self._details_text.setVisible(False)
        self._details_toggle.setText("▶  Show Technical Details")

        # Resize to cover parent
        self._resize_to_parent()
        self.show()
        self.raise_()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # Semi-transparent dark background
        self.setStyleSheet("""
            #ErrorOverlay {
                background-color: rgba(15, 15, 15, 210);
            }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)

        # Central card
        card = QWidget()
        card.setObjectName("ErrorCard")
        card.setFixedWidth(560)
        card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        card.setStyleSheet("""
            #ErrorCard {
                background-color: #252526;
                border: 1px solid #c53030;
                border-radius: 12px;
            }
        """)

        # Drop shadow
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(197, 48, 48, 120))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(16)

        # ── Header bar (red accent) ──
        header = QWidget()
        header.setFixedHeight(4)
        header.setStyleSheet("background-color: #e53e3e; border-radius: 2px;")
        card_layout.addWidget(header)

        # ── Category label ──
        self._category_label = QLabel("⚠️  Simulation Error")
        self._category_label.setStyleSheet("""
            color: #fc8181;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Segoe UI', Arial, sans-serif;
        """)
        card_layout.addWidget(self._category_label)

        # ── User-friendly message ──
        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet("""
            color: #e2e8f0;
            font-size: 14px;
            line-height: 1.5;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding: 4px 0px;
        """)
        card_layout.addWidget(self._message_label)

        # ── Collapsible technical details ──
        self._details_toggle = QPushButton("▶  Show Technical Details")
        self._details_toggle.setFlat(True)
        self._details_toggle.setCursor(Qt.PointingHandCursor)
        self._details_toggle.setStyleSheet("""
            QPushButton {
                color: #718096;
                font-size: 12px;
                text-align: left;
                border: none;
                padding: 2px 0px;
                background: transparent;
            }
            QPushButton:hover {
                color: #a0aec0;
            }
        """)
        self._details_toggle.clicked.connect(self._toggle_details)
        card_layout.addWidget(self._details_toggle)

        self._details_text = QTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setVisible(False)
        self._details_text.setMaximumHeight(160)
        self._details_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a2e;
                color: #a0aec0;
                border: 1px solid #2d3748;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        card_layout.addWidget(self._details_text)

        # ── Action buttons ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._dismiss_btn = QPushButton("✕  Dismiss")
        self._dismiss_btn.setCursor(Qt.PointingHandCursor)
        self._dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d3748;
                color: #e2e8f0;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 13px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #4a5568;
            }
            QPushButton:pressed {
                background-color: #1a202c;
            }
        """)
        self._dismiss_btn.clicked.connect(self.hide)

        self._restart_btn = QPushButton("🔄  Restart Simulation")
        self._restart_btn.setCursor(Qt.PointingHandCursor)
        self._restart_btn.setStyleSheet("""
            QPushButton {
                background-color: #c53030;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 13px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #e53e3e;
            }
            QPushButton:pressed {
                background-color: #9b2c2c;
            }
        """)
        self._restart_btn.clicked.connect(self._on_restart)

        btn_layout.addStretch()
        btn_layout.addWidget(self._dismiss_btn)
        btn_layout.addWidget(self._restart_btn)

        card_layout.addLayout(btn_layout)

        # Centre the card horizontally
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(card)
        h_layout.addStretch(1)

        outer.addLayout(h_layout)
        outer.addStretch(1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _toggle_details(self):
        visible = not self._details_text.isVisible()
        self._details_text.setVisible(visible)
        self._details_toggle.setText(
            "▼  Hide Technical Details" if visible else "▶  Show Technical Details"
        )

    def _on_restart(self):
        self.hide()
        self.restart_requested.emit()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _resize_to_parent(self):
        if self.parent():
            self.setGeometry(self.parent().rect())

    def resizeEvent(self, event):
        """Keep overlay sized to parent on window resize."""
        super().resizeEvent(event)
        self._resize_to_parent()
