"""
Timeline Widget Module
Provides a timeline slider for navigating through recorded simulation history.
"""
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSlider, 
                             QLabel, QPushButton, QFrame, QSizePolicy,
                             QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from src.gui.theme import COLORS, BUTTON_SECONDARY_STYLE


class TimelineWidget(QWidget):
    """
    Timeline slider widget for simulation playback navigation.
    Allows scrubbing through recorded simulation history.
    """
    
    # Signals
    timeChanged = pyqtSignal(float)  # Emitted when user scrubs to new time
    playbackStarted = pyqtSignal()
    playbackStarted = pyqtSignal()
    playbackPaused = pyqtSignal()
    
    # Settings Signals
    durationChanged = pyqtSignal(str)
    customDurationChanged = pyqtSignal(int)
    recordingIntervalChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_time = 0.0
        self.current_time = 0.0
        self.is_playing = False
        self.playback_speed = 1.0
        
        # Playback timer for replay mode
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_playback_tick)
        self.playback_timer.setInterval(50)  # 20 fps playback
        
        self._setup_ui()
        # Timeline is always visible at the bottom
        
    def _setup_ui(self):
        """Setup the timeline UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(5)
        
        # Container with background
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(15, 10, 15, 10)
        container_layout.setSpacing(15)
        
        # Step back button
        self.step_back_btn = QPushButton("⏮")
        self.step_back_btn.setFixedSize(32, 28)
        self.step_back_btn.setToolTip("Step Back")
        self.step_back_btn.clicked.connect(self._on_step_back)
        self._style_button(self.step_back_btn)
        container_layout.addWidget(self.step_back_btn)
        
        # Play/Pause button for replay
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(32, 28)
        self.play_btn.setToolTip("Play Recording")
        self.play_btn.clicked.connect(self._on_play_pause)
        self._style_button(self.play_btn)
        container_layout.addWidget(self.play_btn)
        
        # Step forward button
        self.step_forward_btn = QPushButton("⏭")
        self.step_forward_btn.setFixedSize(32, 28)
        self.step_forward_btn.setToolTip("Step Forward")
        self.step_forward_btn.clicked.connect(self._on_step_forward)
        self._style_button(self.step_forward_btn)
        container_layout.addWidget(self.step_forward_btn)
        
        # Current time label
        self.current_time_label = QLabel("0.00s")
        self.current_time_label.setFixedWidth(55)
        self.current_time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.current_time_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent']};
                font-size: 12px;
                font-weight: bold;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
        """)
        container_layout.addWidget(self.current_time_label)
        
        # Timeline slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)  # Use 1000 steps for smooth scrubbing
        self.slider.setValue(0)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {COLORS['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLORS['primary_hover']};
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent']}, stop:1 {COLORS['primary_light']});
                border-radius: 3px;
            }}
        """)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        container_layout.addWidget(self.slider, stretch=1)
        
        # Total time label
        self.total_time_label = QLabel("/ 0.00s")
        self.total_time_label.setFixedWidth(60)
        self.total_time_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
        """)
        container_layout.addWidget(self.total_time_label)
        
        main_layout.addWidget(container)
        
        # State tracking
        self._user_scrubbing = False

        # --- Settings Section (Duration & Recording) ---
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet(f"background-color: {COLORS['border']};")
        sep1.setFixedHeight(20)
        container_layout.addWidget(sep1)

        # Duration
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        container_layout.addWidget(duration_label)

        self.duration_combo = QComboBox() # QComboBox is not imported yet, need to check imports
        self.duration_combo.addItems(["1s", "5s", "10s", "20s", "30s", "60s", "Custom", "∞ Infinite"])
        self.duration_combo.setCurrentText("30s")
        self.duration_combo.setToolTip("Simulation Duration")
        self.duration_combo.setFixedWidth(85)
        self.duration_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['input_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 2px 4px;
            }}
        """)
        self.duration_combo.currentTextChanged.connect(self._on_duration_changed)
        container_layout.addWidget(self.duration_combo)

        # Custom Duration SpinBox (Initially Hidden)
        self.custom_duration_spin = QSpinBox() # QSpinBox is not imported yet
        self.custom_duration_spin.setRange(1, 3600)
        self.custom_duration_spin.setValue(30)
        self.custom_duration_spin.setSuffix("s")
        self.custom_duration_spin.setToolTip("Custom Duration (s)")
        self.custom_duration_spin.setFixedWidth(60)
        self.custom_duration_spin.setVisible(False)
        self.custom_duration_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['input_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        self.custom_duration_spin.valueChanged.connect(self._on_custom_duration_changed)
        container_layout.addWidget(self.custom_duration_spin)

        # Recording
        rec_label = QLabel("Rec:")
        rec_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        container_layout.addWidget(rec_label)

        self.recording_combo = QComboBox()
        self.recording_combo.addItems(["Every Frame", "Every 3rd", "Every 5th", "Every 10th"])
        self.recording_combo.setCurrentText("Every 5th")
        self.recording_combo.setToolTip("Recording Interval (Snapshot Frequency)")
        self.recording_combo.setFixedWidth(90)
        self.recording_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['input_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 2px 4px;
            }}
        """)
        self.recording_combo.currentTextChanged.connect(self._on_recording_changed)
        container_layout.addWidget(self.recording_combo)

        # Separator 2
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet(f"background-color: {COLORS['border']};")
        sep2.setFixedHeight(20)
        container_layout.addWidget(sep2)

        
    def _style_button(self, btn):
        """Apply consistent styling to buttons"""
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                font-size: 14px;
                padding: 2px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
                border-color: {COLORS['accent']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_pressed']};
            }}
        """)
    
    def set_time_range(self, start_time: float, end_time: float):
        """
        Set the time range for the timeline.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        self.max_time = end_time - start_time
        self.total_time_label.setText(f"/ {self.max_time:.2f}s")
        self._update_time_display()
        
    def set_current_time(self, time: float, emit_signal: bool = True):
        """
        Set the current time position.
        
        Args:
            time: Time in seconds
            emit_signal: Whether to emit timeChanged signal
        """
        self.current_time = max(0, min(time, self.max_time))
        
        # Update slider without triggering callback
        if not self._user_scrubbing:
            slider_value = int((self.current_time / self.max_time) * 1000) if self.max_time > 0 else 0
            self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)
        
        self._update_time_display()
        
        if emit_signal:
            self.timeChanged.emit(self.current_time)
    
    def _update_time_display(self):
        """Update the time labels"""
        self.current_time_label.setText(f"{self.current_time:.2f}s")
    
    def _on_slider_changed(self, value):
        """Handle slider value changes"""
        if self.max_time > 0:
            self.current_time = (value / 1000.0) * self.max_time
            self._update_time_display()
            self.timeChanged.emit(self.current_time)
    
    def _on_slider_pressed(self):
        """Handle slider press - pause playback while scrubbing"""
        self._user_scrubbing = True
        if self.is_playing:
            self.playback_timer.stop()
    
    def _on_slider_released(self):
        """Handle slider release"""
        self._user_scrubbing = False
        if self.is_playing:
            self.playback_timer.start()
    
    def _on_step_back(self):
        """Step back by one snapshot interval"""
        step_size = 0.25  # 250ms step
        self.set_current_time(self.current_time - step_size)
    
    def _on_step_forward(self):
        """Step forward by one snapshot interval"""
        step_size = 0.25  # 250ms step
        self.set_current_time(self.current_time + step_size)
    
    def _on_play_pause(self):
        """Toggle playback"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start playing back the recording"""
        self.is_playing = True
        self.play_btn.setText("⏸")
        self.play_btn.setToolTip("Pause Playback")
        self.playback_timer.start()
        self.playbackStarted.emit()
    
    def pause_playback(self):
        """Pause playback"""
        self.is_playing = False
        self.play_btn.setText("▶")
        self.play_btn.setToolTip("Play Recording")
        self.playback_timer.stop()
        self.playbackPaused.emit()
    
    def _on_playback_tick(self):
        """Advance playback by one tick"""
        if self.current_time >= self.max_time:
            # Reached end, stop playback
            self.pause_playback()
            return
        
        # Advance by 50ms * playback_speed
        new_time = self.current_time + (0.05 * self.playback_speed)
        self.set_current_time(new_time)
    
    def reset(self):
        """Reset the timeline to initial state"""
        self.pause_playback()
        self.current_time = 0.0
        self.max_time = 0.0
        self.slider.setValue(0)
        self._update_time_display()
        self.total_time_label.setText("/ 0.00s")
        # Timeline stays visible (don't hide)
    
    def show_timeline(self, start_time: float, end_time: float):
        """
        Show the timeline with the given time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        self.set_time_range(start_time, end_time)
        self.set_current_time(end_time, emit_signal=True)
        self.set_current_time(end_time, emit_signal=True)
        self.show()

    # --- Settings Slots ---
    
    def _on_duration_changed(self, text):
        """Handle duration combo change"""
        # Show/hide custom spinbox
        is_custom = text == "Custom"
        self.custom_duration_spin.setVisible(is_custom)
        
        self.durationChanged.emit(text)
        
    def _on_custom_duration_changed(self, value):
        """Handle custom duration spinbox change"""
        self.customDurationChanged.emit(value)
        
    def _on_recording_changed(self, text):
        """Handle recording interval change"""
        self.recordingIntervalChanged.emit(text)

