import numpy as np
import gc
import time as _time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFrame, QSplitter, QCheckBox,
    QSpinBox, QComboBox, QDockWidget, QGridLayout, QGroupBox,
    QTextEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPen, QColor, QFont
import pyqtgraph as pg
from collections import deque

from src.api.ai_gym_server import AIGymServer
from src.api.ai_training_facade import AITrainingAPI
from src.core.localization.Localization_alghorthime import LocalizationAlgorthimes
from src.core.localization.Alghortimes_doc import Alghortimes_doc
from src.core.localization.base_algorithm import AlgorithmInput
from src.core.motion import MotionController
from src.core.uwb.uwb_devices import Tag, Position
from src.core.uwb.energy_model import EnergyCalculator

AVAILABLE_FILTERS = [
    'Trilateration',
    'Extended Kalman Filter',
    'Unscented Kalman Filter',
    'Cubature Kalman Filter',
    'Adaptive Extended Kalman Filter'
]

class AITrainingWindow(QMainWindow):
    """
    Independent window for AI Training.
    Connects to AIGymServer, pauses simulation to wait for RL agent actions,
    and visualizes the results (chosen anchors) in real-time.
    Supports N simultaneous agents (multi-point).
    """
    def __init__(self, main_app, num_agents=1, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.num_agents = num_agents
        self.setWindowTitle("PULSE - AI Training Environment")
        self.resize(800, 600)
        
        # Core data copied from main app
        self.anchors = self.main_app.anchors
        self.nlos_manager = self.main_app.nlos_manager
        self.trajectory_manager = self.main_app.trajectory_manager
        self.channel_model = self.main_app.channel_conditions
        
        # Algorithm name from main app (read once, kept in sync via combo)
        self.algorithm = getattr(self.main_app, 'algorithm', "Trilateration")
        
        # NLOS Aware / Algorithm detailed parameters
        self.los_aware_alpha = getattr(self.main_app, 'los_aware_alpha', 0.5)
        self.los_aware_beta = getattr(self.main_app, 'los_aware_beta', 0.5)
        self.los_aware_nlos_factor = getattr(self.main_app, 'los_aware_nlos_factor', 100)
        
        # Algorithm dispatch mapping (same as SimulationManager)
        self.algorithm_methods = Alghortimes_doc().get_algorithm_methods()
        self.algorithm_instances = {} # Cache for class-based algorithms
        
        # Configurable port (default 5555, user can change via UI before start)
        self._server_port = 5555
        self.server = AIGymServer(port=self._server_port)
        self.server.start()
        
        # AI Training Facade for enriched observation building
        self.training_api = AITrainingAPI(self.main_app)
        
        # Measurement source: "uwb", "imu", or "both"
        self._measurement_source = "uwb"
        
        # Pause main simulation so we take control
        if hasattr(self.main_app, 'pause_simulation'):
            self.main_app.pause_simulation()
            
        # Copy theme from main application
        if hasattr(self.main_app, 'styleSheet'):
            self.setStyleSheet(self.main_app.styleSheet())
            # Ensure proper theming of plots
            pg.setConfigOption('background', '#1e1e1e')
            pg.setConfigOption('foreground', '#d4d4d4')
            
        self.is_playing = False
        self._client_was_connected = False
        self.current_step = 0
        
        # Log message buffer — must be initialized before setup_ui / log_message
        self._log_buffer = []
        self._last_status_text = ""
        
        self.setup_ui()
        self.log_message(f"🟢 [System] PULSE AI Training Environment initialized. Listening on port {self._server_port}.")
        
        # Physics state
        self.state_sent_for_step = False
        
        # ── Per-agent state ──────────────────────────────────────────────
        # Each agent has its own tag, simulation clock, EKF state, and
        # estimated position so that multi-agent training is independent.
        self.agent_tags = []         # Tag objects with IMU
        self.agent_sim_times = []    # Independent sim clock per agent
        self.ekf_states = []
        self.ekf_Ps = []
        self.ekf_initializeds = []
        self.ekf_Qs = []
        self.ekf_Rs = []
        self.prev_errors = []
        self.curr_errors = []
        self.agent_est_positions = []
        self.agent_energy_calculators = []
        
        for _ in range(self.num_agents):
            t = Tag(Position(0, 0))
            t.imu_data_on = True
            self.agent_tags.append(t)
            self.agent_sim_times.append(0.0)
            self.ekf_states.append(np.array([0.0, 0.0, 0.0, 0.0]))
            self.ekf_Ps.append(np.eye(4) * 5.0)
            self.ekf_initializeds.append(False)
            self.ekf_Qs.append(getattr(self.main_app, 'aekf_Q', None))
            self.ekf_Rs.append(getattr(self.main_app, 'aekf_R', None))
            self.prev_errors.append(0.0)
            self.curr_errors.append(0.0)
            self.agent_est_positions.append(None)
            self.agent_energy_calculators.append(EnergyCalculator())
        
        # Per-agent adaptive EKF parameters
        self.adaptive_iekf_iteration_counts = [0] * self.num_agents
        self.adaptive_iekf_prev_Rs = [None] * self.num_agents
        self.adaptive_iekf_innovation_histories = [None] * self.num_agents
        
        self.adaptive_iekf_mu = getattr(self.main_app, 'adaptive_iekf_mu', 0.95)
        self.adaptive_iekf_alpha = getattr(self.main_app, 'adaptive_iekf_alpha', 0.3)
        self.adaptive_iekf_xi = getattr(self.main_app, 'adaptive_iekf_xi', 20)
        self.adaptive_iekf_lambda_min = getattr(self.main_app, 'adaptive_iekf_lambda_min', 0.1)
        self.adaptive_iekf_lambda_max = getattr(self.main_app, 'adaptive_iekf_lambda_max', 3.0)
        self.adaptive_iekf_tau = getattr(self.main_app, 'adaptive_iekf_tau', 0.95)
        
        # Per-agent cumulative rewards (server-side)
        self.agent_cumul_rewards = [0.0] * self.num_agents
        self.agent_step_rewards = [0.0] * self.num_agents
        
        # ── Time-based simulation parameters ──────────────────────────────
        self._tau_seconds = 1.0        # Macro-step duration (s), updated from client
        self._imu_freq = 100.0         # IMU update rate (Hz)
        self._uwb_freq = 10.0          # UWB update rate (Hz)
        self._walking_speed = 1.4      # Human walking speed (m/s)
        
        # Per-agent last measurement source (for energy in next observation)
        self.agent_last_sources = ["uwb"] * self.num_agents
        
        # Timer for stepping the environment
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.environment_step)
        self.timer.setInterval(10)  # 10ms tick — fast enough, gives Qt event loop room
        
        # Plot refresh throttle: limit redraws to ~5 FPS to keep UI responsive
        self._PLOT_REFRESH_INTERVAL = 0.200  # seconds between plot redraws
        self._last_plot_update_time = 0.0


        # Build initial trajectory preview from current config
        self.trajectory_points = []
        self._generate_trajectory_preview()
        self.refresh_base_plot()

    # ── Live configuration readers (read from main_app panel widgets) ───

    @property
    def movement_pattern(self) -> str:
        """Read the current movement pattern from the main window's combo box."""
        combo = getattr(self.main_app, 'pattern_combo', None)
        if combo is not None:
            return combo.currentText()
        return getattr(self.main_app, 'movement_pattern', 'Circular')

    @property
    def movement_speed(self) -> float:
        """Read the current speed from the main window's speed slider (value/10)."""
        slider = getattr(self.main_app, 'speed_slider', None)
        if slider is not None:
            return slider.value() / 10.0
        return getattr(self.main_app, 'movement_speed', 1.0)

    @property
    def dt(self) -> float:
        """Read the current timestep from the main window's timestep slider."""
        slider = getattr(self.main_app, 'timestep_slider', None)
        if slider is not None:
            return slider.value() / 1000.0   # ms → s
        return getattr(self.main_app, 'dt', 0.005)

    @property
    def point(self) -> tuple:
        """Read the fixed-point target from the main window's spin boxes."""
        fp_x = getattr(self.main_app, 'fp_x_spin', None)
        fp_y = getattr(self.main_app, 'fp_y_spin', None)
        if fp_x is not None and fp_y is not None:
            return (fp_x.value(), fp_y.value())
        return getattr(self.main_app, 'point', (0.0, 0.0))

    # ── Trajectory preview (for the dashed line on the map) ──────────────

    def _generate_trajectory_preview(self):
        """Build a visual preview of the trajectory for the map overlay.
        This is purely cosmetic; actual motion uses MotionController per step."""
        self.trajectory_points = []
        try:
            pattern = self.movement_pattern
            speed = self.movement_speed
            step = self.dt
            pt = self.point

            if pattern.startswith("Custom:"):
                trajectory_name = pattern.split(":", 1)[1]
                t_points = MotionController.load_custom_trajectory(trajectory_name)
                if t_points:
                    self.trajectory_points = [[p[0], p[1]] for p in t_points]
            else:
                side = 8
                period = (4 * side) / speed if speed > 0 else 10
                t_range = np.arange(0, period, step) if step > 0 else np.linspace(0, period, 500)
                temp_tag = Tag(Position(0, 0))
                for t in t_range:
                    MotionController.update_tag_position(
                        tag=temp_tag,
                        movement_pattern=pattern,
                        movement_speed=speed,
                        t=t,
                        frequence=1 / step if step > 0 else 200,
                        point=pt,
                        dt=step,
                    )
                    self.trajectory_points.append([temp_tag.position.x, temp_tag.position.y])
        except Exception as e:
            import traceback
            print(f"Error generating trajectory preview: {e}\n{traceback.format_exc()}")

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("ai_training_central")
        central_widget.setStyleSheet("#ai_training_central { background-color: #1e1e1e; }")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 1. Status Bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Server: Waiting for Client... Click Play to begin streaming.")
        status_layout.addWidget(self.status_label)
        
        self.toggle_decision_btn = QPushButton("Show Agent Decisions")
        self.toggle_decision_btn.setCheckable(True)
        self.toggle_decision_btn.setStyleSheet("padding: 5px 10px; background-color: #2b2b2b; border: 1px solid #3c3c3c; border-radius: 4px;")
        status_layout.addWidget(self.toggle_decision_btn)
        
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Dock Widget for Agent Decisions
        self.decision_dock = QDockWidget("Agent Decisions", self)
        self.decision_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        
        dock_content = QWidget()
        self.decision_layout = QGridLayout(dock_content)
        self.agent_decision_labels = []
        
        # Technology source colors: consistent across the UI
        self._tech_colors = {
            "uwb": "#2196F3",    # blue
            "imu": "#4CAF50",    # green
            "both": "#AB47BC",   # purple
        }
        self._tech_icons = {
            "uwb": "📡",
            "imu": "🧭",
            "both": "📡🧭",
        }
        
        for i in range(self.num_agents):
            group = QGroupBox(f"Agent {i} Status")
            group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; color: #aaa; }")
            g_layout = QVBoxLayout(group)
            
            algo_label = QLabel("Algorithm: N/A")
            source_label = QLabel("Source: N/A")
            anchors_label = QLabel("Anchors: N/A")
            # Large technology badge showing current tech
            tech_badge = QLabel("📡 UWB")
            tech_badge.setAlignment(Qt.AlignCenter)
            tech_badge.setStyleSheet(
                "background-color: #1a3a5c; color: #2196F3; font-weight: bold; "
                "font-size: 14px; padding: 6px 12px; border-radius: 8px; "
                "border: 2px solid #2196F3;"
            )
            # Energy breakdown label
            energy_label = QLabel("Energy: N/A")
            energy_label.setStyleSheet("color: #FF5722; font-size: 10px;")
            
            algo_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
            source_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 11px;")
            anchors_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 11px;")
            
            g_layout.addWidget(tech_badge)
            g_layout.addWidget(algo_label)
            g_layout.addWidget(source_label)
            g_layout.addWidget(anchors_label)
            g_layout.addWidget(energy_label)
            self.agent_decision_labels.append({
                "algo": algo_label, 
                "source": source_label,
                "anchors": anchors_label,
                "tech_badge": tech_badge,
                "energy": energy_label,
            })
            self.decision_layout.addWidget(group, i // 2, i % 2) # 2 columns max
        
        self.decision_dock.setWidget(dock_content)
        self.addDockWidget(Qt.RightDockWidgetArea, self.decision_dock)
        self.decision_dock.hide()
        
        self.toggle_decision_btn.toggled.connect(self.decision_dock.setVisible)
        self.decision_dock.visibilityChanged.connect(self.toggle_decision_btn.setChecked)

        # Bottom Training Console Dock Widget
        self.console_dock = QDockWidget("Training Console", self)
        self.console_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("background-color: black; color: #00FF00; font-family: Consolas, Monaco, monospace; font-size: 11px;")
        
        self.console_dock.setWidget(self.console_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        
        # 2. Splitter for Map and Metrics
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Main Plot area
        from src.gui.widgets.plot_helpers import create_themed_plot
        self.plot_widget = create_themed_plot(title="AI Training Map", show_grid=True)
        self.plot_widget.setAspectLocked(True)
        self.splitter.addWidget(self.plot_widget)
        
        # Metrics area – use a 2×2 grid for cleaner layout
        self.metrics_widget = QWidget()
        metrics_grid = QGridLayout(self.metrics_widget)
        metrics_grid.setContentsMargins(2, 2, 2, 2)
        metrics_grid.setSpacing(4)
        
        # ── Row 0, Col 0: Localization Error ──────────────────────────────
        self.error_plot = create_themed_plot(title="Localization Error (m)", y_label="Error (m)", x_label="Time (s)")
        self.error_plot.addLegend(offset=(10, 10))
        
        # ── Row 0, Col 1: Technology Used Over Time ──────────────────
        self.tech_timeline_plot = create_themed_plot(
            title="Technology Used Over Time", y_label="", x_label="Time (s)"
        )
        self.tech_timeline_plot.addLegend(offset=(10, 10))
        y_axis = self.tech_timeline_plot.getAxis('left')
        y_axis.setTicks([[(1, 'UWB'), (2, 'Both'), (3, 'IMU')]])
        self.tech_timeline_plot.setYRange(0.5, 3.5)
        
        # ── Row 1, Col 0: Cumulative Reward ───────────────────────────────
        self.reward_plot = create_themed_plot(title="Cumulative Reward", y_label="Reward", x_label="Step")
        self.reward_plot.addLegend(offset=(10, 10))
        
        # ── Row 1, Col 1: Step Reward ─────────────────────────────────────
        self.step_reward_plot = create_themed_plot(title="Step Reward", y_label="Reward", x_label="Step")
        self.step_reward_plot.addLegend(offset=(10, 10))
        
        # ── Row 2, Col 0: Cumulative Energy (µJ) ─────────────────────────
        self.cumul_energy_plot = create_themed_plot(
            title="Cumulative Energy (µJ)",
            y_label="Energy (µJ)", x_label="Time (s)"
        )
        self.cumul_energy_plot.addLegend(offset=(10, 10))
        
        # Initialize per-agent metric data and curves
        MAX_PLOT_POINTS = 2000
        self.metric_time_data = deque(maxlen=MAX_PLOT_POINTS)  # shared time axis (seconds)
        self.per_agent_errors = [deque(maxlen=MAX_PLOT_POINTS) for _ in range(self.num_agents)]
        self.per_agent_cumul_rewards = [deque(maxlen=MAX_PLOT_POINTS) for _ in range(self.num_agents)]
        self.per_agent_step_rewards = [deque(maxlen=MAX_PLOT_POINTS) for _ in range(self.num_agents)]
        
        # Per-agent technology source history (1=UWB, 2=Both, 3=IMU)
        self.per_agent_tech_source = [deque(maxlen=MAX_PLOT_POINTS) for _ in range(self.num_agents)]
        
        # Per-agent step energy (µJ) and cumulative energy (µJ)
        self.per_agent_step_energy_uJ = [deque(maxlen=MAX_PLOT_POINTS) for _ in range(self.num_agents)]
        self.per_agent_cumul_energy_uJ = [deque(maxlen=MAX_PLOT_POINTS) for _ in range(self.num_agents)]
        
        # Per-agent step counters for IMU / UWB usage
        self.agent_imu_step_counts = [0] * self.num_agents
        self.agent_uwb_step_counts = [0] * self.num_agents
        
        # Baseline calculators (constant reference lines)
        self.imu_baseline_calc = EnergyCalculator()
        self.imu_baseline_calc.config.uwb_disabled = True
        self.imu_baseline_calc.config.imu_enabled = True
        self.imu_baseline_calc.config.num_anchors = 0
        
        self.uwb_baseline_calc = EnergyCalculator()
        self.uwb_baseline_calc.config.uwb_disabled = False
        self.uwb_baseline_calc.config.imu_enabled = True
        self.uwb_baseline_calc.config.num_anchors = len(self.anchors) if self.anchors else 8
        
        # Compute constant baseline step-energy values (µJ) = power_mW * dt * 1000
        imu_only_result = self.imu_baseline_calc.calculate()
        uwb_only_result = self.uwb_baseline_calc.calculate()
        self.imu_only_power_mW = imu_only_result.total_power_mW
        self.uwb_only_power_mW = uwb_only_result.total_power_mW
        self.imu_only_step_energy_uJ = self.imu_only_power_mW * self.dt * 1000.0
        self.uwb_only_step_energy_uJ = self.uwb_only_power_mW * self.dt * 1000.0
        
        self.error_curves = []
        self.reward_curves = []
        self.step_reward_curves = []
        self.cumul_energy_curves = []
        self.tech_timeline_curves = []
        for i in range(self.num_agents):
            hue = int((i / max(1, self.num_agents)) * 360)
            color = QColor.fromHsl(hue, 255, 127)
            pen = pg.mkPen(color, width=2)
            lbl = f"Agent {i}"
            self.error_curves.append(self.error_plot.plot(pen=pen, name=lbl))
            self.reward_curves.append(self.reward_plot.plot(pen=pen, name=lbl))
            self.step_reward_curves.append(self.step_reward_plot.plot(pen=pen, name=lbl))
            self.cumul_energy_curves.append(self.cumul_energy_plot.plot(pen=pen, name=lbl))
            
            # Technology timeline: scatter dots colored by source type
            tech_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None))
            self.tech_timeline_plot.addItem(tech_scatter)
            self.tech_timeline_curves.append(tech_scatter)
            
        imu_pen = pg.mkPen(QColor(46, 204, 113), width=2, style=Qt.DashLine)
        uwb_pen = pg.mkPen(QColor(231, 76, 60), width=2, style=Qt.DashLine)
        
        # Baseline reference lines on the cumulative energy plot (linearly growing)
        self.imu_baseline_cumul_energy_curve = self.cumul_energy_plot.plot(
            pen=imu_pen, name="IMU Only"
        )
        self.uwb_baseline_cumul_energy_curve = self.cumul_energy_plot.plot(
            pen=uwb_pen, name="UWB Always"
        )
        
        # Layout: 2×2 grid + cumulative energy taking the whole row at bottom
        metrics_grid.addWidget(self.error_plot, 0, 0)
        metrics_grid.addWidget(self.tech_timeline_plot, 0, 1)
        metrics_grid.addWidget(self.reward_plot, 1, 0)
        metrics_grid.addWidget(self.step_reward_plot, 1, 1)
        metrics_grid.addWidget(self.cumul_energy_plot, 2, 0, 1, 2)
        
        # Equal row stretches for top 2 rows, slightly more for energy plots
        metrics_grid.setRowStretch(0, 3)
        metrics_grid.setRowStretch(1, 3)
        metrics_grid.setRowStretch(2, 4)
        metrics_grid.setColumnStretch(0, 1)
        metrics_grid.setColumnStretch(1, 1)
        
        self.splitter.addWidget(self.metrics_widget)
        self.metrics_widget.setVisible(True)
        self.splitter.setSizes([500, 500]) # Equal split for grid layout
        
        layout.addWidget(self.splitter, stretch=1)
        
        # Generate unique colors for each agent
        self.agent_colors = []
        self.agent_brushes = []
        for i in range(max(1, self.num_agents)):
            hue = int((i / max(1, self.num_agents)) * 360)
            color = QColor.fromHsl(hue, 255, 127) # Full saturation, 50% lightness
            self.agent_colors.append(color)
            self.agent_brushes.append(pg.mkBrush(color))
            
        # Plot items
        self.anchor_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('b'), brush=pg.mkBrush(0, 0, 255, 120))
        self.trajectory_line = pg.PlotDataItem(pen=pg.mkPen('k', width=2, style=Qt.DashLine))
        self.true_pos_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
        self.est_pos_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None)) # Brushes set per-point
        self.chosen_anchors_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None)) # Small dots for chosen anchors
        
        self.plot_widget.addItem(self.trajectory_line)
        self.plot_widget.addItem(self.anchor_scatter)
        self.plot_widget.addItem(self.true_pos_scatter)
        self.plot_widget.addItem(self.est_pos_scatter)
        self.plot_widget.addItem(self.chosen_anchors_scatter) # Small colored dots for chosen anchors
        
        # Removed connection lines to anchors based on user feedback to prevent visual cascade
            
        # 3. Controls
        controls_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶️ Start Training")
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_reset = QPushButton("⏹️ Reset")
        
        self.cb_show_metrics = QCheckBox("Show Live Metrics & Selection")
        self.cb_show_metrics.setChecked(True)
        self.cb_show_metrics.toggled.connect(self.metrics_widget.setVisible)
        
        # Port configuration
        port_label = QLabel("Port:")
        self.port_spinner = QSpinBox()
        self.port_spinner.setRange(1024, 65535)
        self.port_spinner.setValue(self._server_port)
        self.port_spinner.setToolTip("TCP port for RL client connections. Change requires restart.")
        self.port_spinner.valueChanged.connect(self._on_port_changed)
        
        # Measurement source selector
        source_label = QLabel("Source:")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["UWB", "IMU", "Both"])
        self.source_combo.setCurrentText("UWB")
        self.source_combo.setToolTip("Measurement source: UWB ranging, IMU only, or fused")
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_reset.clicked.connect(self.reset)
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addWidget(self.cb_show_metrics)
        controls_layout.addWidget(port_label)
        controls_layout.addWidget(self.port_spinner)
        controls_layout.addWidget(source_label)
        controls_layout.addWidget(self.source_combo)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

    def refresh_base_plot(self):
        """Draws the static elements: Anchors, NLOS Zones, and Trajectory"""
        # Plot anchor items with text labels
        spots = [{'pos': (a.position.x, a.position.y), 'data': 1} for a in self.anchors]
        self.anchor_scatter.setData(spots)
        
        # Clear existing text items first
        for item in list(self.plot_widget.items()):
            if isinstance(item, pg.TextItem):
                self.plot_widget.removeItem(item)
                
        # Add labels to anchors
        for i, a in enumerate(self.anchors):
            text = pg.TextItem(text=str(i), anchor=(0.5, 1.5), color=(0, 0, 255))
            text.setPos(a.position.x, a.position.y)
            self.plot_widget.addItem(text)
        
        # NLOS Zones
        all_zones = self.channel_model.nlos_zones + self.channel_model.moving_nlos_zones
        for zone in all_zones:
            try:
                if hasattr(zone, 'points'): # PolygonNLOSZone
                    corners = zone.points
                    if corners and corners[0] != corners[-1]:
                        corners = list(corners) + [corners[0]]
                elif hasattr(zone, 'get_corners'): # MovingNLOSZone
                    corners = zone.get_corners()
                else: # Standard NLOSZone
                    corners = [(zone.x1, zone.y1), (zone.x2, zone.y1),
                               (zone.x2, zone.y2), (zone.x1, zone.y2),
                               (zone.x1, zone.y1)]
                
                x_val, y_val = zip(*corners)
                color = self.nlos_manager.get_zone_color(zone)
                zone_item = pg.PlotDataItem(
                    list(x_val), list(y_val),
                    fillLevel=0,
                    brush=pg.mkBrush(color[0], color[1], color[2], 50),
                    pen=pg.mkPen(color[0], color[1], color[2], 255)
                )
                self.plot_widget.addItem(zone_item)
            except Exception as e:
                print(f"Failed to plot AI Training NLOS zone: {e}")
        
        # Trajectory
        if len(self.trajectory_points) > 0:
            pts = np.array(self.trajectory_points)
            self.trajectory_line.setData(pts[:, 0], pts[:, 1])

    def play(self):
        # Re-generate the trajectory preview with the latest config
        self._generate_trajectory_preview()
        self.refresh_base_plot()
            
        self.is_playing = True
        self.status_label.setText("Running Simulation... Waiting for RL Actions.")
        self.timer.start()

    def pause(self):
        self.is_playing = False
        self.status_label.setText("Paused")
        self.timer.stop()

    def reset(self):
        self.pause()
        self.current_step = 0
        self.state_sent_for_step = False
        self.true_pos_scatter.setData([])
        self.est_pos_scatter.setData([])
        self.chosen_anchors_scatter.setData([])
        
        # Reset all per-agent state
        for i in range(self.num_agents):
            tag = self.agent_tags[i]
            tag.position.x = 0.0
            tag.position.y = 0.0
            tag.velocity.x = 0.0
            tag.velocity.y = 0.0
            tag.acceleration.x = 0.0
            tag.acceleration.y = 0.0
            tag.imu_data.clear()
            tag.imu_simulator.reset()
            tag.last_update_time = None  # force first-sample logic in MotionController
            
            self.agent_sim_times[i] = 0.0
            self.ekf_states[i] = np.array([0.0, 0.0, 0.0, 0.0])
            self.ekf_Ps[i] = np.eye(4) * 5.0
            self.ekf_initializeds[i] = False
            self.ekf_Qs[i] = getattr(self.main_app, 'aekf_Q', None)
            self.ekf_Rs[i] = getattr(self.main_app, 'aekf_R', None)
            self.prev_errors[i] = 0.0
            self.curr_errors[i] = 0.0
            self.agent_est_positions[i] = None
            self.agent_energy_calculators[i].reset_accumulator()
            self.adaptive_iekf_iteration_counts[i] = 0
            self.adaptive_iekf_prev_Rs[i] = None
            self.adaptive_iekf_innovation_histories[i] = None
            self.agent_cumul_rewards[i] = 0.0
            self.agent_step_rewards[i] = 0.0
            self.agent_last_sources[i] = "uwb"
        
        # Reset algorithm instances (so they re-initialize)
        self.algorithm_instances.clear()
        
        # Reset energy accumulator
        self.training_api.reset_energy_accumulator()
        
        # Regenerate trajectory preview with latest config
        self._generate_trajectory_preview()
        self.refresh_base_plot()
        
        # Reset per-agent Metric data
        self.metric_time_data.clear()
        for i in range(self.num_agents):
            self.per_agent_errors[i].clear()
            self.per_agent_cumul_rewards[i].clear()
            self.per_agent_step_rewards[i].clear()
            self.per_agent_tech_source[i].clear()
            self.per_agent_step_energy_uJ[i].clear()
            self.per_agent_cumul_energy_uJ[i].clear()
            self.agent_imu_step_counts[i] = 0
            self.agent_uwb_step_counts[i] = 0
            self.error_curves[i].setData([], [])
            self.reward_curves[i].setData([], [])
            self.step_reward_curves[i].setData([], [])
            self.step_energy_curves[i].setData([], [])
            self.cumul_energy_curves[i].setData([], [])
            self.tech_timeline_curves[i].setData([], [])
        
        self.imu_baseline_energy_curve.setData([], [])
        self.uwb_baseline_energy_curve.setData([], [])
        self.imu_baseline_cumul_energy_curve.setData([], [])
        self.uwb_baseline_cumul_energy_curve.setData([], [])
        
        self.status_label.setText("Reset to beginning.")
        self.log_message("⏹ [System] Simulation state reset.")

    def log_message(self, message: str):
        """Buffer a log message for batched console output.
        
        Messages are flushed to the QTextEdit during the throttled plot update
        cycle (~5 FPS) to avoid per-step overhead.
        """
        self._log_buffer.append(message)

    def _flush_log_buffer(self):
        """Write all buffered log messages to the console widget at once."""
        if not self._log_buffer or not hasattr(self, 'console_text'):
            return
        
        # Join all messages and append in one call
        batch_text = "\n".join(self._log_buffer)
        self.console_text.append(batch_text)
        self._log_buffer.clear()
        
        # Cap console log to prevent unbounded memory growth
        MAX_LOG_LINES = 5000
        doc = self.console_text.document()
        if doc.blockCount() > MAX_LOG_LINES:
            full_text = self.console_text.toPlainText()
            lines = full_text.split('\n')
            keep_lines = lines[-MAX_LOG_LINES:]
            self.console_text.setPlainText('\n'.join(keep_lines))
            
        cursor = self.console_text.textCursor()
        cursor.movePosition(cursor.End)
        self.console_text.setTextCursor(cursor)

    def closeEvent(self, event):
        """Clean up server when window closes."""
        self.pause()
        self.server.stop()
        
        # Sync the button state on the main window
        if hasattr(self.main_app, 'ai_data_btn'):
            self.main_app.ai_data_btn.blockSignals(True)
            self.main_app.ai_data_btn.setChecked(False)
            self.main_app.ai_data_btn.setText("🤖 Start AI")
            self.main_app.ai_data_btn.blockSignals(False)
            
        # Ensure it creates a fresh window next time
        self.main_app.ai_window = None
        super().closeEvent(event)

    def _advance_agent_motion(self, agent_idx: int):
        """Advance a single agent's tag using MotionController (same as main window).
        This computes position, velocity, acceleration, orientation, angular
        velocity, and generates IMU data using proper cubic-spline interpolation."""
        tag = self.agent_tags[agent_idx]
        sim_t = self.agent_sim_times[agent_idx]
        dt = self.dt
        pattern = self.movement_pattern
        speed = self.movement_speed
        pt = self.point

        MotionController.update_tag_position(
            tag=tag,
            movement_pattern=pattern,
            movement_speed=speed,
            t=sim_t,
            frequence=1.0 / dt if dt > 0 else 200,
            point=pt,
            dt=dt,
        )
        # MotionController already calls tag.update_imu(sim_t) internally

    def environment_step(self):
        """The core Loop: Send State → Wait Action → Compute → Advance"""
        if not self.is_playing:
            return

        # Track client connection state change
        is_conn = self.server.connected
        if is_conn != self._client_was_connected:
            self._client_was_connected = is_conn
            if is_conn:
                self.log_message(f"🟢 [Connection] Client connected to server on port {self._server_port}.")
            else:
                self.log_message(f"🔴 [Connection] Client disconnected.")

        # 1. SEND STATE (enriched observation via facade)
        if not self.state_sent_for_step:
            all_states = []
            for a_idx in range(self.num_agents):
                # Advance this agent's tag along the trajectory
                self._advance_agent_motion(a_idx)
                agent_tag = self.agent_tags[a_idx]
                true_pos = [agent_tag.position.x, agent_tag.position.y]

                measurements = {}
                los_conditions = []
                
                for i, anchor in enumerate(self.anchors):
                    is_los = self.channel_model.check_los_to_anchor(
                        anchor.position, Position(true_pos[0], true_pos[1])
                    )
                    los_conditions.append(is_los)
                    
                    true_distance = np.linalg.norm([
                        anchor.position.x - true_pos[0],
                        anchor.position.y - true_pos[1]
                    ])
                    try:
                        dist, _ = self.channel_model.measure_distance(
                            true_distance=true_distance,
                            is_los=is_los,
                            anchor_pos=anchor.position
                        )
                    except Exception:
                        dist, _ = self.channel_model.measure_distance(
                            true_distance=true_distance,
                            is_los=is_los
                        )
                    measurements[anchor.id] = dist
                
                # Build enriched state via the facade
                # Include server-computed reward so the agent has it in the state
                state_dict = self.training_api.build_step_observation(
                    agent_id=a_idx,
                    step=self.current_step,
                    dt=self.dt,
                    true_pos=true_pos,
                    est_pos=self.agent_est_positions[a_idx],
                    tag=agent_tag,
                    anchors=self.anchors,
                    measurements=measurements,
                    los_conditions=los_conditions,
                    curr_error=self.curr_errors[a_idx],
                    prev_error=self.prev_errors[a_idx],
                    algorithm_name=self.algorithm,
                    movement_speed=self.movement_speed,
                    movement_pattern=self.movement_pattern,
                    channel_model=self.channel_model,
                    measurement_source=self.agent_last_sources[a_idx],
                )
                # Inject server-computed reward data into the state
                state_dict["reward"] = {
                    "step_reward": float(self.agent_step_rewards[a_idx]),
                    "cumulative_reward": float(self.agent_cumul_rewards[a_idx]),
                    "num_agents": self.num_agents,
                }
                all_states.append(state_dict)
            
            success = self.server.send_state(all_states)
            if success:
                self.state_sent_for_step = True
                # Log periodically (not every step) to reduce overhead
                if self.current_step % 100 == 0:
                    self.log_message(f"📤 t={self.agent_sim_times[0]:.2f}s — state sent (step {self.current_step})")
            else:
                self.status_label.setText("Error: RL Client not connected.")
                return  # Try again next tick
                
        # 2. WAIT FOR ACTION
        action_response = self.server.wait_for_action(timeout=0.01) # Non-blocking poll
        if action_response is None:
            return # Wait for next GUI tick
            
        all_action_indices, metrics = action_response
        if self.current_step % 100 == 0:
            self.log_message(f"📥 Received actions at t={self.agent_sim_times[0]:.2f}s")
        
        # Legacy global cumulative_reward override is removed; now handled per-agent in the loop below.
        
        # 3. APPLY ACTION (Compute Location for all agents)
        
        all_true_spots = []
        all_est_spots = []
        
        # Read current true positions from the advanced tags
        true_poses = [[t.position.x, t.position.y] for t in self.agent_tags]
        
        # Robust Action Parser
        if all_action_indices is None:
            all_action_indices = []
            
        if isinstance(all_action_indices, dict):
            all_action_indices = [all_action_indices]
            
        if isinstance(all_action_indices, list):
            if len(all_action_indices) > 0 and isinstance(all_action_indices[0], (int, float)):
                all_action_indices = [all_action_indices]
                
        if len(all_action_indices) == 0:
            all_action_indices = [{
                "filter": self.algorithm,
                "measurement_source": self._measurement_source,
                "anchors": list(range(min(4, len(self.anchors))))
            }] * self.num_agents
        elif len(all_action_indices) < self.num_agents:
            last_action = all_action_indices[-1]
            all_action_indices = list(all_action_indices) + [last_action] * (self.num_agents - len(all_action_indices))

        for a_idx in range(self.num_agents):
            if a_idx >= len(all_action_indices):
                break
                
            action_obj = all_action_indices[a_idx]
            
            if isinstance(action_obj, dict):
                action_indices = action_obj.get("anchors", [])
                agent_algo_name = action_obj.get("filter", self.algorithm)
                agent_source = action_obj.get("measurement_source", self._measurement_source)
            else:
                action_indices = action_obj
                agent_algo_name = self.algorithm
                agent_source = self._measurement_source
                
            if a_idx < len(self.agent_decision_labels):
                # Cache badge style — only update labels, skip expensive setStyleSheet
                self.agent_decision_labels[a_idx]["algo"].setText(f"Algorithm: {agent_algo_name}")
                self.agent_decision_labels[a_idx]["source"].setText(f"Source: {agent_source}")
                self.agent_decision_labels[a_idx]["anchors"].setText(f"Anchors: {action_indices}")
                
                # Only update badge if technology changed (avoids CSS re-parse)
                src_key = agent_source.lower()
                if getattr(self, f'_last_badge_src_{a_idx}', None) != src_key:
                    setattr(self, f'_last_badge_src_{a_idx}', src_key)
                    tech_icon = self._tech_icons.get(src_key, "📡")
                    tech_color = self._tech_colors.get(src_key, "#2196F3")
                    tech_name = {"uwb": "UWB", "imu": "IMU", "both": "UWB + IMU"}.get(src_key, agent_source.upper())
                    self.agent_decision_labels[a_idx]["tech_badge"].setText(f"{tech_icon} {tech_name}")
                    bg_color = QColor(tech_color)
                    bg_color.setAlpha(40)
                    self.agent_decision_labels[a_idx]["tech_badge"].setStyleSheet(
                        f"background-color: {bg_color.name()}; color: {tech_color}; font-weight: bold; "
                        f"font-size: 14px; padding: 6px 12px; border-radius: 8px; "
                        f"border: 2px solid {tech_color};"
                    )
            
            # Store this agent's source for the next step's energy observation
            self.agent_last_sources[a_idx] = agent_source
                
            true_pos = true_poses[a_idx]
            measurements_list = []
            chosen_anchors = []
            
            all_true_spots.append({'pos': (true_pos[0], true_pos[1])})
            
            if agent_source.lower() != "imu":
                for idx in action_indices:
                    if 0 <= idx < len(self.anchors):
                        anchor = self.anchors[idx]
                        
                        is_los = self.channel_model.check_los_to_anchor(
                            anchor.position, Position(true_pos[0], true_pos[1])
                        )
                        true_distance = np.linalg.norm([
                            anchor.position.x - true_pos[0],
                            anchor.position.y - true_pos[1]
                        ])
                        try:
                            dist, _ = self.channel_model.measure_distance(
                                true_distance=true_distance,
                                is_los=is_los,
                                anchor_pos=anchor.position
                            )
                        except Exception:
                            dist, _ = self.channel_model.measure_distance(
                                true_distance=true_distance,
                                is_los=is_los
                            )
                        measurements_list.append(dist)
                        chosen_anchors.append(anchor)
            
            # Compute location for this agent using current algorithm
            # Use the agent_tag which has real IMU data from update_imu()
            agent_tag = self.agent_tags[a_idx]
            agent_tag.position.x = true_pos[0]
            agent_tag.position.y = true_pos[1]
            # Determine if IMU data should be passed based on measurement source
            use_imu = agent_source.lower() in ("imu", "both")
            
            # Dynamically resize state and P if algorithm dimension requires 6 dimensions (IMU Only, IMU assisted NLOS-Aware)
            expected_dim = 4
            if "IMU assisted NLOS-Aware AEKF" in agent_algo_name or "IMU Only" in agent_algo_name:
                expected_dim = 6
            
            if len(self.ekf_states[a_idx]) != expected_dim:
                self.ekf_states[a_idx] = np.zeros(expected_dim)
                self.ekf_states[a_idx][0:2] = true_pos
                if expected_dim == 6:
                    self.ekf_Ps[a_idx] = np.diag([1.0, 1.0, 0.1, 0.1, 0.05, 0.05])
                else:
                    self.ekf_Ps[a_idx] = np.eye(expected_dim) * 5.0
                self.ekf_initializeds[a_idx] = False
            
            try:
                method = self.algorithm_methods.get(agent_algo_name)
                
                # Get real IMU 3D acceleration and gyroscope measurements
                imu_data = agent_tag.imu_data
                if len(imu_data) > 0:
                    accel_3d = np.array([
                        float(imu_data.acc_x[-1]),
                        float(imu_data.acc_y[-1]),
                        float(imu_data.acc_z[-1])
                    ])
                    gyro_3d = np.array([
                        float(imu_data.gyro_x[-1]),
                        float(imu_data.gyro_y[-1]),
                        float(imu_data.gyro_z[-1])
                    ])
                    u = np.array([float(imu_data.acc_x[-1]), float(imu_data.acc_y[-1])])
                else:
                    accel_3d = np.array([0.0, 0.0, 0.0])
                    gyro_3d = np.array([0.0, 0.0, 0.0])
                    u = np.array([0.0, 0.0])
                
                import inspect
                from src.core.localization.base_algorithm import BaseLocalizationAlgorithm
                
                if method and inspect.isclass(method) and issubclass(method, BaseLocalizationAlgorithm):
                    if (agent_algo_name, a_idx) not in self.algorithm_instances:
                        self.algorithm_instances[(agent_algo_name, a_idx)] = method()
                        self.algorithm_instances[(agent_algo_name, a_idx)].initialize()
                    
                    algo_instance = self.algorithm_instances[(agent_algo_name, a_idx)]
                    
                    # Ensure Q and R are initialized and have correct dimensions (avoids aekf.py crashes)
                    # This logic handles changes in the number of anchors smoothly
                    num_anchors = len(chosen_anchors)
                    
                    # 1. Initialize Q if missing (Process Noise)
                    if self.ekf_Qs[a_idx] is None:
                        # Standard 4x4 Q for [x, y, vx, vy]
                        q_noise = 0.1
                        self.ekf_Qs[a_idx] = np.eye(4) * q_noise
                        
                    # 2. Initialize or Resize R if missing or wrong dimension (Measurement Noise)
                    if self.ekf_Rs[a_idx] is None or self.ekf_Rs[a_idx].shape[0] != num_anchors:
                        r_noise = 0.15
                        self.ekf_Rs[a_idx] = np.eye(num_anchors) * (r_noise**2)
                    
                    input_data = AlgorithmInput(
                        measurements=measurements_list,
                        anchors=chosen_anchors,
                        tag=agent_tag,
                        dt=self.dt,
                        state=self.ekf_states[a_idx],
                        covariance=self.ekf_Ps[a_idx],
                        initialized=self.ekf_initializeds[a_idx],
                        Q=self.ekf_Qs[a_idx],
                        R=self.ekf_Rs[a_idx],
                        imu_data_on=use_imu,
                        accel=accel_3d if use_imu else None,
                        gyro=gyro_3d if use_imu else None,
                        control_input=u,
                        is_los=[0 if self.channel_model.check_los_to_anchor(a.position, Position(true_pos[0], true_pos[1])) else 1 for a in chosen_anchors],
                        params={
                            "movement_speed": self.movement_speed,
                            "movement_pattern": self.movement_pattern,
                            "dt": self.dt
                        }
                    )
                    
                    output = algo_instance.update(input_data)
                    est_pos = output.position
                    self.ekf_states[a_idx] = output.state
                    self.ekf_Ps[a_idx] = output.covariance
                    self.ekf_initializeds[a_idx] = output.initialized
                    self.ekf_Qs[a_idx] = output.Q
                    self.ekf_Rs[a_idx] = output.R
                elif method:
                    # Function-based algorithms (legacy)
                    # Mapping logic for NLOS-Aware etc.
                    is_los_bits = [0 if self.channel_model.check_los_to_anchor(a.position, Position(true_pos[0], true_pos[1])) else 1 for a in chosen_anchors]
                    
                    if "Improved Adaptive EKF" in agent_algo_name:
                        # Special handling if needed, or fallback to general call
                        result = method(
                            measurements=measurements_list,
                            tag=agent_tag,
                            anchors=chosen_anchors,
                            aekf_state=self.ekf_states[a_idx],
                            aekf_P=self.ekf_Ps[a_idx],
                            aekf_initialized=self.ekf_initializeds[a_idx],
                            dt=self.dt,
                            mu=self.adaptive_iekf_mu,
                            alpha=self.adaptive_iekf_alpha,
                            xi=self.adaptive_iekf_xi,
                            lambda_min=self.adaptive_iekf_lambda_min,
                            lambda_max=self.adaptive_iekf_lambda_max,
                            tau=self.adaptive_iekf_tau,
                            iteration_count=self.adaptive_iekf_iteration_counts[a_idx],
                            prev_R=self.adaptive_iekf_prev_Rs[a_idx],
                            innovation_history=self.adaptive_iekf_innovation_histories[a_idx],
                            imu_data_on=use_imu,
                            u=u
                        )
                    elif "NLOS-Aware" in agent_algo_name:
                        if "IMU assisted NLOS-Aware AEKF" in agent_algo_name:
                            result = method(
                                measurements=measurements_list,
                                tag=agent_tag,
                                anchors=chosen_anchors,
                                state=self.ekf_states[a_idx],
                                P=self.ekf_Ps[a_idx],
                                initialized=self.ekf_initializeds[a_idx],
                                is_los=is_los_bits,
                                alpha=self.los_aware_alpha,
                                beta=self.los_aware_beta,
                                nlos_factor=self.los_aware_nlos_factor,
                                dt=self.dt,
                                zupt_threshold=0.05,
                                R=self.ekf_Rs[a_idx]
                            )
                        else:
                            result = method(
                                measurements=measurements_list,
                                tag=agent_tag,
                                anchors=chosen_anchors,
                                aekf_state=self.ekf_states[a_idx],
                                aekf_P=self.ekf_Ps[a_idx],
                                aekf_initialized=self.ekf_initializeds[a_idx],
                                is_los=is_los_bits,
                                alpha=self.los_aware_alpha,
                                beta=self.los_aware_beta,
                                nlos_factor=self.los_aware_nlos_factor,
                                dt=self.dt,
                                imu_data_on=use_imu,
                                u=u,
                                R=self.ekf_Rs[a_idx],
                                Q=self.ekf_Qs[a_idx]
                            )
                    elif "Kalman" in agent_algo_name:
                        if "Adaptive Extended Kalman Filter" in agent_algo_name:
                            result = method(
                                measurements=measurements_list,
                                tag=agent_tag,
                                anchors=chosen_anchors,
                                aekf_state=self.ekf_states[a_idx],
                                aekf_P=self.ekf_Ps[a_idx],
                                aekf_initialized=self.ekf_initializeds[a_idx],
                                dt=self.dt,
                                Q=self.ekf_Qs[a_idx],
                                R=self.ekf_Rs[a_idx],
                                imu_data_on=use_imu,
                                u=u
                            )
                        else:
                            result = method(
                                measurements_list, agent_tag, chosen_anchors,
                                self.ekf_states[a_idx], self.ekf_Ps[a_idx],
                                self.ekf_initializeds[a_idx], self.dt,
                                imu_data_on=use_imu, u=u
                            )
                    elif "IMU Only" in agent_algo_name:
                        measurements_imu = [float(agent_tag.imu_data.acc_x[-1]), 
                                           float(agent_tag.imu_data.acc_y[-1])]
                        result = method(
                            tag=agent_tag,
                            measurements=measurements_imu,
                            state=self.ekf_states[a_idx],
                            P=self.ekf_Ps[a_idx],
                            initialized=self.ekf_initializeds[a_idx],
                            dt=self.dt
                        )
                    else:
                        # Generic trilateration etc.
                        result = method(measurements_list, chosen_anchors)
                        
                    # Unpack result
                    if isinstance(result, tuple):
                        if "Improved Adaptive EKF" in agent_algo_name:
                            # (position, innovation_history, state, P, initialized, aekf_Q, aekf_prev_R)
                            est_pos = result[0]
                            self.adaptive_iekf_innovation_histories[a_idx] = result[1]
                            self.ekf_states[a_idx] = result[2]
                            self.ekf_Ps[a_idx] = result[3]
                            self.ekf_initializeds[a_idx] = result[4]
                            self.ekf_Qs[a_idx] = result[5]
                            self.adaptive_iekf_prev_Rs[a_idx] = result[6]
                            self.ekf_Rs[a_idx] = result[6]
                        elif "IMU assisted NLOS-Aware AEKF" in agent_algo_name:
                            # (position, imu_state, imu_P, kf_initialized, aekf_R)
                            est_pos = result[0]
                            self.ekf_states[a_idx] = result[1]
                            self.ekf_Ps[a_idx] = result[2]
                            self.ekf_initializeds[a_idx] = result[3]
                            self.ekf_Rs[a_idx] = result[4]
                        elif len(result) >= 6:
                            est_pos = result[0]
                            self.ekf_states[a_idx] = result[1]
                            self.ekf_Ps[a_idx] = result[2]
                            self.ekf_initializeds[a_idx] = result[3]
                            self.ekf_Qs[a_idx] = result[4]
                            self.ekf_Rs[a_idx] = result[5]
                        elif len(result) >= 4:
                            est_pos = result[0]
                            self.ekf_states[a_idx] = result[1]
                            self.ekf_Ps[a_idx] = result[2]
                            self.ekf_initializeds[a_idx] = result[3]
                        else:
                            est_pos = result[0]
                    else:
                        est_pos = result
                else:
                    # Fallback to trilateration
                    est_pos = LocalizationAlgorthimes.trilateration(measurements_list, chosen_anchors)
                
                all_est_spots.append({
                    'pos': (est_pos[0], est_pos[1]), 
                    'brush': self.agent_brushes[a_idx]
                })
                
                # Store estimated position for next step's observation
                self.agent_est_positions[a_idx] = [float(est_pos[0]), float(est_pos[1])]
                
                # Update errors for state/reward
                error = np.linalg.norm([est_pos[0] - true_pos[0], est_pos[1] - true_pos[1]])
                self.prev_errors[a_idx] = self.curr_errors[a_idx]
                self.curr_errors[a_idx] = error
                
                # Compute local energy and update energy accumulator
                calc = self.agent_energy_calculators[a_idx]
                calc.set_num_anchors(max(0, len(chosen_anchors)))
                calc.set_imu_enabled(use_imu)
                calc.set_uwb_enabled(agent_source.lower() in ("uwb", "both"))
                energy_res = calc.calculate_step(self.dt)
                energy_uJ = energy_res.total_power_mW * self.dt * 1000.0
                
                ALPHA_local = 0.6
                E_min_step = 16.5165  # 165.165 / 10
                E_max_step = 1300.0   # 13000.0 / 10
                
                # Check if client sent metrics
                if metrics is not None and "step_rewards" in metrics:
                    step_reward = metrics["step_rewards"][a_idx]
                    self.agent_step_rewards[a_idx] = step_reward
                else:
                    # Micro-step local calculation (fallback or intermediate)
                    error_improvement = self.prev_errors[a_idx] - error
                    r_precision = ALPHA_local * error_improvement
                    
                    # Normalize step energy
                    norm_energy = np.clip((energy_uJ - E_min_step) / (E_max_step - E_min_step), 0.0, 1.0)
                    r_energy = (1.0 - ALPHA_local) * (1.0 - norm_energy)
                    
                    step_reward = r_precision + r_energy
                    self.agent_step_rewards[a_idx] = step_reward
                
                if metrics is not None and "cumulative_rewards" in metrics:
                    self.agent_cumul_rewards[a_idx] = metrics["cumulative_rewards"][a_idx]
                
                # Collect per-agent metrics for the plots
                self.per_agent_errors[a_idx].append(float(error))
                self.per_agent_cumul_rewards[a_idx].append(self.agent_cumul_rewards[a_idx])
                self.per_agent_step_rewards[a_idx].append(step_reward)
                
                # Track which technology was used at this step
                # Map source to Y value: UWB=1, Both=2, IMU=3
                src_key = agent_source.lower()
                tech_y = {"uwb": 1, "both": 2, "imu": 3}.get(src_key, 1)
                self.per_agent_tech_source[a_idx].append(tech_y)
                
                # Count IMU and UWB steps
                if src_key in ("imu", "both"):
                    self.agent_imu_step_counts[a_idx] += 1
                if src_key in ("uwb", "both"):
                    self.agent_uwb_step_counts[a_idx] += 1
                
                # Record step energy (µJ) and cumulative energy (µJ)
                step_energy_uJ_val = energy_res.total_power_mW * self.dt * 1000.0
                self.per_agent_step_energy_uJ[a_idx].append(step_energy_uJ_val)
                self.per_agent_cumul_energy_uJ[a_idx].append(calc.cumulative_energy_uJ)
                
                # Update energy label in decision dock with power + step counts
                if a_idx < len(self.agent_decision_labels):
                    n_imu = self.agent_imu_step_counts[a_idx]
                    n_uwb = self.agent_uwb_step_counts[a_idx]
                    total_steps = self.current_step + 1
                    sim_t = self.agent_sim_times[a_idx]
                    agent_energy_mJ = calc.cumulative_energy_uJ * 1e-3
                    self.agent_decision_labels[a_idx]["energy"].setText(
                        f"⚡ {energy_res.total_power_mW:.1f} mW  |  "
                        f"UWB: {n_uwb} ticks  |  "
                        f"IMU: {n_imu} ticks  |  "
                        f"t={sim_t:.1f}s  |  "
                        f"Cumul: {agent_energy_mJ:.2f} mJ"
                    )

            except Exception as e:
                import traceback
                err = traceback.format_exc()
                print(f"Algorithm error on chosen anchors for agent {a_idx}: {e}\n{err}")

        # Shared time axis (use Agent 0's simulation time)
        self.metric_time_data.append(self.agent_sim_times[0])
        
        # Deque automatically handles trimming points to MAX_PLOT_POINTS
        
        # ── Throttled plot updates (limit to ~5 FPS) ─────────────────────
        now = _time.monotonic()
        should_redraw = (now - self._last_plot_update_time) >= self._PLOT_REFRESH_INTERVAL
        
        if should_redraw:
            self._last_plot_update_time = now
            
            # Flush buffered log messages to the console widget
            self._flush_log_buffer()
            
            # Update status label (only during redraws)
            sim_t = self.agent_sim_times[0]
            self.status_label.setText(f"t={sim_t:.2f}s  |  step {self.current_step}")
            
            # Update all per-agent curves
            for i in range(self.num_agents):
                time_list = list(self.metric_time_data)
                self.error_curves[i].setData(time_list, list(self.per_agent_errors[i]))
                self.reward_curves[i].setData(time_list, list(self.per_agent_cumul_rewards[i]))
                self.step_reward_curves[i].setData(time_list, list(self.per_agent_step_rewards[i]))
                self.cumul_energy_curves[i].setData(time_list, list(self.per_agent_cumul_energy_uJ[i]))
                
                # Update technology timeline with colored scatter dots
                if len(self.per_agent_tech_source[i]) > 0:
                    time_arr = np.array(time_list)
                    tech_arr = np.array(self.per_agent_tech_source[i])
                    
                    if not hasattr(self, '_tech_brushes'):
                        self._tech_brushes = {
                            1: pg.mkBrush(33, 150, 243, 200),
                            2: pg.mkBrush(171, 71, 188, 200),
                            3: pg.mkBrush(76, 175, 80, 200)
                        }
                    
                    brushes = [self._tech_brushes.get(int(v), self._tech_brushes[1]) for v in tech_arr]
                    self.tech_timeline_curves[i].setData(x=time_arr, y=tech_arr, symbol='o', symbolSize=6, symbolBrush=brushes)

            # Update baseline energy lines on the cumulative energy plot
            if len(self.metric_time_data) > 1:
                x_range = [self.metric_time_data[0], self.metric_time_data[-1]]
                time_range = np.array(x_range)
                self.imu_baseline_cumul_energy_curve.setData(
                    time_range, self.imu_only_power_mW * (time_range + self.dt) * 1000.0
                )
                self.uwb_baseline_cumul_energy_curve.setData(
                    time_range, self.uwb_only_power_mW * (time_range + self.dt) * 1000.0
                )

        # Map scatter plots (always update — cheap operation)
        self.true_pos_scatter.setData(all_true_spots)
        self.est_pos_scatter.setData(all_est_spots)
            
        # 4. ADVANCE STEP
        self.current_step += 1
        dt = self.dt
        for i in range(self.num_agents):
            self.agent_sim_times[i] += dt
            
        self.state_sent_for_step = False

        # Periodic garbage collection to free accumulated objects
        if self.current_step % 500 == 0:
            gc.collect()

    # ── Port / Source Configuration ─────────────────────────────────────

    def _on_port_changed(self, new_port: int):
        """Handle port spinner value change. Restarts server on new port."""
        if new_port == self._server_port:
            return
        self._server_port = new_port
        self.server.stop()
        self.server = AIGymServer(port=new_port)
        self.server.start()
        self.status_label.setText(f"Server restarted on port {new_port}")

    def _on_source_changed(self, text: str):
        """Handle measurement source combo change."""
        source_map = {"UWB": "uwb", "IMU": "imu", "Both": "both"}
        source = source_map.get(text, "uwb")
        self._measurement_source = source
        self.training_api.set_measurement_source(source)
        self.status_label.setText(f"Measurement source: {text}")
