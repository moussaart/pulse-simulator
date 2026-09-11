import numpy as np
import gc
import time as _time
import inspect
import traceback
import psutil
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFrame, QSplitter, QCheckBox,
    QSpinBox, QComboBox, QDockWidget, QGridLayout, QGroupBox,
    QTextEdit, QScrollArea, QTabWidget
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPen, QColor, QFont
import pyqtgraph as pg
from collections import deque
from src.gui.widgets import ActionButton
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm
from src.core.parallel.gpu_backend import gpu_manager

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
        
        # Algorithm name — inherit from main app if available, else use default
        self.algorithm = getattr(self.main_app, 'algorithm', "Duty-Cycled UWB-IMU NA-AEKF")
        
        # Algorithm dispatch mapping (same as SimulationManager)
        self.algorithm_methods = Alghortimes_doc().get_algorithm_methods()
        self.algorithm_instances = {} # Cache for class-based algorithms
        
        # Configurable port (default 5555, user can change via UI before start)
        self._server_port = 5555
        self.server = AIGymServer(port=self._server_port)
        self.server.start()
        
        # AI Training Facade for enriched observation building
        self.training_api = AITrainingAPI(self.main_app)
        
        # Measurement source: inherit from main app's energy calculator config if available
        self._measurement_source = "uwb"
        if hasattr(self.main_app, 'energy_calculator'):
            config = self.main_app.energy_calculator.config
            imu_enabled = getattr(config, 'imu_enabled', False)
            uwb_disabled = getattr(config, 'uwb_disabled', False)
            if imu_enabled and not uwb_disabled:
                self._measurement_source = "both"
            elif imu_enabled and uwb_disabled:
                self._measurement_source = "imu"
            elif not uwb_disabled:
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
        
        # Per-agent last algorithm extra_data (duty-cycle state for observation)
        self.agent_algo_extra_data = []

        for _ in range(self.num_agents):
            # Inherit starting position from main app
            start_x, start_y = 0.0, 0.0
            if hasattr(self.main_app, 'tag') and hasattr(self.main_app.tag, 'position'):
                start_x = self.main_app.tag.position.x
                start_y = self.main_app.tag.position.y
                
            t = Tag(Position(start_x, start_y))
            t.imu_data_on = True
            self.agent_tags.append(t)
            self.agent_sim_times.append(0.0)
            
            # Dynamically size the initial state vector based on the algorithm
            is_duty_cycled = "Duty-Cycled" in self.algorithm or "Duty Cycled" in self.algorithm
            is_imu_uwb = "IMU-UWB AEKF" in self.algorithm or "IMU-UWB Adaptive EKF" in self.algorithm or "IMU assisted NLOS-Aware AEKF" in self.algorithm or "IMU Only" in self.algorithm
            
            expected_dim = 8 if is_duty_cycled else (6 if is_imu_uwb else 4)
            initial_state = np.zeros(expected_dim)
            initial_state[0:2] = [start_x, start_y]
            self.ekf_states.append(initial_state)
            
            if expected_dim == 8:
                self.ekf_Ps.append(np.diag([5.0, 5.0, 10.0, 10.0, 1.0, 1.0, 0.1, 0.1]))
            elif expected_dim == 6:
                self.ekf_Ps.append(np.diag([1.0, 1.0, 0.1, 0.1, 0.05, 0.05]))
            else:
                self.ekf_Ps.append(np.eye(expected_dim) * 5.0)
                
            self.ekf_initializeds.append(False)
            self.ekf_Qs.append(None)
            self.ekf_Rs.append(None)
            self.prev_errors.append(0.0)
            self.curr_errors.append(0.0)
            self.agent_est_positions.append(None)
            self.agent_energy_calculators.append(EnergyCalculator())
            self.agent_algo_extra_data.append({})
        
        # Per-agent adaptive EKF parameters
        self.adaptive_iekf_iteration_counts = [0] * self.num_agents
        self.adaptive_iekf_prev_Rs = [None] * self.num_agents
        self.adaptive_iekf_innovation_histories = [None] * self.num_agents
        
        # Per-agent cumulative rewards (server-side)
        self.agent_cumul_rewards = [0.0] * self.num_agents
        self.agent_step_rewards = [0.0] * self.num_agents
        
        # ── Time-based simulation parameters ──────────────────────────────
        self._tau_seconds = self.dt  # Inherit from property
        
        # Try to inherit frequencies from main app config if available
        self._imu_freq = 100.0
        self._uwb_freq = 10.0
        if hasattr(self.main_app, 'energy_calculator'):
            config = self.main_app.energy_calculator.config
            if hasattr(config, 'imu_frequency_hz'):
                self._imu_freq = float(config.imu_frequency_hz)
            if hasattr(config, 'uwb_frequency_hz'):
                self._uwb_freq = float(config.uwb_frequency_hz)
                
        self._walking_speed = self.movement_speed  # Inherit from property
        
        # Per-agent last measurement source (for energy in next observation)
        self.agent_last_sources = [self._measurement_source] * self.num_agents
        
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

    @property
    def anchors(self):
        return self.main_app.anchors

    @property
    def nlos_manager(self):
        return self.main_app.nlos_manager

    @property
    def trajectory_manager(self):
        return self.main_app.trajectory_manager

    @property
    def channel_model(self):
        return self.main_app.channel_conditions

    @property
    def los_aware_alpha(self):
        return getattr(self.main_app, 'los_aware_alpha', 0.5)

    @property
    def los_aware_beta(self):
        return getattr(self.main_app, 'los_aware_beta', 0.5)

    @property
    def los_aware_nlos_factor(self):
        return getattr(self.main_app, 'los_aware_nlos_factor', 100)

    @property
    def adaptive_iekf_mu(self):
        return getattr(self.main_app, 'adaptive_iekf_mu', 0.95)

    @property
    def adaptive_iekf_alpha(self):
        return getattr(self.main_app, 'adaptive_iekf_alpha', 0.3)

    @property
    def adaptive_iekf_xi(self):
        return getattr(self.main_app, 'adaptive_iekf_xi', 20)

    @property
    def adaptive_iekf_lambda_min(self):
        return getattr(self.main_app, 'adaptive_iekf_lambda_min', 0.1)

    @property
    def adaptive_iekf_lambda_max(self):
        return getattr(self.main_app, 'adaptive_iekf_lambda_max', 3.0)

    @property
    def adaptive_iekf_tau(self):
        return getattr(self.main_app, 'adaptive_iekf_tau', 0.95)

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
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
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
        
        # 2. Tab 1: Agent Dashboard Tab
        self.dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout(self.dashboard_tab)
        dashboard_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        scroll_grid = QGridLayout(scroll_content)
        scroll_grid.setContentsMargins(5, 5, 5, 5)
        scroll_grid.setSpacing(10)
        
        self.agent_decision_labels = []
        for i in range(self.num_agents):
            group = QGroupBox(f"🤖 Agent {i} Status")
            group.setStyleSheet("""
                QGroupBox { 
                    border: 1px solid #444; 
                    border-radius: 6px; 
                    margin-top: 10px; 
                    padding: 10px; 
                    background-color: #2b2b2b;
                } 
                QGroupBox::title { 
                    subcontrol-origin: margin; 
                    left: 10px; 
                    padding: 0 4px; 
                    color: #2196F3; 
                    font-weight: bold;
                    font-size: 11px;
                }
            """)
            g_layout = QVBoxLayout(group)
            g_layout.setSpacing(6)
            
            # Large technology badge showing current tech
            tech_badge = QLabel("📡 UWB")
            tech_badge.setAlignment(Qt.AlignCenter)
            tech_badge.setStyleSheet(
                "background-color: #1a3a5c; color: #2196F3; font-weight: bold; "
                "font-size: 12px; padding: 5px 10px; border-radius: 6px; "
                "border: 2px solid #2196F3;"
            )
            
            algo_label = QLabel("Algorithm: N/A")
            algo_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
            
            source_label = QLabel("Source: N/A")
            source_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 11px;")
            
            anchors_label = QLabel("Anchors: N/A")
            anchors_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 11px;")
            
            # Energy breakdown label
            energy_label = QLabel("Energy: N/A")
            energy_label.setStyleSheet("color: #FF5722; font-weight: bold; font-size: 11px;")
            
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
            scroll_grid.addWidget(group, i // 2, i % 2) # 2 columns max
            
        scroll_area.setWidget(scroll_content)
        dashboard_layout.addWidget(scroll_area)

        # 3. Console Logs Setup (inside Tab Widget instead of Dock)
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("background-color: black; color: #00FF00; font-family: Consolas, Monaco, monospace; font-size: 11px;")

        # 4. Splitter for Map and Tab Widget
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Main Plot area
        from src.gui.widgets.plot_helpers import create_themed_plot
        self.plot_widget = create_themed_plot(title="AI Training Map", show_grid=True)
        self.plot_widget.setAspectLocked(True)
        self.splitter.addWidget(self.plot_widget)
        
        # ── Localization Error ──────────────────────────────
        self.error_plot = create_themed_plot(title="Localization Error (m)", y_label="Error (m)", x_label="Time (s)")
        self.error_plot.addLegend(offset=(10, 10))
        
        # ── Technology Used Over Time (Multi-plot Grid) ──
        self.tech_timeline_widget = QWidget()
        self.tech_timeline_grid = QGridLayout(self.tech_timeline_widget)
        self.tech_timeline_grid.setContentsMargins(0, 0, 0, 0)
        self.tech_timeline_grid.setSpacing(5)
        
        self.tech_timeline_plots = []
        
        # ── Cumulative Reward ───────────────────────────────
        self.reward_plot = create_themed_plot(title="Cumulative Reward", y_label="Reward", x_label="Step")
        self.reward_plot.addLegend(offset=(10, 10))
        
        # ── Step Reward ─────────────────────────────────────
        self.step_reward_plot = create_themed_plot(title="Step Reward", y_label="Reward", x_label="Step")
        self.step_reward_plot.addLegend(offset=(10, 10))
        
        # ── Cumulative Energy (µJ) ─────────────────────────
        self.cumul_energy_plot = create_themed_plot(
            title="Cumulative Energy (µJ)",
            y_label="Energy (µJ)", x_label="Time (s)"
        )
        self.cumul_energy_plot.addLegend(offset=(10, 10))
        
        # Initialize per-agent metric data and curves
        MAX_PLOT_POINTS = 6000  # Retain 60s of history at 100Hz (1 full episode) to prevent UI freezing
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
        
        cols = 2 if self.num_agents >= 2 else 1
        for i in range(self.num_agents):
            hue = int((i / max(1, self.num_agents)) * 360)
            color = QColor.fromHsl(hue, 255, 127)
            pen = pg.mkPen(color, width=2)
            brush = pg.mkBrush(color.red(), color.green(), color.blue(), 30) # Add translucent fill
            lbl = f"Agent {i}"
            self.error_curves.append(self.error_plot.plot(pen=pen, name=lbl, fillLevel=0, brush=brush))
            self.reward_curves.append(self.reward_plot.plot(pen=pen, name=lbl, fillLevel=0, brush=brush))
            self.step_reward_curves.append(self.step_reward_plot.plot(pen=pen, name=lbl))
            self.cumul_energy_curves.append(self.cumul_energy_plot.plot(pen=pen, name=lbl, fillLevel=0, brush=brush))
            
            # Create a separate technology timeline plot for this agent
            plot = create_themed_plot(
                title=f"Agent {i} Resource Usage", y_label="", x_label="Time (s)"
            )
            y_axis = plot.getAxis('left')
            y_axis.setTicks([[(1, 'UWB'), (2, 'Both'), (3, 'IMU')]])
            y_axis.setWidth(80)  # Width of 80 to prevent text truncation on Windows
            plot.setYRange(0.5, 3.5)
            self.tech_timeline_plots.append(plot)
            
            row = i // cols
            col = i % cols
            self.tech_timeline_grid.addWidget(plot, row, col)
            
            # Technology timeline: scatter dots colored by source type
            tech_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None))
            plot.addItem(tech_scatter)
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
        
        # Create Tab Widget for the right side
        self.metrics_widget = QTabWidget()
        self.metrics_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #1e1e1e;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: #b1b1b1;
                border: 1px solid #3c3c3c;
                border-bottom-color: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QTabBar::tab:hover {
                background-color: #353535;
                color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #2196F3;
                border-bottom: 2px solid #2196F3;
            }
        """)
        
        # Precision Tab Layout
        precision_tab = QWidget()
        precision_layout = QVBoxLayout(precision_tab)
        precision_layout.setContentsMargins(5, 5, 5, 5)
        precision_layout.addWidget(self.error_plot)
        precision_layout.addWidget(self.tech_timeline_widget)
        
        # Reward Tab Layout
        reward_tab = QWidget()
        reward_layout = QVBoxLayout(reward_tab)
        reward_layout.setContentsMargins(5, 5, 5, 5)
        reward_layout.addWidget(self.reward_plot)
        reward_layout.addWidget(self.step_reward_plot)
        
        # Energy Tab Layout
        energy_tab = QWidget()
        energy_layout = QVBoxLayout(energy_tab)
        energy_layout.setContentsMargins(5, 5, 5, 5)
        energy_layout.addWidget(self.cumul_energy_plot)
        
        # Logs Tab Layout
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        logs_layout.setContentsMargins(5, 5, 5, 5)
        logs_layout.addWidget(self.console_text)
        
        # Add tabs
        self.metrics_widget.addTab(self.dashboard_tab, "🤖 Dashboard")
        self.metrics_widget.addTab(precision_tab, "🎯 Precision")
        self.metrics_widget.addTab(reward_tab, "🏆 Rewards")
        self.metrics_widget.addTab(energy_tab, "⚡ Energy")
        self.metrics_widget.addTab(logs_tab, "📜 Logs")
        
        self.splitter.addWidget(self.metrics_widget)
        self.metrics_widget.setVisible(True)
        self.splitter.setSizes([600, 400]) # Give more space to the map by default
        
        layout.addWidget(self.splitter, stretch=1)
        
        # Generate unique colors for each agent
        self.agent_colors = []
        self.agent_brushes = []
        for i in range(max(1, self.num_agents)):
            hue = int((i / max(1, self.num_agents)) * 360)
            # Increase saturation and lightness for better visibility on dark theme
            color = QColor.fromHsl(hue, 255, 180) 
            self.agent_colors.append(color)
            self.agent_brushes.append(pg.mkBrush(color))
            
        # Plot items
        self.anchor_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('b'), brush=pg.mkBrush(0, 0, 255, 120))
        # Change trajectory line color to white so it's visible on the dark background
        self.trajectory_line = pg.PlotDataItem(pen=pg.mkPen('w', width=2, style=Qt.DashLine))
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
        self.btn_play = ActionButton("▶️ Start Training", variant="success")
        self.btn_pause = ActionButton("⏸️ Pause", variant="primary")
        self.btn_reset = ActionButton("⏹️ Reset", variant="danger")
        
        self.cb_show_metrics = QCheckBox("Show Live Metrics & Selection")
        self.cb_show_metrics.setChecked(True)
        self.cb_show_metrics.toggled.connect(self.metrics_widget.setVisible)
        
        # Render Mode dropdown configuration
        render_label = QLabel("Render:")
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems([
            "Always",
            "Every N steps",
            "None"
        ])
        self.render_mode_combo.setCurrentIndex(0)
        self.render_mode_combo.setToolTip("Disable or throttle rendering to maximize RL training speed and stability.")
        
        render_n_label = QLabel("N:")
        self.render_n_spin = QSpinBox()
        self.render_n_spin.setRange(2, 50000)
        self.render_n_spin.setValue(100)
        self.render_n_spin.setEnabled(False)
        self.render_n_spin.setToolTip("Steps interval for throttled rendering.")
        
        def on_render_mode_changed(idx):
            self.render_n_spin.setEnabled(idx == 1)
            # If No Rendering is selected, hide the widgets completely to bypass paint pipeline
            if idx == 2:
                self.plot_widget.setVisible(False)
                self.metrics_widget.setVisible(False)
                self.cb_show_metrics.setChecked(False)
                self.cb_show_metrics.setEnabled(False)
            else:
                self.plot_widget.setVisible(True)
                self.cb_show_metrics.setEnabled(True)
                self.cb_show_metrics.setChecked(True)
                self.metrics_widget.setVisible(True)
                
        self.render_mode_combo.currentIndexChanged.connect(on_render_mode_changed)
        
        # Port configuration
        port_label = QLabel("Port:")
        self.port_spinner = QSpinBox()
        self.port_spinner.setRange(1024, 65535)
        self.port_spinner.setValue(self._server_port)
        self.port_spinner.setToolTip("TCP port for RL client connections. Change requires Connect.")
        
        # Reconnect/Connect button
        self.btn_reconnect = ActionButton("🔌 Connect", variant="secondary")
        self.btn_reconnect.clicked.connect(self.reconnect_server)
        
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_reset.clicked.connect(self.reset)
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addWidget(self.cb_show_metrics)
        controls_layout.addWidget(render_label)
        controls_layout.addWidget(self.render_mode_combo)
        controls_layout.addWidget(render_n_label)
        controls_layout.addWidget(self.render_n_spin)
        controls_layout.addWidget(port_label)
        controls_layout.addWidget(self.port_spinner)
        controls_layout.addWidget(self.btn_reconnect)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

    def refresh_base_plot(self):
        """Draws the static elements: Anchors, NLOS Zones, and Trajectory"""
        if not hasattr(self, '_nlos_plot_items'):
            self._nlos_plot_items = []
            
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
        
        # Clear existing NLOS plot items to prevent memory leaks
        for item in self._nlos_plot_items:
            try:
                self.plot_widget.removeItem(item)
            except:
                pass
        self._nlos_plot_items.clear()
        
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
                self._nlos_plot_items.append(zone_item)
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
            # Inherit starting position from main app
            start_x, start_y = 0.0, 0.0
            if hasattr(self.main_app, 'tag') and hasattr(self.main_app.tag, 'position'):
                start_x = self.main_app.tag.position.x
                start_y = self.main_app.tag.position.y
                
            tag = self.agent_tags[i]
            tag.position.x = start_x
            tag.position.y = start_y
            tag.velocity.x = 0.0
            tag.velocity.y = 0.0
            tag.acceleration.x = 0.0
            tag.acceleration.y = 0.0
            tag.imu_data.clear()
            tag.imu_simulator.reset()
            tag.last_update_time = None  # force first-sample logic in MotionController
            
            self.agent_sim_times[i] = 0.0
            
            is_duty_cycled = "Duty-Cycled" in self.algorithm or "Duty Cycled" in self.algorithm
            is_imu_uwb = "IMU-UWB AEKF" in self.algorithm or "IMU-UWB Adaptive EKF" in self.algorithm or "IMU assisted NLOS-Aware AEKF" in self.algorithm or "IMU Only" in self.algorithm
            
            expected_dim = 8 if is_duty_cycled else (6 if is_imu_uwb else 4)
            initial_state = np.zeros(expected_dim)
            initial_state[0:2] = [start_x, start_y]
            self.ekf_states[i] = initial_state
            
            if expected_dim == 8:
                self.ekf_Ps[i] = np.diag([5.0, 5.0, 10.0, 10.0, 1.0, 1.0, 0.1, 0.1])
            elif expected_dim == 6:
                self.ekf_Ps[i] = np.diag([1.0, 1.0, 0.1, 0.1, 0.05, 0.05])
            else:
                self.ekf_Ps[i] = np.eye(expected_dim) * 5.0
            self.ekf_Qs[i] = None
            self.ekf_Rs[i] = None
            self.prev_errors[i] = 0.0
            self.curr_errors[i] = 0.0
            self.agent_est_positions[i] = None
            self.agent_energy_calculators[i].reset_accumulator()
            self.adaptive_iekf_iteration_counts[i] = 0
            self.adaptive_iekf_prev_Rs[i] = None
            self.adaptive_iekf_innovation_histories[i] = None
            self.agent_cumul_rewards[i] = 0.0
            self.agent_step_rewards[i] = 0.0
            self.agent_last_sources[i] = self._measurement_source
            self.agent_algo_extra_data[i] = {}
        
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
            self.cumul_energy_curves[i].setData([], [])
            self.tech_timeline_curves[i].setData([], [])
        
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
        
        # Explicit clean up of PyQtGraph plot items to prevent reference leaks
        try:
            self.plot_widget.clear()
            self.error_plot.clear()
            self.reward_plot.clear()
            self.step_reward_plot.clear()
            self.cumul_energy_plot.clear()
            for plot in self.tech_timeline_plots:
                plot.clear()
        except Exception as e:
            print(f"[AITrainingWindow] Error cleaning up plot items: {e}")
            
        # Clean up GPU manager pre-allocations and reclaim CuPy GPU memory
        try:
            gpu_manager.clear_allocations()
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
            
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

        t_step_start = _time.perf_counter()

        # Track client connection state change
        is_conn = self.server.connected
        if is_conn != self._client_was_connected:
            self._client_was_connected = is_conn
            if is_conn:
                self.log_message(f"🟢 [Connection] Client connected to server on port {self._server_port}.")
            else:
                self.log_message(f"🔴 [Connection] Client disconnected.")

        # Build anchor positions array once for GPU batch check
        if not hasattr(self, '_anchor_positions_array') or self._anchor_positions_array.shape[0] != len(self.anchors):
            self._anchor_positions_array = np.array([[a.position.x, a.position.y] for a in self.anchors], dtype=float)

        if not hasattr(self, '_agent_measurements_cache'):
            self._agent_measurements_cache = {}
        if not hasattr(self, '_agent_los_cache'):
            self._agent_los_cache = {}

        # 1. SEND STATE (enriched observation via facade)
        if not self.state_sent_for_step:
            t_obs_start = _time.perf_counter()
            self._agent_measurements_cache.clear()
            self._agent_los_cache.clear()
            
            all_states = []
            for a_idx in range(self.num_agents):
                # Advance this agent's tag along the trajectory
                self._advance_agent_motion(a_idx)
                agent_tag = self.agent_tags[a_idx]
                true_pos = [agent_tag.position.x, agent_tag.position.y]

                # GPU-accelerated batch check for all anchors
                is_los_batch = self.channel_model.batch_update_los_conditions(
                    self._anchor_positions_array, agent_tag.position
                )
                los_conditions = is_los_batch.tolist()
                
                agent_measurements = {}
                agent_los = {}
                
                for i, anchor in enumerate(self.anchors):
                    is_los = bool(los_conditions[i])
                    agent_los[anchor.id] = is_los
                    
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
                    agent_measurements[anchor.id] = dist
                
                self._agent_measurements_cache[a_idx] = agent_measurements
                self._agent_los_cache[a_idx] = agent_los
                
                # Build enriched state via the facade
                agent_cumul_J = self.agent_energy_calculators[a_idx].cumulative_energy_uJ * 1e-6
                state_dict = self.training_api.build_step_observation(
                    agent_id=a_idx,
                    step=self.current_step,
                    dt=self.dt,
                    true_pos=true_pos,
                    est_pos=self.agent_est_positions[a_idx],
                    tag=agent_tag,
                    anchors=self.anchors,
                    measurements=agent_measurements,
                    los_conditions=los_conditions,
                    curr_error=self.curr_errors[a_idx],
                    prev_error=self.prev_errors[a_idx],
                    algorithm_name=self.algorithm,
                    movement_speed=self.movement_speed,
                    movement_pattern=self.movement_pattern,
                    channel_model=self.channel_model,
                    measurement_source=self.agent_last_sources[a_idx],
                    cumulative_energy_J=agent_cumul_J,
                    algo_extra_data=self.agent_algo_extra_data[a_idx],
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
                self._t_obs = _time.perf_counter() - t_obs_start
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
        
        # 3. APPLY ACTION (Compute Location for all agents)
        t_comp_start = _time.perf_counter()
        
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
                if action_indices is None:
                    action_indices = []
                elif not isinstance(action_indices, (list, tuple)):
                    action_indices = [action_indices]
                
                # Force to list of ints to prevent TypeError later
                try:
                    action_indices = [int(x) for x in action_indices if isinstance(x, (int, float, str))]
                except ValueError:
                    action_indices = []
                
                agent_algo_name = action_obj.get("filter", self.algorithm)
                if not agent_algo_name or agent_algo_name not in self.algorithm_methods:
                    agent_algo_name = self.algorithm
                agent_algo_name = str(agent_algo_name)
                    
                agent_source = action_obj.get("measurement_source", self._measurement_source)
                if not agent_source:
                    agent_source = self._measurement_source
                agent_source = str(agent_source)
            else:
                action_indices = action_obj if isinstance(action_obj, (list, tuple)) else [action_obj]
                agent_algo_name = str(self.algorithm)
                agent_source = str(self._measurement_source)
                
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
                        # Retrieve UWB distance from the cache to avoid recalculating
                        dist = self._agent_measurements_cache[a_idx].get(anchor.id)
                        if dist is not None:
                            measurements_list.append(dist)
                            chosen_anchors.append(anchor)
            
            # Compute location for this agent using current algorithm
            agent_tag = self.agent_tags[a_idx]
            agent_tag.position.x = true_pos[0]
            agent_tag.position.y = true_pos[1]
            use_imu = agent_source.lower() in ("imu", "both")
            
            # Dynamically resize state and P if algorithm dimension requires it
            expected_dim = 4
            if "Duty-Cycled" in agent_algo_name or "Duty Cycled" in agent_algo_name:
                expected_dim = 8  # [x, y, vx, vy, ax, ay, yaw, gyro_bias]
            elif "IMU-UWB AEKF" in agent_algo_name or "IMU-UWB Adaptive EKF" in agent_algo_name or "IMU assisted NLOS-Aware AEKF" in agent_algo_name or "IMU Only" in agent_algo_name:
                expected_dim = 6
            
            if len(self.ekf_states[a_idx]) != expected_dim:
                self.ekf_states[a_idx] = np.zeros(expected_dim)
                self.ekf_states[a_idx][0:2] = true_pos
                if expected_dim == 8:
                    self.ekf_Ps[a_idx] = np.diag([5.0, 5.0, 10.0, 10.0, 1.0, 1.0, 0.1, 0.1])
                elif expected_dim == 6:
                    self.ekf_Ps[a_idx] = np.diag([1.0, 1.0, 0.1, 0.1, 0.05, 0.05])
                else:
                    self.ekf_Ps[a_idx] = np.eye(expected_dim) * 5.0
                self.ekf_initializeds[a_idx] = False
                self.ekf_Qs[a_idx] = None
                self.ekf_Rs[a_idx] = None
            
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
                
                if method and inspect.isclass(method) and issubclass(method, BaseLocalizationAlgorithm):
                    if (agent_algo_name, a_idx) not in self.algorithm_instances:
                        self.algorithm_instances[(agent_algo_name, a_idx)] = method()
                        self.algorithm_instances[(agent_algo_name, a_idx)].initialize()
                    
                    # Apply agent's duty-cycle action (T_IMU / T_UWB) if present
                    if hasattr(self.algorithm_instances[(agent_algo_name, a_idx)], 'set_duty_cycle'):
                        t_imu_val = action_obj.get("t_imu", None) if isinstance(action_obj, dict) else None
                        t_uwb_val = action_obj.get("t_uwb", None) if isinstance(action_obj, dict) else None
                        if t_imu_val is not None and t_uwb_val is not None:
                            try:
                                self.algorithm_instances[(agent_algo_name, a_idx)].set_duty_cycle(
                                    cycle_length=t_imu_val + t_uwb_val,
                                    active_window=t_uwb_val,
                                )
                            except ValueError as ve:
                                self.log_message(f"⚠️ Invalid duty-cycle: {ve}")
                    
                    algo_instance = self.algorithm_instances[(agent_algo_name, a_idx)]
                    
                    # Ensure R is sized correctly to avoid dynamic anchor selection crashes
                    num_anchors = len(chosen_anchors)
                    if self.ekf_Rs[a_idx] is None or self.ekf_Rs[a_idx].shape[0] != num_anchors:
                        r_noise = 0.15
                        self.ekf_Rs[a_idx] = np.eye(num_anchors) * (r_noise**2)
                    
                    # Retrieve cached LOS bits (0 for LOS, 1 for NLOS) to avoid redundant geometry checks
                    cached_los_bits = [0 if self._agent_los_cache[a_idx].get(a.id, True) else 1 for a in chosen_anchors]
                    
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
                        is_los=cached_los_bits,
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
                    # Persist duty-cycle extra_data for next observation
                    if getattr(output, 'extra_data', None):
                        self.agent_algo_extra_data[a_idx] = output.extra_data
                elif method:
                    # Function-based algorithms (legacy static methods)
                    if "IMU-UWB AEKF" in agent_algo_name:
                        result = method(
                            measurements=measurements_list,
                            tag=agent_tag,
                            anchors=chosen_anchors,
                            state=self.ekf_states[a_idx],
                            P=self.ekf_Ps[a_idx],
                            initialized=self.ekf_initializeds[a_idx],
                            alpha=self.los_aware_alpha,
                            dt=self.dt,
                            zupt_threshold=0.08,
                            R=self.ekf_Rs[a_idx],
                            Q=self.ekf_Qs[a_idx]
                        )
                        (est_pos, self.ekf_states[a_idx], self.ekf_Ps[a_idx], 
                         self.ekf_initializeds[a_idx], self.ekf_Qs[a_idx],
                         self.ekf_Rs[a_idx]) = result
                    elif "Duty Cycled IMU-UWB AEKF" in agent_algo_name:
                        result = method(
                            measurements=measurements_list,
                            tag=agent_tag,
                            anchors=chosen_anchors,
                            state=self.ekf_states[a_idx],
                            P=self.ekf_Ps[a_idx],
                            initialized=self.ekf_initializeds[a_idx],
                            alpha=self.los_aware_alpha,
                            dt=self.dt,
                            zupt_threshold=0.08,
                            R=self.ekf_Rs[a_idx],
                            Q=self.ekf_Qs[a_idx]
                        )
                        (est_pos, self.ekf_states[a_idx], self.ekf_Ps[a_idx], 
                         self.ekf_initializeds[a_idx], self.ekf_Qs[a_idx],
                         self.ekf_Rs[a_idx]) = result
                    else:
                        # Generic trilateration etc.
                        est_pos = method(measurements_list, chosen_anchors)
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
                
                # Use algorithm's internal duty-cycle state for UWB if available
                extra = self.agent_algo_extra_data[a_idx] if self.agent_algo_extra_data else {}
                if extra and "uwb_window_open" in extra:
                    calc.set_uwb_enabled(extra["uwb_window_open"])
                else:
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
                
                # Track which technology was used at this step and count ticks
                extra = self.agent_algo_extra_data[a_idx] if self.agent_algo_extra_data else {}
                if extra and "uwb_window_open" in extra:
                    uwb_open = extra["uwb_window_open"]
                    tech_y = 2 if uwb_open else 3  # Both = 2, IMU = 3
                    
                    self.agent_imu_step_counts[a_idx] += 1
                    if uwb_open:
                        self.agent_uwb_step_counts[a_idx] += 1
                else:
                    src_key = agent_source.lower()
                    tech_y = {"uwb": 1, "both": 2, "imu": 3}.get(src_key, 1)
                    
                    if src_key in ("imu", "both"):
                        self.agent_imu_step_counts[a_idx] += 1
                    if src_key in ("uwb", "both"):
                        self.agent_uwb_step_counts[a_idx] += 1
                        
                self.per_agent_tech_source[a_idx].append(tech_y)
                
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
                err = traceback.format_exc()
                print(f"Algorithm error on chosen anchors for agent {a_idx}: {e}\n{err}")

        t_comp = _time.perf_counter() - t_comp_start

        # Shared time axis (use Agent 0's simulation time)
        self.metric_time_data.append(self.agent_sim_times[0])
        
        # ── Throttled/Configurable plot updates ─────────────────────────
        now = _time.monotonic()
        render_idx = self.render_mode_combo.currentIndex()
        if render_idx == 0:
            should_redraw = (now - self._last_plot_update_time) >= self._PLOT_REFRESH_INTERVAL
        elif render_idx == 1:
            should_redraw = (self.current_step % self.render_n_spin.value() == 0)
        else: # None (Headless No-Rendering Mode)
            should_redraw = False
        
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
                    self.tech_timeline_curves[i].setData(x=time_arr, y=tech_arr, symbol='o', symbolSize=6, brush=brushes)

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

        # Map scatter plots (only update if rendering is not completely disabled)
        if render_idx != 2:
            self.true_pos_scatter.setData(all_true_spots)
            self.est_pos_scatter.setData(all_est_spots)
        else:
            # If rendering is disabled, flush log buffer to stdout directly to prevent RAM growth
            self._flush_log_buffer()
            
        # 4. ADVANCE STEP
        self.current_step += 1
        dt = self.dt
        for i in range(self.num_agents):
            self.agent_sim_times[i] += dt
            
        self.state_sent_for_step = False

        t_step_total = _time.perf_counter() - t_step_start

        # Periodic resource & timing report (every 1000 steps)
        if self.current_step % 1000 == 0:
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
            
            # GPU memory report
            gpu_mem_report = ""
            try:
                import cupy
                free_mem = cupy.get_default_memory_pool().free_bytes()
                used_mem = cupy.get_default_memory_pool().used_bytes()
                total_mem = free_mem + used_mem
                gpu_mem_report = f" | GPU VRAM Used: {used_mem / (1024*1024):.1f} MB (Total Pool: {total_mem / (1024*1024):.1f} MB)"
            except:
                pass
                
            obs_time_val = getattr(self, '_t_obs', 0.0)
            self.log_message(
                f"📊 [Metrics Step {self.current_step}] "
                f"RAM: {ram_mb:.1f} MB | CPU: {cpu_percent:.1f}%{gpu_mem_report}\n"
                f"⏱️ Timers: Step: {t_step_total*1000.0:.2f}ms | Obs: {obs_time_val*1000.0:.2f}ms | Comp: {t_comp*1000.0:.2f}ms"
            )

        # Periodic garbage collection and CuPy GPU memory pool reclamation
        if self.current_step % 500 == 0:
            gc.collect()
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass

    def reconnect_server(self):
        """Handle manual server port reconnection."""
        new_port = self.port_spinner.value()
        self._server_port = new_port
        self.server.stop()
        self.server = AIGymServer(port=new_port)
        self.server.start()
        self.status_label.setText(f"Server restarted on port {new_port}")
        self.log_message(f"🔄 [System] Server restarted on port {new_port}")
