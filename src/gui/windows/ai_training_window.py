import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFrame, QSplitter, QCheckBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPen, QColor
import pyqtgraph as pg

from src.api.ai_gym_server import AIGymServer
from src.core.localization.Localization_alghorthime import LocalizationAlgorthimes
from src.core.motion import MotionController
from src.core.uwb.uwb_devices import Tag, Position

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
        
        self.server = AIGymServer(port=5555)
        self.server.start()
        
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
        self.current_step = 0
        self.setup_ui()
        
        # Physics state
        self.state_sent_for_step = False
        self.ekf_states = [None] * self.num_agents
        self.ekf_Ps = [None] * self.num_agents
        self.ekf_initializeds = [False] * self.num_agents
        
        # Timer for stepping the environment
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.environment_step)
        self.timer.setInterval(1) # As fast as possible for AI training
        
        self.trajectory_points = []
        self._generate_trajectory()
        
        # Distributed start points for multi-agents
        # Now we make all N agents track the EXACT SAME point to follow a single tag
        self.agent_steps = [0] * self.num_agents
            
        self.refresh_base_plot()

    def _generate_trajectory(self):
        """Generates the static trajectory path for the AI environment to step over."""
        try:
            if self.main_app.movement_pattern.startswith("Custom:"):
                trajectory_name = self.main_app.movement_pattern.split(":", 1)[1]
                t_points = MotionController.load_custom_trajectory(trajectory_name)
                if t_points:
                    self.trajectory_points = [[p[0], p[1]] for p in t_points]
            else:
                side = 8
                period = (4 * side) / self.main_app.movement_speed if self.main_app.movement_speed > 0 else 10
                t_points = np.arange(0, period, self.main_app.dt) if hasattr(self.main_app, 'dt') and self.main_app.dt > 0 else np.linspace(0, period, 500)
                temp_tag = Tag(Position(0, 0))
                
                for t in t_points:
                    MotionController.update_tag_position(
                        tag=temp_tag,
                        movement_pattern=self.main_app.movement_pattern,
                        movement_speed=self.main_app.movement_speed,
                        t=t,
                        frequence=1/self.main_app.dt if hasattr(self.main_app, 'dt') and self.main_app.dt > 0 else 200,
                        point=self.main_app.point
                    )
                    self.trajectory_points.append([temp_tag.position.x, temp_tag.position.y])
        except Exception as e:
            import traceback
            from PyQt5.QtWidgets import QMessageBox
            err_msg = traceback.format_exc()
            print(f"Error generating trajectory: {e}\n{err_msg}")
            QMessageBox.warning(None, "Trajectory Gen Error", f"Could not generate training path: {e}")

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
        layout.addLayout(status_layout)
        
        # 2. Splitter for Map and Metrics
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Main Plot area
        from src.gui.widgets.plot_helpers import create_themed_plot
        self.plot_widget = create_themed_plot(title="AI Training Map", show_grid=True)
        self.plot_widget.setAspectLocked(True)
        self.splitter.addWidget(self.plot_widget)
        
        # Metrics area
        self.metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        
        self.reward_plot = create_themed_plot(title="Cumulative Reward", y_label="Reward", x_label="Step")
        self.loss_plot = create_themed_plot(title="Policy Loss", y_label="Loss", x_label="Step")
        self.entropy_plot = create_themed_plot(title="Entropy", y_label="Entropy", x_label="Step")
        
        # Initialize curves
        self.metric_steps = []
        self.metric_rewards = []
        self.metric_loss = []
        self.metric_entropy = []
        
        self.reward_curve = self.reward_plot.plot(pen='y')
        self.loss_curve = self.loss_plot.plot(pen='r')
        self.entropy_curve = self.entropy_plot.plot(pen='c')
        
        metrics_layout.addWidget(self.reward_plot)
        metrics_layout.addWidget(self.loss_plot)
        metrics_layout.addWidget(self.entropy_plot)
        
        self.splitter.addWidget(self.metrics_widget)
        self.metrics_widget.setVisible(False)
        self.splitter.setSizes([700, 300]) # Default ratio
        
        layout.addWidget(self.splitter, stretch=1)
        
        # Generate unique colors for each agent
        self.agent_colors = []
        for i in range(max(1, self.num_agents)):
            hue = int((i / max(1, self.num_agents)) * 360)
            color = QColor.fromHsl(hue, 255, 127) # Full saturation, 50% lightness
            self.agent_colors.append(color)
            
        # Plot items
        self.anchor_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('b'), brush=pg.mkBrush(0, 0, 255, 120))
        self.trajectory_line = pg.PlotDataItem(pen=pg.mkPen('k', width=2, style=Qt.DashLine))
        self.true_pos_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
        self.est_pos_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None)) # Brushes set per-point
        self.chosen_anchors_scatter = pg.ScatterPlotItem(size=25, pen=pg.mkPen('g', width=3), brush=pg.mkBrush(0, 255, 0, 50))
        
        self.plot_widget.addItem(self.trajectory_line)
        self.plot_widget.addItem(self.anchor_scatter)
        self.plot_widget.addItem(self.true_pos_scatter)
        self.plot_widget.addItem(self.est_pos_scatter)
        self.plot_widget.addItem(self.chosen_anchors_scatter) # Large hollow circles for chosen anchors
        
        # Removed connection lines to anchors based on user feedback to prevent visual cascade
            
        # 3. Controls
        controls_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶️ Start Training")
        self.btn_step = QPushButton("⏭ Step")
        self.btn_reset = QPushButton("🔄 Reset")
        
        self.cb_show_metrics = QCheckBox("Show Live Metrics")
        self.cb_show_metrics.toggled.connect(self.metrics_widget.setVisible)
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_step)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addWidget(self.cb_show_metrics)
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_reset = QPushButton("⏹️ Reset")
        
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_reset.clicked.connect(self.reset)
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_reset)
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
        if len(self.trajectory_points) == 0:
            self.status_label.setText("Error: No trajectory defined in main window!")
            return
            
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
            
        self.ekf_states = [None] * self.num_agents
        self.ekf_Ps = [None] * self.num_agents
        self.ekf_initializeds = [False] * self.num_agents
        
        if len(self.trajectory_points) > 0:
            self.agent_steps = [0] * self.num_agents
        
        # Reset Metrics
        self.metric_steps.clear()
        self.metric_rewards.clear()
        self.metric_loss.clear()
        self.metric_entropy.clear()
        self.reward_curve.setData([], [])
        self.loss_curve.setData([], [])
        self.entropy_curve.setData([], [])
        
        self.status_label.setText("Reset to beginning.")

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

    def get_current_true_poses(self):
        """Safely gets true position for all agents at current step"""
        if len(self.trajectory_points) == 0:
            return None
        return [self.trajectory_points[s % len(self.trajectory_points)] for s in self.agent_steps]

    def environment_step(self):
        """The core Loop: Send State -> Wait Action -> Compute -> Advance"""
        if not self.is_playing:
            return
            
        true_poses = self.get_current_true_poses()
        if true_poses is None:
            self.pause()
            self.status_label.setText("Simulation Finished.")
            return

        # 1. SEND STATE
        if not self.state_sent_for_step:
            all_states = []
            for a_idx in range(self.num_agents):
                true_pos = true_poses[a_idx]
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
                    
                state_dict = {
                    "agent_id": a_idx,
                    "step": self.current_step,
                    "timestamp": self.current_step * (self.main_app.dt if hasattr(self.main_app, 'dt') else 0.1),
                    "tag_position_gt": [float(true_pos[0]), float(true_pos[1]), 0.0],
                    "distances_measured": [float(measurements[a.id]) for a in self.anchors],
                    "los_conditions": los_conditions
                }
                all_states.append(state_dict)
            
            success = self.server.send_state(all_states)
            if success:
                self.state_sent_for_step = True
                self.status_label.setText(f"Step {self.current_step}: Sent state ({self.num_agents} agents). Waiting for action...")
            else:
                self.status_label.setText("Error: RL Client not connected.")
                return # Try again next tick
                
        # 2. WAIT FOR ACTION
        action_response = self.server.wait_for_action(timeout=0.01) # Non-blocking poll
        if action_response is None:
            return # Wait for next GUI tick
            
        all_action_indices, metrics = action_response
        
        # Parse metrics if any and update plots
        if metrics is not None:
            self.metric_steps.append(self.current_step)
            self.metric_rewards.append(metrics.get("reward", 0))
            self.metric_loss.append(metrics.get("policy_loss", 0))
            self.metric_entropy.append(metrics.get("entropy", 0))
            
            # Keep only last 1000 steps to prevent slowdowns
            if len(self.metric_steps) > 1000:
                self.metric_steps.pop(0)
                self.metric_rewards.pop(0)
                self.metric_loss.pop(0)
                self.metric_entropy.pop(0)
                
            self.reward_curve.setData(self.metric_steps, self.metric_rewards)
            self.loss_curve.setData(self.metric_steps, self.metric_loss)
            self.entropy_curve.setData(self.metric_steps, self.metric_entropy)
            
        # 3. APPLY ACTION (Compute Location for all agents)
        self.status_label.setText(f"Step {self.current_step}: Received actions")
        
        all_true_spots = []
        all_est_spots = []
        all_chosen_spots = []
        
        # In case action_indices is not a list of lists, wrap it
        if len(all_action_indices) > 0 and not isinstance(all_action_indices[0], (list, tuple)):
            all_action_indices = [all_action_indices] * self.num_agents
        
        for a_idx in range(self.num_agents):
            if a_idx >= len(all_action_indices):
                break
                
            action_indices = all_action_indices[a_idx]
            true_pos = true_poses[a_idx]
            measurements_list = []
            chosen_anchors = []
            
            all_true_spots.append({'pos': (true_pos[0], true_pos[1])})
            
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
                    all_chosen_spots.append({'pos': (anchor.position.x, anchor.position.y), 'data': 1})
            
            # Compute EKF for this agent
            temp_tag = Tag(Position(true_pos[0], true_pos[1]))
            try:
                est_pos, state, cov, initialized = LocalizationAlgorthimes.extended_kalman_filter(
                    measurements=measurements_list,
                    tag=temp_tag,
                    anchors=chosen_anchors,
                    ekf_state=self.ekf_states[a_idx],
                    ekf_P=self.ekf_Ps[a_idx],
                    ekf_initialized=self.ekf_initializeds[a_idx],
                    dt=self.main_app.dt if hasattr(self.main_app, 'dt') else 0.1
                )
                self.ekf_states[a_idx] = state
                self.ekf_Ps[a_idx] = cov
                self.ekf_initializeds[a_idx] = initialized
                all_est_spots.append({
                    'pos': (est_pos[0], est_pos[1]), 
                    'brush': pg.mkBrush(self.agent_colors[a_idx])
                })
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                print(f"Algorithm error on chosen anchors for agent {a_idx}: {e}\n{err}")

        # Highlight chosen anchors
        self.chosen_anchors_scatter.setData(all_chosen_spots)
        self.true_pos_scatter.setData(all_true_spots)
        self.est_pos_scatter.setData(all_est_spots)
            
        # 4. ADVANCE STEP
        self.current_step += 1
        for i in range(self.num_agents):
            self.agent_steps[i] += 1
        self.state_sent_for_step = False
