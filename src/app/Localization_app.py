from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, 
                             QScrollArea, QMenuBar, QMenu, QAction, QSplitter, QToolBar,
                             QSizePolicy, QPushButton, QCheckBox, QLabel, QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QIcon
import numpy as np
import time
import os
import datetime

# Fix for PyQtGraph alignment on high-DPI screens
import platform
import ctypes
try:
    if platform.system() == 'Windows' and float(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
except Exception:
    pass

from src.gui.panels.dockable_panel import PanelManager

# UWB and simulation imports
from src.core.uwb.uwb_devices import Anchor, Tag, Position
from src.core.uwb.channel_model import ChannelConditions, UWBParameters, PathLossParams
from src.core.uwb.config_loader import load_channel_configs

# Localization and Motion imports
from src.core.localization import CustomMessageBox
from src.core.motion import MotionController
from src.gui import IMUWindow, IMUData

# GUI module imports
from src.gui.theme import MODERN_STYLESHEET
from src.gui.managers.plot_manager import PlotManager
from src.gui.panels.control_panels import ControlPanelFactory
from src.gui.panels.timeline_widget import TimelineWidget
from src.gui.managers.event_handlers import EventHandler
from src.gui.managers.simulation_manager import SimulationManager
from src.gui.managers.nlos_manager import NLOSManager
from src.gui.managers.trajectory_manager import TrajectoryManager
from src.gui.managers.file_manager import FileManager
from src.gui.interactions.selection_manager import SelectionManager
from src.gui.interactions.anchor_selection_manager import AnchorSelectionManager
from src.gui.windows.nlos_config_window import NLOSConfigManager
from src.gui.windows.Nlos_aware_params_window import NLOSAwareParamsWindow
from src.gui.windows.Distance_plot_window import DistancePlotsWindow
from src.gui.windows.cir_window import CIRWindow

# AI Training API import
from src.api import TrainingDataAPI
from src.gui.windows.algorithm_creation_window import AlgorithmCreationWindow
from src.core.uwb.energy_model import EnergyCalculator, EnergyConfig
from src.gui.windows.energy_window import EnergyWindow

# Error handling
from src.core.error_handler import SimulationErrorHandler
from src.gui.widgets.error_overlay import ErrorOverlayWidget


class LocalizationApp(QMainWindow):
    """Main application window - now using modular components"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize window references early to avoid AttributeError
        self.distance_plots_window = None
        self.imu_window = None
        self.cir_window = None
        self.los_aware_window = None
        self.energy_window = None
        
        # Energy calculator (shared between panel and window)
        self.energy_calculator = EnergyCalculator()
        self.setWindowTitle("PULSE - Localization UWB Simulator")
        
        # Set window icon
        try:
             from src.utils.resource_loader import get_resource_path
             logo_path = get_resource_path(os.path.join('assets', 'logo.ico'))
             self.setWindowIcon(QIcon(logo_path))
        except Exception as e:
             print(f"Error setting window icon: {e}")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 768)
        self.setStyleSheet(MODERN_STYLESHEET)
        
        # Initialize basic parameters
        self.dt = 0.005  # Default 5ms time step
        self.point = (0.0, 0.0)
        self.movement_pattern = "Circular"
        self.movement_speed = 1.0
        self.algorithm = "Trilateration"
        
        # Initialize channel conditions
        self.channel_conditions = ChannelConditions()
        
        # Load custom channel configurations
        try:
            from src.utils.resource_loader import get_resource_path
            
            # Use safe resource path loader
            csv_path = get_resource_path(os.path.join('src','UWB data', 'uwb_channel_config_database.csv'))
            
            self.loaded_configs = load_channel_configs(csv_path)
            print(f"Loaded {len(self.loaded_configs)} channel configurations from CSV")
        except Exception as e:
            print(f"Failed to load channel configs: {e}")
            self.loaded_configs = {}
        
        # Initialize NLOS configuration manager
        self.nlos_config_manager = NLOSConfigManager()
        
        # Initialize managers (order matters for dependencies)
        self.plot_manager = PlotManager(self)
        self.nlos_manager = NLOSManager(self)
        self.trajectory_manager = TrajectoryManager(self)
        self.event_handler = EventHandler(self)
        self.simulation_manager = SimulationManager(self)
        self.file_manager = FileManager(self)
        self.selection_manager = SelectionManager(self)
        self.anchor_selection_manager = AnchorSelectionManager(self)
        
        # Initialize AI Training Data API (disabled by default)
        # Access via: app.training_api.enable_collection() / disable_collection()
        self.training_api = TrainingDataAPI(buffer_size=10000)
        


        
        # NLOS-Aware parameters
        self.los_aware_alpha = 0.5
        self.los_aware_beta = 0.5
        self.los_aware_nlos_factor = 100
        self.los_aware_probabilite_erreur = 0.1
        self.add_imperfections = False
        
        # Initialize panel widget references (will be set when panels are opened)
        # Anchor panel widgets
        self.x_coord = None
        self.y_coord = None
        self.anchor_list = None
        self.add_anchor_btn = None
        self.delete_anchor_btn = None
        # NLOS panel widgets
        self.nlos_widgets = None
        self.draw_mode_btn = None
        self.draw_polygon_btn = None
        # Channel panel widgets
        self.channel_combo = None
        self.tx_gain_spin = None
        self.rx_gain_spin = None
        self.noise_spin = None
        self.los_pl_spin = None
        self.los_shadow_spin = None
        self.noise_model_combo = None
        # Movement panel widgets
        self.pattern_combo = None
        self.timestep_slider = None
        self.timestep_value_label = None
        self.speed_slider = None
        self.speed_value_label = None
        self.draw_trajectory_btn = None
        self.import_traj_btn = None
        self.target_point_btn = None
        self.fp_x_spin = None
        self.fp_y_spin = None
        # Algorithm panel widgets
        self.algo_combo = None
        self.ma_window_spin = None

        # Status panel widgets
        self.status_display = None
        self.user_scrolling = False
        
        # Simulation settings (handled by TimelineWidget)
        
        # Timeline widget (for playback and settings)
        self.timeline_widget = None
        
        # Create UI
        self.setup_ui()
        
        # Initialize timer first (before initialize_simulation which may use it)
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_manager.update_simulation)
        
        # --- Error handling system ---
        self.error_handler = SimulationErrorHandler(self)
        self.error_overlay = ErrorOverlayWidget(self.centralWidget())
        self.error_handler.error_occurred.connect(self.error_overlay.show_error)
        self.error_overlay.restart_requested.connect(self._restart_after_error)
        self.simulation_manager.error_handler = self.error_handler
        
        # Initialize simulation
        self.initialize_simulation()
        
        # Start simulation in PAUSED state
        # self.timer.start(50)  # Don't start timer automatically
        self.pause_button.setChecked(True)
        self.pause_button.setText("▶️ Start")
        self.pause_button.setToolTip("Start Simulation")
        
        # Initialize the energy calculations with the default simulation parameters
        self.sync_energy_parameters()
    
    def setup_ui(self):
        """Setup the user interface with modular panel system"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create Custom Toolbar
        self.toolbar = self.create_custom_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # Create horizontal splitter for dock area + main content
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter, stretch=1)
        
        # Left dock panel container
        self.dock_container_widget = QWidget()
        self.dock_container_widget.setStyleSheet("background-color: #252526;")
        self.dock_container_widget.setMinimumWidth(0)
        self.dock_container_widget.setMaximumWidth(350)
        self.dock_layout = QVBoxLayout(self.dock_container_widget)
        self.dock_layout.setContentsMargins(4, 4, 4, 4)
        self.dock_layout.setSpacing(8)
        self.dock_layout.addStretch()  # Push panels to top
        self.main_splitter.addWidget(self.dock_container_widget)
        
        # Central content area (plots)
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(8, 8, 8, 8)
        central_layout.setSpacing(8)
        self.main_splitter.addWidget(central_widget)
        
        # Set splitter sizes (dock: 0, main: all)
        self.main_splitter.setSizes([0, 1000])
        self.main_splitter.setCollapsible(0, True)
        self.main_splitter.setCollapsible(1, False)
        
        # Setup plots in central area
        self.setup_plots(central_layout)
        
        # Initialize panel manager
        self.panel_manager = PanelManager(self.dock_layout, self)
        
        # Register all panels (creates them lazily)
        self.register_panels()
        
        # Connect mouse events
        self.connect_mouse_events()
    
    def create_menu_bar(self):
        """Create and return a QMenuBar widget for integration into toolbar"""
        menubar = QMenuBar()
        
        # === File Menu ===
        file_menu = menubar.addMenu("  File  ")
        
        save_action = QAction("💾  Save Configuration", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(lambda: self.file_manager.save_map_config())
        file_menu.addAction(save_action)
        
        load_action = QAction("📂  Load Configuration", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(lambda: self.file_manager.load_map_config())
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        clean_action = QAction("🗑️  Reset Map", self)
        clean_action.triggered.connect(self.on_clean_map)
        file_menu.addAction(clean_action)
        
        # === Simulation Setup Menu ===
        self.setup_menu = menubar.addMenu("  Setup  ")

        # === Analysis Menu (formerly Metrics + Windows) ===
        self.analysis_menu = menubar.addMenu("  Analysis  ")
        
        # Note: Actions are populated in populate_panels_menu to avoid duplication
        
        return menubar

    def create_custom_toolbar(self):
        """Create a custom toolbar widget with centered controls"""
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-bottom: 1px solid #3e3e42;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        toolbar_layout.setSpacing(10)
        
        # === Left: Menu Bar (File, Panels, View) ===
        # Create and add the menu bar directly to the layout
        self.menu_bar = self.create_menu_bar()
        toolbar_layout.addWidget(self.menu_bar)
        
        # Add Algorithm Indicator
        self.algo_indicator_btn = QPushButton(f"Algorithm: {self.algorithm}")
        self.algo_indicator_btn.setToolTip("Current Localization Algorithm - Click to Configure")
        self.algo_indicator_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                color: #d4d4d4;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
                border-color: #007acc;
            }
        """)
        self.algo_indicator_btn.clicked.connect(lambda: self.panel_manager.toggle_panel("algorithm"))
        toolbar_layout.addWidget(self.algo_indicator_btn)
        
        # Spacer
        toolbar_layout.addStretch()
        
        # === Center: Main Simulation Controls ===
        # Create a container for the center controls to ensure they stay centered
        center_container = QWidget()
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout, restart_btn, self.pause_button = ControlPanelFactory.create_main_controls()
        restart_btn.clicked.connect(self.restart_simulation)
        self.pause_button.clicked.connect(self.toggle_pause)
        
        center_layout.addLayout(main_layout)
        toolbar_layout.addWidget(center_container)
        
        # Spacer
        toolbar_layout.addStretch()
        
        # === Right: Sensors & View ===
        right_container = QWidget()
        right_layout = QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # Sensor Controls
        sensor_layout, imu_btn, distance_btn, self.ai_data_btn = ControlPanelFactory.create_sensor_controls()
        imu_btn.clicked.connect(self.toggle_imu_window)
        distance_btn.clicked.connect(self.toggle_distance_window)
        self.ai_data_btn.clicked.connect(self.toggle_ai_data_collection)
        right_layout.addLayout(sensor_layout)
        
        # Energy button
        from src.gui.widgets import ActionButton
        energy_btn = ActionButton("⚡ Energy", variant="secondary")
        energy_btn.setToolTip("Toggle Energy Consumption Panel")
        energy_btn.clicked.connect(lambda: self.panel_manager.toggle_panel("energy"))
        right_layout.addWidget(energy_btn)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background-color: #3e3e42;")
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        right_layout.addWidget(sep)
        
        # View Controls
        view_layout, self.path_toggle_btn, self.lines_toggle_btn = ControlPanelFactory.create_view_controls()
        self.path_toggle_btn.clicked.connect(self.toggle_path_visibility)
        self.lines_toggle_btn.clicked.connect(self.toggle_measurement_lines)
        right_layout.addLayout(view_layout)
        
        toolbar_layout.addWidget(right_container)
        
        return toolbar_widget
    
    def register_panels(self):
        """Register all available panels with the panel manager"""
        # Create panel factories - these return the content widgets
        
        # Anchor Configuration Panel
        def create_anchor_panel():
            anchor_group, anchor_widgets = ControlPanelFactory.create_anchor_config_panel()
            self.x_coord = anchor_widgets['x_coord']
            self.y_coord = anchor_widgets['y_coord']
            self.anchor_list = anchor_widgets['anchor_list']
            self.add_anchor_btn = anchor_widgets['add_anchor_btn']
            self.delete_anchor_btn = anchor_widgets['delete_anchor_btn']
            
            anchor_widgets['add_coord_btn'].clicked.connect(self.add_anchor_by_coords)
            self.add_anchor_btn.clicked.connect(self.toggle_add_anchor)
            self.delete_anchor_btn.clicked.connect(self.toggle_delete_anchor)
            return anchor_group
        
        # NLOS Panel
        def create_nlos_panel():
            nlos_group, self.nlos_widgets = ControlPanelFactory.create_nlos_panel()
            self.draw_mode_btn = self.nlos_widgets['draw_mode_btn']
            self.draw_polygon_btn = self.nlos_widgets['draw_polygon_btn']
            
            self.draw_mode_btn.clicked.connect(self.toggle_line_mode)
            self.draw_polygon_btn.clicked.connect(self.toggle_polygon_mode)
            self.nlos_widgets['delete_mode_btn'].clicked.connect(self.toggle_delete_mode)
            self.nlos_widgets['import_image_btn'].clicked.connect(self.nlos_manager.import_from_image)
            self.nlos_widgets['add_moving_btn'].clicked.connect(self.nlos_manager.toggle_moving_zone_placement)
            return nlos_group
        
        # UWB Channel Panel
        def create_channel_panel():
            channel_group, channel_widgets = ControlPanelFactory.create_uwb_channel_panel()
            self.channel_combo = channel_widgets['channel_combo']
            self.tx_gain_spin = channel_widgets['tx_gain_spin']
            self.rx_gain_spin = channel_widgets['rx_gain_spin']
            self.noise_spin = channel_widgets['noise_spin']
            self.los_pl_spin = channel_widgets['los_pl_spin']
            self.los_shadow_spin = channel_widgets['los_shadow_spin']
            self.noise_model_combo = channel_widgets['noise_model_combo']
            # New IEEE 802.15.3a S-V model widgets
            self.freq_decay_spin = channel_widgets['freq_decay_spin']
            self.cluster_decay_spin = channel_widgets['cluster_decay_spin']
            self.ray_decay_spin = channel_widgets['ray_decay_spin']
            self.toa_threshold_spin = channel_widgets['toa_threshold_spin']
            self.environment_combo = channel_widgets['environment_combo']
            self.fixed_noise_spin = channel_widgets['fixed_noise_spin']
            
            # Update environment combo with loaded configs if available
            if self.loaded_configs:
                self.environment_combo.clear()
                # Add loaded configs
                for name in self.loaded_configs.keys():
                    self.environment_combo.addItem(name)
                # Also ensure custom or default options if needed, but CSV takes precedence
            
            self.channel_combo.currentTextChanged.connect(self.on_channel_changed)
            self.tx_gain_spin.valueChanged.connect(self.on_tx_gain_changed)
            self.rx_gain_spin.valueChanged.connect(self.on_rx_gain_changed)
            self.noise_spin.valueChanged.connect(self.on_noise_figure_changed)
            self.los_pl_spin.valueChanged.connect(self.on_path_loss_changed)
            self.los_shadow_spin.valueChanged.connect(self.on_shadow_std_changed)
            self.noise_model_combo.currentTextChanged.connect(self.on_noise_model_changed)
            # Connect new S-V model widgets
            self.freq_decay_spin.valueChanged.connect(self.on_freq_decay_changed)
            self.cluster_decay_spin.valueChanged.connect(self.on_cluster_decay_changed)
            self.ray_decay_spin.valueChanged.connect(self.on_ray_decay_changed)
            self.toa_threshold_spin.valueChanged.connect(self.on_toa_threshold_changed)
            self.environment_combo.currentTextChanged.connect(self.on_environment_changed)
            self.fixed_noise_spin.valueChanged.connect(self.on_fixed_noise_changed)
            return channel_group
        
        # Movement Panel
        def create_movement_panel():
            movement_group, movement_widgets = ControlPanelFactory.create_movement_panel()
            self.pattern_combo = movement_widgets['pattern_combo']
            self.timestep_slider = movement_widgets['timestep_slider']
            self.timestep_value_label = movement_widgets['timestep_value_label']
            self.speed_slider = movement_widgets['speed_slider']
            self.speed_value_label = movement_widgets['speed_value_label']
            self.draw_trajectory_btn = movement_widgets['draw_trajectory_btn']
            self.import_traj_btn = movement_widgets['import_traj_btn']
            self.target_point_btn = movement_widgets['target_point_btn']
            self.fp_x_spin = movement_widgets.get('fp_x_spin')
            self.fp_y_spin = movement_widgets.get('fp_y_spin')
            
            self.pattern_combo.currentTextChanged.connect(self.update_movement_pattern)
            self.timestep_slider.valueChanged.connect(self.update_timestep_with_label)
            self.speed_slider.valueChanged.connect(self.update_speed_with_label)
            if self.fp_x_spin and self.fp_y_spin:
                self.fp_x_spin.valueChanged.connect(self.update_fixed_point_from_spinbox)
                self.fp_y_spin.valueChanged.connect(self.update_fixed_point_from_spinbox)
            self.draw_trajectory_btn.clicked.connect(self.trajectory_manager.toggle_trajectory_drawing)
            self.import_traj_btn.clicked.connect(self.trajectory_manager.import_csv_trajectory)
            
            if 'delete_traj_btn' in movement_widgets:
                movement_widgets['delete_traj_btn'].clicked.connect(self.trajectory_manager.delete_custom_trajectory_dialog)
                
                movement_widgets['open_traj_folder_btn'].clicked.connect(self.trajectory_manager.open_trajectory_folder)
                
            if 'play_exact_btn' in movement_widgets:
                movement_widgets['play_exact_btn'].clicked.connect(self.play_exact_trajectory)
            
            self.target_point_btn.clicked.connect(self.event_handler.toggle_target_point_mode)
            return movement_group
        
        # Algorithm Panel
        def create_algorithm_panel():
            algo_group, algo_widgets = ControlPanelFactory.create_algorithm_panel()
            self.algo_combo = algo_widgets['algo_combo']
            self.ma_window_spin = algo_widgets['ma_window_spin']

            
            self.algo_combo.currentTextChanged.connect(self.update_algorithm)
            self.ma_window_spin.valueChanged.connect(self.update_moving_average)
            
            if 'add_algo_btn' in algo_widgets:
                algo_widgets['add_algo_btn'].clicked.connect(self.open_algorithm_wizard)
            
            if 'delete_algo_btn' in algo_widgets:
                algo_widgets['delete_algo_btn'].clicked.connect(self.delete_selected_algorithm)
                
            if 'open_algo_folder_btn' in algo_widgets:
                algo_widgets['open_algo_folder_btn'].clicked.connect(self.open_algorithm_folder)

            return algo_group
        
        # Status Panel
        def create_status_panel():
            status_group, self.status_display = ControlPanelFactory.create_status_panel()
            self.user_scrolling = False
            self.status_display.verticalScrollBar().sliderPressed.connect(
                lambda: setattr(self, 'user_scrolling', True))
            self.status_display.verticalScrollBar().sliderReleased.connect(
                lambda: setattr(self, 'user_scrolling', False))
            return status_group
        

        
        # Energy Panel
        def create_energy_panel():
            energy_group, energy_widgets = ControlPanelFactory.create_energy_panel()
            self.energy_widgets = energy_widgets

            # Wire up auto-recalculation on any parameter change
            def _recalculate():
                self.energy_calculator.config.apply_hardware_profile(energy_widgets['device_combo'].currentText())
                self.energy_calculator.set_ranging_mode(energy_widgets['ranging_mode_combo'].currentText())
                self.energy_calculator.set_frequency(energy_widgets['uwb_freq_spin'].value())
                self.energy_calculator.set_num_anchors(energy_widgets['num_anchors_spin'].value())
                self.energy_calculator.set_imu_enabled(energy_widgets['imu_enabled_check'].isChecked())
                self.energy_calculator.config.battery_capacity_mAh = energy_widgets['battery_spin'].value()
                result = self.energy_calculator.calculate()
                # Update result labels
                avg_msg_e = (result.energy_per_tx_message_uJ + result.energy_per_rx_message_uJ) / 2
                energy_widgets['energy_msg_label'].setText(f"{avg_msg_e:.4f} µJ")
                energy_widgets['energy_ranging_label'].setText(f"{result.energy_per_ranging_uJ:.4f} µJ")
                energy_widgets['total_power_label'].setText(f"{result.total_power_mW:.2f} mW")
                if result.battery_life_days > 365:
                    energy_widgets['battery_life_label'].setText(f"{result.battery_life_days/365:.1f} years")
                elif result.battery_life_days > 1:
                    energy_widgets['battery_life_label'].setText(f"{result.battery_life_days:.1f} days")
                else:
                    energy_widgets['battery_life_label'].setText(f"{result.battery_life_hours:.1f} hours")
                # Refresh energy window if open
                if self.energy_window is not None and self.energy_window.isVisible():
                    self.energy_window.refresh()

            energy_widgets['device_combo'].currentTextChanged.connect(lambda: _recalculate())
            energy_widgets['ranging_mode_combo'].currentTextChanged.connect(lambda: _recalculate())
            energy_widgets['uwb_freq_spin'].valueChanged.connect(lambda: _recalculate())
            energy_widgets['num_anchors_spin'].valueChanged.connect(lambda: _recalculate())
            energy_widgets['imu_enabled_check'].toggled.connect(lambda: _recalculate())
            energy_widgets['battery_spin'].valueChanged.connect(lambda: _recalculate())
            energy_widgets['open_window_btn'].clicked.connect(self.toggle_energy_window)

            # Trigger initial calculation
            _recalculate()
            return energy_group

        # Register panels
        self.panel_manager.register_panel("anchor", "📍 Anchor Configuration", create_anchor_panel)
        self.panel_manager.register_panel("nlos", "🚧 NLOS Regions", create_nlos_panel)
        self.panel_manager.register_panel("channel", "📡 UWB Channel", create_channel_panel)
        self.panel_manager.register_panel("movement", "🚶 Movement Pattern", create_movement_panel)
        self.panel_manager.register_panel("algorithm", "🧮 Algorithm", create_algorithm_panel)
        self.panel_manager.register_panel("status", "📋 Status", create_status_panel)
        self.panel_manager.register_panel("energy", "⚡ Energy Consumption", create_energy_panel)
        
        # Pre-load essential panels that contain widgets needed for simulation
        # These panels are created but hidden, their widgets are accessible
        self._preload_essential_panels()
        
        # Populate panels menu with entries
        self.populate_panels_menu()
    
    def _preload_essential_panels(self):
        """Initialize essential panels so their widgets are available, but keep them hidden"""
        # Initialize panels without showing them
        self.panel_manager.init_panel_hidden("movement")     # speed_slider, pattern_combo
        self.panel_manager.init_panel_hidden("algorithm")    # algo_combo, ma_window_spin
        self.panel_manager.init_panel_hidden("status")       # status_display
        self.panel_manager.init_panel_hidden("channel")      # channel settings
        self.panel_manager.init_panel_hidden("anchor")       # anchor list
        self.panel_manager.init_panel_hidden("nlos")         # nlos widgets
        self.panel_manager.init_panel_hidden("nlos")         # nlos widgets
        
        # Keep dock area collapsed since no panels are visible
        self.main_splitter.setSizes([0, 1280])
    
    def populate_panels_menu(self):
        """Add panel entries to the Panels menu in organized categories"""
        
        # === Setup Menu Population ===
        
        # 1. Environment (Anchors & Map)
        self.setup_menu.addSection("Environment")
        
        # Anchors
        anchor_action = QAction("⚓  Anchors Configuration", self)
        anchor_action.triggered.connect(lambda: self.panel_manager.toggle_panel("anchor"))
        self.setup_menu.addAction(anchor_action)
        
        # NLOS Layout
        nlos_action = QAction("🚧  Obstacles & NLOS Zones", self)
        nlos_action.triggered.connect(lambda: self.panel_manager.toggle_panel("nlos"))
        self.setup_menu.addAction(nlos_action)
        
        # 2. Simulation Configuration
        self.setup_menu.addSection("Simulation Configuration")
        
        # Channel
        channel_action = QAction("📶  UWB Channel Model", self)
        channel_action.triggered.connect(lambda: self.panel_manager.toggle_panel("channel"))
        self.setup_menu.addAction(channel_action)
        
        # Motion
        motion_action = QAction("🏃  Tag Motion Controller", self)
        motion_action.triggered.connect(lambda: self.panel_manager.toggle_panel("movement"))
        self.setup_menu.addAction(motion_action)
        
        # Algorithm
        algo_action = QAction("🧮  Localization Algorithm", self)
        algo_action.triggered.connect(lambda: self.panel_manager.toggle_panel("algorithm"))
        self.setup_menu.addAction(algo_action)
        
        
        # === Simulation Menu Population ===
        



        # === Analysis Menu Population ===
        
        # 1. Real-time Data Visualization
        self.analysis_menu.addSection("Real-time Data")
        
        # IMU Data
        imu_metrics = QAction("🌀  IMU Data Stream", self)
        imu_metrics.triggered.connect(self.toggle_imu_window)
        self.analysis_menu.addAction(imu_metrics)
        
        # CIR Visualization
        cir_action = QAction("📊  Channel Impulse Response (CIR)", self)
        cir_action.triggered.connect(self.toggle_cir_window)
        self.analysis_menu.addAction(cir_action)
        
        # 2. Performance Analysis
        self.analysis_menu.addSection("Performance")
        
        # Ranging Errors
        dist_metrics = QAction("📏  Ranging Error Analysis", self)
        dist_metrics.triggered.connect(self.toggle_distance_window)
        self.analysis_menu.addAction(dist_metrics)
        
        # 3. Simulation Status & Logs
        self.analysis_menu.addSection("Simulation Status")
        
        # Status Log
        status_action = QAction("📝  Event Log", self)
        status_action.triggered.connect(lambda: self.panel_manager.toggle_panel("status"))
        self.analysis_menu.addAction(status_action)
        
        # Energy Consumption
        energy_action = QAction("⚡  Energy Consumption", self)
        energy_action.triggered.connect(lambda: self.panel_manager.toggle_panel("energy"))
        self.analysis_menu.addAction(energy_action)


    
    def setup_plots(self, layout):
        """Setup all plotting widgets"""
        # Create plot header with controls
        plot_header_layout = QHBoxLayout()
        
        # Title/Label
        plot_label = QLabel("Real-Time Localization View")
        plot_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #d0d0d0;")
        plot_header_layout.addWidget(plot_label)
        
        plot_header_layout.addStretch()
        
        # History Toggle
        self.history_toggle = QCheckBox("Show Full History")
        self.history_toggle.setToolTip("Switch between showing the last 100 points (tail) or the full trajectory history.")
        self.history_toggle.setChecked(False)
        self.history_toggle.setVisible(False)  # Hidden by default until simulation ends
        self.history_toggle.toggled.connect(self.toggle_history_mode)
        plot_header_layout.addWidget(self.history_toggle)
        
        # Export Button
        self.export_btn = QPushButton("💾 Export Data")
        self.export_btn.setToolTip("Export simulation data to files")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #094770;
            }
        """)
        self.export_btn.clicked.connect(self.export_simulation_data)
        self.export_btn.setVisible(False)  # Hidden by default until simulation ends
        plot_header_layout.addWidget(self.export_btn)
        
        layout.addLayout(plot_header_layout)

        # Create position plot
        self.position_plot = self.plot_manager.create_position_plot()
        # Remove internal title since we have an external header now
        self.position_plot.setTitle(None)
        layout.addWidget(self.position_plot)
        
        # Create plot items (trajectories, points, etc.)
        self.plot_items = self.plot_manager.create_plot_items(self.position_plot)
        
        # Create error plot
        self.error_plot_handler = self.plot_manager.create_error_plot()
        layout.addWidget(self.error_plot_handler.get_widget())
        
        # Create timeline widget
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.timeChanged.connect(self._on_timeline_scrub)
        # Connect settings signals
        self.timeline_widget.durationChanged.connect(self._on_duration_changed)
        self.timeline_widget.customDurationChanged.connect(self._on_custom_duration_changed)
        self.timeline_widget.recordingIntervalChanged.connect(self._on_snapshot_interval_changed)
        layout.addWidget(self.timeline_widget)
        
        # Add coordinate labels
        self.plot_manager.add_coordinate_labels(self.position_plot)
    
    
    def create_separator(self):
        """Create a vertical separator line"""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background-color: #404040;")
        separator.setFixedWidth(1)
        return separator
    
    def connect_mouse_events(self):
        """Connect mouse events to event handler"""
        self.position_plot.scene().sigMouseClicked.connect(self.event_handler.handle_plot_click)
        self.position_plot.scene().sigMouseMoved.connect(self.event_handler.handle_mouse_move)
        # Install event filter for mouse press/release events (needed for proper drag-and-drop)
        self.event_handler.install_event_filter()
    
    def initialize_simulation(self):
        """Initialize simulation state"""
        # Clear error plot data
        if hasattr(self, 'error_plot_handler'):
            self.error_plot_handler.clear_data()
            self.error_plot_handler.reset_plot()
        
        # Create tag with IMU data
        self.tag = Tag(Position(5, 5))
        self.tag.imu_data = IMUData()
        
        # Initialize Kalman filter states
        self.kf_state = np.array([5.0, 5.0, 0.0, 0.0])
        self.kf_P = np.eye(4) * 1.0
        self.kf_initialized = False
        self.aekf_R = None
        self.aekf_Q = None
        
        # Initialize IMU filter states
        self.imu_state = np.array([self.tag.position.x, self.tag.position.y, 0.0, 0.0, 0.0, 0.0])
        self.imu_P = np.diag([1.0, 1.0, 0.1, 0.1, 0.05, 0.05])
        
        # Initialize improved adaptive EKF parameters
        self.adaptive_iekf_window_size = 10
        self.adaptive_iekf_mu = 0.95
        self.adaptive_iekf_alpha = 0.3
        self.adaptive_iekf_xi = 20
        self.adaptive_iekf_lambda_min = 0.1
        self.adaptive_iekf_lambda_max = 3.0
        self.adaptive_iekf_tau = 0.95
        self.adaptive_iekf_iteration_count = 0
        self.adaptive_iekf_prev_R = None
        self.adaptive_iekf_innovation_history = None
        
        # Initialize IMU ZUPT EKF state
        self.imu_aekf_state = np.zeros(9)
        self.imu_aekf_P = np.eye(9) * 0.1
        self.imu_aekf_initialized = False
        
        # Create initial anchors if they don't exist
        if not hasattr(self, 'anchors') or len(self.anchors) == 0:
            self.anchors = [
                Anchor(Position(-8, -8)),
                Anchor(Position(8, -8)),
                Anchor(Position(0, 8))
            ]
            self.renormalize_anchors()
        
        # Clear trajectory histories
        self.plot_manager.clear_trajectory_histories(self.plot_items)
        
        # Initialize simulation time
        self.start_time = time.time()
        self.simulation_manager.start_time = self.start_time
        self.simulation_manager.last_update = self.start_time
        self.simulation_manager.simulation_time = 0
        self.simulation_manager.is_paused = True
        self.is_paused = True
        
        # Update trajectory patterns and plan
        self.trajectory_manager.update_trajectory_patterns()
        self.trajectory_manager.update_trajectory_plan()
        
        # Make sure pause button shows correct state
        if hasattr(self, 'pause_button'):
            self.pause_button.setChecked(True)
    
    def restart_simulation(self):
        """Restart simulation while preserving current parameters"""
        # Clear error plot
        self.error_plot_handler.clear_data()
        self.error_plot_handler.reset_plot()
        
        # Reset IMU window if open
        if self.imu_window is not None:
            self.imu_window.reset_plots()
        
        # Reset Distance plots window if exists
        if self.distance_plots_window is not None:
            self.distance_plots_window.reset_plots()
            
        # Reset CIR window if exists
        if self.cir_window is not None:
            self.cir_window.reset_plot()
        
        # Reset timeline widget
        if hasattr(self, 'timeline_widget') and self.timeline_widget:
            self.timeline_widget.reset()
        
        # Reset simulation manager (clears recorder, resets state)
        self.simulation_manager.reset()
        
        # Reset states
        self.initialize_simulation()
        
        # Set current speed from slider
        self.movement_speed = self.speed_slider.value() / 10.0
        
        # Update pause button state to show Start (paused by default)
        if hasattr(self, 'pause_button') and self.pause_button:
            self.pause_button.setChecked(True)
            self.pause_button.setText("▶️ Start")
            self.pause_button.setToolTip("Start Simulation")
        
        # Update elapsed time display
        # if self.elapsed_time_display:
        #     max_str = f"{self.simulation_manager.max_duration:.2f}s" if self.simulation_manager.max_duration else "∞"
        #     self.elapsed_time_display.setText(f"0.00s / {max_str}")
        
        # NOTE: Timer is NOT started here - simulation starts paused
    
    def _restart_after_error(self):
        """Full recovery after a simulation error."""
        # Clear error state
        self.error_handler.clear_error()
        self.error_overlay.hide()
        # Perform full simulation restart
        self.restart_simulation()
    
    def resizeEvent(self, event):
        """Keep error overlay sized to the central widget on window resize."""
        super().resizeEvent(event)
        if hasattr(self, 'error_overlay') and self.error_overlay and self.centralWidget():
            self.error_overlay.setGeometry(self.centralWidget().rect())
    
    # ==================== Event Handlers ====================
    
    def toggle_add_anchor(self):
        """Toggle anchor addition mode (mutually exclusive with other modes)"""
        from PyQt5.QtCore import Qt
        
        is_activating = self.add_anchor_btn.isChecked()
        
        if is_activating:
            # First disable all other modes (including delete mode)
            self.event_handler.disable_other_modes('adding_anchor')
            # Then activate add mode
            self.event_handler.adding_anchor = True
            self.add_anchor_btn.setText("Cancel Adding")
            self.position_plot.setCursor(Qt.CrossCursor)
            # Disable delete button while add mode is active (visual feedback)
            self.delete_anchor_btn.setEnabled(False)
        else:
            # Deactivate add mode
            self.event_handler.adding_anchor = False
            self.add_anchor_btn.setText("Click to Add")
            self.position_plot.setCursor(Qt.ArrowCursor)
            # Re-enable delete button
            self.delete_anchor_btn.setEnabled(True)
    
    def toggle_delete_anchor(self):
        """Toggle anchor deletion mode (mutually exclusive with other modes)"""
        from PyQt5.QtCore import Qt
        
        is_activating = self.delete_anchor_btn.isChecked()
        
        if is_activating:
            # First disable all other modes (including add mode)
            self.event_handler.disable_other_modes('deleting_anchor')
            # Then activate delete mode
            self.event_handler.deleting_anchor = True
            self.delete_anchor_btn.setText("Cancel Deleting")
            self.position_plot.setCursor(Qt.CrossCursor)
            # Disable add button while delete mode is active (visual feedback)
            self.add_anchor_btn.setEnabled(False)
        else:
            # Deactivate delete mode
            self.event_handler.deleting_anchor = False
            self.delete_anchor_btn.setText("Delete")
            self.position_plot.setCursor(Qt.ArrowCursor)
            # Re-enable add button
            self.add_anchor_btn.setEnabled(True)
    
    def toggle_line_mode(self):
        """Toggle line drawing mode (mutually exclusive with other modes)"""
        from PyQt5.QtCore import Qt
        import pyqtgraph as pg
        
        is_activating = self.draw_mode_btn.isChecked()
        
        if is_activating:
            # First disable all other modes
            self.event_handler.disable_other_modes('drawing_line')
            # Then activate line drawing mode
            self.event_handler.drawing_line = True
            self.draw_mode_btn.setText("✏️ Drawing Line...")
            self.position_plot.setCursor(Qt.CrossCursor)
            self.event_handler.current_line = pg.PlotDataItem(
                pen=pg.mkPen('r', width=2, style=Qt.DashLine))
            self.position_plot.addItem(self.event_handler.current_line)
            # Disable anchor buttons while drawing
            self.add_anchor_btn.setEnabled(False)
            self.delete_anchor_btn.setEnabled(False)
        else:
            # Deactivate line drawing mode
            self.event_handler.drawing_line = False
            self.draw_mode_btn.setText("🖊️ Draw Line")
            self.position_plot.setCursor(Qt.ArrowCursor)
            if self.event_handler.current_line:
                self.position_plot.removeItem(self.event_handler.current_line)
            self.event_handler.current_line = None
            self.event_handler.start_pos = None
            # Re-enable anchor buttons
            self.add_anchor_btn.setEnabled(True)
            self.delete_anchor_btn.setEnabled(True)
    
    def toggle_polygon_mode(self):
        """Toggle polygon drawing mode (mutually exclusive with other modes)"""
        from PyQt5.QtCore import Qt
        import pyqtgraph as pg
        
        is_activating = self.draw_polygon_btn.isChecked()
        
        if is_activating:
            # First disable all other modes
            self.event_handler.disable_other_modes('drawing_polygon')
            # Then activate polygon drawing mode
            self.event_handler.drawing_polygon = True
            self.draw_polygon_btn.setText("⬡ Drawing Polygon...")
            self.position_plot.setCursor(Qt.CrossCursor)
            self.event_handler.polygon_points = []
            self.event_handler.current_polygon = pg.PlotDataItem(
                pen=pg.mkPen('r', width=2, style=Qt.DashLine))
            self.position_plot.addItem(self.event_handler.current_polygon)
            # Disable anchor buttons while drawing
            self.add_anchor_btn.setEnabled(False)
            self.delete_anchor_btn.setEnabled(False)
        else:
            # Deactivate polygon drawing mode
            self.event_handler.drawing_polygon = False
            self.draw_polygon_btn.setText("⬡ Draw Polygon")
            self.position_plot.setCursor(Qt.ArrowCursor)
            if len(self.event_handler.polygon_points) >= 3:
                self.nlos_manager.create_polygon_from_lines(
                    self.event_handler.polygon_points)
            if self.event_handler.current_polygon:
                self.position_plot.removeItem(self.event_handler.current_polygon)
            self.event_handler.current_polygon = None
            self.event_handler.polygon_points = []
            # Re-enable anchor buttons
            self.add_anchor_btn.setEnabled(True)
            self.delete_anchor_btn.setEnabled(True)
    
    def toggle_path_visibility(self):
        """Toggle path visibility"""
        visible = self.plot_manager.toggle_path_visibility(self.plot_items)
        self.path_toggle_btn.setText("👣 Hide Path" if visible else "👣 Show Path")
    
    def toggle_measurement_lines(self):
        """Toggle measurement lines visibility"""
        visible = self.plot_manager.toggle_measurement_lines_visibility()
        self.lines_toggle_btn.setText("📏 Hide Lines" if visible else "📏 Show Lines")
    
    def toggle_pause(self):
        """Toggle simulation pause state with 3-state logic: Start/Pause/Continue"""
        # Check if simulation has ended
        if self.simulation_manager.simulation_ended:
            # Don't allow resume after simulation ends - user should reset
            return
        
        self.is_paused = self.pause_button.isChecked()
        self.simulation_manager.is_paused = self.is_paused
        
        if self.is_paused:
            # Pausing
            self.timer.stop()
            self.pause_button.setText("▶️ Continue")
            self.pause_button.setToolTip("Continue Simulation")
            
            # Show timeline if we have recorded data
            if self.simulation_manager.recorder.snapshot_count > 0:
                self.timeline_widget.show_timeline(
                    0,
                    self.simulation_manager.simulation_time
                )
        else:
            # Starting/Resuming
            self.timer.start(50)
            self.pause_button.setText("⏸️ Pause")
            self.pause_button.setToolTip("Pause Simulation")
            
            # Resume recording
            self.simulation_manager.recorder.resume_recording()
    
    def toggle_imu_window(self):
        """Toggle IMU data window"""
        if self.imu_window is None:
            self.imu_window = IMUWindow(self)
        
        if self.imu_window.isVisible():
            self.imu_window.hide()
        else:
            self.imu_window.show()
    
    def toggle_distance_window(self):
        """Toggle distance plots window"""
        if self.distance_plots_window is None:
            self.distance_plots_window = DistancePlotsWindow(self)
            self.distance_plots_window.update_anchors(self.anchors)
            self.distance_plots_window.show()
        else:
            if self.distance_plots_window.isVisible():
                self.distance_plots_window.hide()
            else:
                self.distance_plots_window.show()
                
    def toggle_cir_window(self):
        """Toggle CIR window"""
        if self.cir_window is None:
            self.cir_window = CIRWindow(self)
            self.cir_window.update_anchors(self.anchors)
            self.cir_window.show()
        else:
            if self.cir_window.isVisible():
                self.cir_window.hide()
            else:
                self.cir_window.show()
    
    def toggle_energy_window(self):
        """Toggle detailed energy consumption analysis window"""
        if self.energy_window is None:
            self.energy_window = EnergyWindow(self.energy_calculator, self)
        
        if self.energy_window.isVisible():
            self.energy_window.hide()
        else:
            self.energy_window.refresh()
            self.energy_window.show()
            
    def update_energy_displays(self):
        """Update energy consumption displays in the main panel and standalone window."""
        # Safety check: prevent crashes if called before GUI initialization finishes
        if not hasattr(self, 'energy_widgets'):
            return
            
        result = self.energy_calculator.calculate()
        
        # Update main panel inputs (which are now read-only displays)
        self.energy_widgets['uwb_freq_spin'].setValue(self.energy_calculator.config.uwb_frequency_hz)
        self.energy_widgets['num_anchors_spin'].setValue(self.energy_calculator.config.num_anchors)
        
        # Check if IMU is active (either enabled or uwb is disabled/IMU-only)
        imu_active = self.energy_calculator.config.imu_enabled or self.energy_calculator.config.uwb_disabled
        self.energy_widgets['imu_enabled_check'].setChecked(imu_active)
        
        # Update the main panel results labels
        self.energy_widgets['energy_msg_label'].setText(f"{result.energy_per_tx_message_uJ:.2f} µJ")
        self.energy_widgets['energy_ranging_label'].setText(f"{result.energy_per_ranging_uJ:.2f} µJ")
        self.energy_widgets['total_power_label'].setText(f"{result.total_power_mW:.2f} mW")
        self.energy_widgets['total_energy_label'].setText(f"{result.total_energy_consumed_J:.4f} J")
        
        # Battery life string
        if result.battery_life_days < 1:
            batt_str = f"{result.battery_life_hours:.1f} h"
        elif result.battery_life_days > 365:
            batt_str = f"{(result.battery_life_days / 365):.1f} y"
        else:
            batt_str = f"{result.battery_life_days:.1f} d"
        self.energy_widgets['battery_life_label'].setText(batt_str)
        
        # If open, refresh the detailed standalone window
        if self.energy_window is not None and self.energy_window.isVisible():
            self.energy_window.refresh()
    
    def toggle_ai_data_collection(self):
        """Toggle AI Training Data Collection and print data when stopped"""
        if self.ai_data_btn.isChecked():
            # Start collection
            self.training_api.select_data(
                channel=True,
                filter_outputs=True,
                ground_truth=True,
                imu=True
            )
            self.training_api.enable_collection()
            print("=" * 80)
            print("🤖 AI Training Data Collection: ENABLED")
            print("=" * 80)
            self.ai_data_btn.setText("🤖 Stop AI")
            self.ai_data_btn.setToolTip("Stop collection and print data")
        else:
            # Stop collection and print data
            self.training_api.disable_collection()
            
            # Get statistics
            stats = self.training_api.get_statistics()
            sample_count = stats.get('sample_count', 0)
            
            print("\n" + "=" * 80)
            print("🤖 AI Training Data Collection: STOPPED")
            print("=" * 80)
            print(f"\n📊 Collection Statistics:")
            print(f"   Samples collected: {sample_count}")
            
            if sample_count > 0:
                print(f"   Time range: {stats.get('time_range', (0, 0))}")
                print(f"   Duration: {stats.get('duration', 0):.2f}s")
                if 'mean_error' in stats:
                    print(f"   Mean error: {stats['mean_error']*1000:.2f}mm")
                    print(f"   Std error: {stats['std_error']*1000:.2f}mm")
                    print(f"   Max error: {stats['max_error']*1000:.2f}mm")
                if 'mean_snr_db' in stats:
                    print(f"   Mean SNR: {stats['mean_snr_db']:.1f}dB")
                print(f"   LOS percentage: {stats.get('los_percentage', 0):.1f}%")
                
                # Print detailed samples (last 3)
                samples = self.training_api.get_buffer()[-3:]
                print(f"\n" + "=" * 80)
                print(f"📝 Detailed Data for Last {len(samples)} Samples")
                print("=" * 80)
                
                for s in samples:
                    print(f"\n{'─' * 80}")
                    print(f"⏱️  TIMESTAMP: {s.timestamp:.4f}s")
                    print(f"{'─' * 80}")
                    
                    # TAG DATA
                    print(f"\n🏷️  TAG DATA:")
                    print(f"   Real Position (GT):     ({s.tag_position_gt[0]:.4f}, {s.tag_position_gt[1]:.4f})")
                    
                    if s.filter_outputs:
                        for algo_name, fo in s.filter_outputs.items():
                            print(f"   Algorithm:              {algo_name}")
                            print(f"   Estimated Position:     ({fo.estimated_position[0]:.4f}, {fo.estimated_position[1]:.4f})")
                            print(f"   Estimation Error:       {fo.estimation_error*1000:.2f} mm")
                            if fo.state_covariance is not None:
                                import numpy as np
                                cov = np.array(fo.state_covariance)
                                if cov.shape[0] >= 2:
                                    print(f"   Position Variance:      (σx²={cov[0,0]:.6f}, σy²={cov[1,1]:.6f})")
                    
                    # IMU DATA
                    if s.imu_acceleration:
                        print(f"\n🎢 IMU DATA:")
                        print(f"   Acceleration (m/s²):    ax={s.imu_acceleration[0]:.4f}, ay={s.imu_acceleration[1]:.4f}, az={s.imu_acceleration[2]:.4f}")
                    if s.imu_angular_velocity:
                        print(f"   Angular Vel (rad/s):    ωx={s.imu_angular_velocity[0]:.4f}, ωy={s.imu_angular_velocity[1]:.4f}, ωz={s.imu_angular_velocity[2]:.4f}")
                    
                    # PER-ANCHOR DATA
                    print(f"\n📡 ANCHOR DATA ({len(s.anchor_ids)} anchors):")
                    print(f"   {'ID':<6} {'Position':<20} {'Dist(meas)':<12} {'Dist(true)':<12} {'Error':<10} {'LOS/NLOS':<10} {'SNR(dB)':<10} {'PathLoss':<10}")
                    print(f"   {'-'*6} {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
                    
                    for i, anchor_id in enumerate(s.anchor_ids):
                        # Anchor position
                        if i < len(s.anchor_positions):
                            pos = s.anchor_positions[i]
                            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
                        else:
                            pos_str = "N/A"
                        
                        # Measured distance
                        dist_meas = s.distances_measured[i] if i < len(s.distances_measured) else 0
                        
                        # True distance
                        dist_true = s.distances_true[i] if i < len(s.distances_true) else 0
                        
                        # Distance error
                        dist_err = (dist_meas - dist_true) * 1000  # in mm
                        
                        # LOS/NLOS
                        is_los = s.los_conditions[i] if i < len(s.los_conditions) else True
                        los_str = "LOS" if is_los else "NLOS"
                        
                        # Channel data
                        if i < len(s.channel_data):
                            cd = s.channel_data[i]
                            snr_str = f"{cd.snr_db:.1f}"
                            pl_str = f"{cd.path_loss_db:.1f}"
                        else:
                            snr_str = "N/A"
                            pl_str = "N/A"
                        
                        print(f"   {anchor_id:<6} {pos_str:<20} {dist_meas:<12.4f} {dist_true:<12.4f} {dist_err:<10.1f}mm {los_str:<10} {snr_str:<10} {pl_str:<10}")
                
                # Export to file
                export_path = "ai_training_data.npz"
                self.training_api.export_to_file(export_path)
                print(f"\n💾 Data exported to: {export_path}")
            
            print("=" * 80 + "\n")
            
            # Reset button
            self.ai_data_btn.setText("🤖 AI Data")
            self.ai_data_btn.setToolTip("Print/Export AI Training Data")
    
    
    def toggle_delete_mode(self):
        """Toggle delete mode for NLOS zones"""
        is_active = self.nlos_widgets['delete_mode_btn'].isChecked()
        
        if is_active:
            self.event_handler.disable_other_modes('deleting_zone')
            self.event_handler.deleting_zone = True
            self.nlos_widgets['delete_mode_btn'].setText("Click Zone to Delete")
            self.position_plot.setCursor(Qt.ForbiddenCursor)
        else:
            self.event_handler.deleting_zone = False
            self.nlos_widgets['delete_mode_btn'].setText("🗑️ Delete")
            self.position_plot.setCursor(Qt.ArrowCursor)
    
    def clear_nlos_zones(self):
        """Clear all NLOS zones (Legacy method, kept if needed later)"""
        if self.show_question_dialog('Clear Regions',
                                      'Are you sure you want to remove all NLOS regions?'):
            self.nlos_manager.clear_nlos_zones()
    
    def on_clean_map(self):
        """Clean the map"""
        if self.show_question_dialog('Clean Map',
                                      'Are you sure you want to remove all anchors and NLOS zones?'):
            self.file_manager.clean_map()
    
    # ==================== Parameter Updates ====================
    
    def update_fixed_point_from_spinbox(self):
        """Update target coordinates when spinbox values change"""
        if hasattr(self, 'fp_x_spin') and hasattr(self, 'fp_y_spin') and self.fp_x_spin and self.fp_y_spin:
            x = self.fp_x_spin.value()
            y = self.fp_y_spin.value()
            self.point = (x, y)
            if hasattr(self, 'plot_items') and 'target_point_marker' in self.plot_items:
                self.plot_items['target_point_marker'].setData([x], [y])
            
            # If currently in Fixed Point pattern, update the plan
            if hasattr(self, 'pattern_combo') and self.pattern_combo.currentText() == "Fixed Point":
                if hasattr(self, 'trajectory_manager'):
                    self.trajectory_manager.update_trajectory_plan()

    def update_movement_pattern(self, pattern):
        """Update movement pattern"""
        self.movement_pattern = pattern
        self.exact_trajectory_mode = False # Reset exact mode on pattern change
        if not hasattr(self, 'tag'):
            self.tag = Tag(Position(5, 5))
            self.tag.imu_data = IMUData()
        
        # Only update trajectory plan if initialized
        if hasattr(self, 'trajectory_manager'):
            self.trajectory_manager.update_trajectory_plan()
    
    def play_exact_trajectory(self):
        """Play exact trajectory based on time step without looping"""
        if not self.movement_pattern.startswith("Custom:"):
            self.show_error_message("Error", "Please select a custom trajectory first.")
            return
            
        traj_name = self.movement_pattern[7:]
        points = MotionController.load_custom_trajectory(traj_name)
        if not points:
            self.show_error_message("Error", f"Could not load trajectory {traj_name}.")
            return
            
        self.exact_trajectory_mode = True
        
        auto_duration = len(points) * self.dt
        
        # Change duration dropdown to custom
        if hasattr(self, 'timeline_widget'):
            index = self.timeline_widget.duration_combo.findText("Custom")
            if index >= 0:
                self.timeline_widget.duration_combo.setCurrentIndex(index)
            self.timeline_widget.custom_duration_spin.setValue(int(auto_duration))
            
        self.simulation_manager.set_duration(auto_duration)
        
        self.restart_simulation()
        
        if self.is_paused:
            self.toggle_pause()
    
    def update_speed_with_label(self):
        """Update speed and label"""
        speed = self.speed_slider.value() / 10.0
        self.speed_value_label.setText(f"{speed:.1f} m/s")
        self.movement_speed = speed
    
    def sync_energy_parameters(self):
        """Syncs the energy calculator configuration with current UI and simulation state, then updates displays."""
        if not hasattr(self, 'energy_calculator') or not hasattr(self, 'energy_widgets'):
            return
            
        # Update frequency based on current time step (dt)
        if hasattr(self, 'dt') and self.dt > 0:
            self.energy_calculator.config.uwb_frequency_hz = 1.0 / self.dt
            
        # Update number of anchors
        if hasattr(self, 'anchors'):
            self.energy_calculator.config.num_anchors = len(self.anchors)
            
        # Update algorithm flag
        if hasattr(self, 'algorithm'):
            algo = self.algorithm.lower()
            if "imu only" in algo or "imu-only" in algo:
                self.energy_calculator.config.uwb_disabled = True
                self.energy_calculator.config.imu_enabled = True
            elif "hybrid" in algo or "aekf" in algo or "ekf" in algo:
                self.energy_calculator.config.uwb_disabled = False
                self.energy_calculator.config.imu_enabled = True
            else:
                self.energy_calculator.config.uwb_disabled = False
                self.energy_calculator.config.imu_enabled = False
                
        self.update_energy_displays()

    def update_timestep_with_label(self):
        """Update time step and label"""
        timestep_ms = self.timestep_slider.value()
        self.dt = timestep_ms / 1000.0
        frequency = 1000.0 / timestep_ms
        self.timestep_value_label.setText(f"{timestep_ms} ms ({frequency:.1f} Hz)")
        self.sync_energy_parameters()
    
    def update_algorithm(self, algorithm):
        """Update localization algorithm"""
        self.algorithm = algorithm
        self.sync_energy_parameters()
        
        # Update toolbar indicator if it exists
        if hasattr(self, 'algo_indicator_btn'):
            self.algo_indicator_btn.setText(f"Algorithm: {algorithm}")
        
        # Show/hide NLOS-Aware parameters window
        if "NLOS-Aware AEKF" in algorithm:
            if self.los_aware_window is None:
                self.los_aware_window = NLOSAwareParamsWindow()
                self.los_aware_window.params_changed.connect(self.update_los_aware_params)
            self.los_aware_window.show()
        elif self.los_aware_window is not None:
            self.los_aware_window.hide()
        
        # Reset states
        self.kf_state = np.array([self.tag.position.x, self.tag.position.y, 0.0, 0.0])
        self.kf_P = np.eye(4) * 1.0
        self.kf_initialized = False
        self.aekf_R = None
        self.aekf_Q = None
        
        # Clear error data
        self.error_plot_handler.clear_data()
    
    def update_moving_average(self):
        """Update moving average window"""
        pass  # Implementation handled by error plot handler
    
    def update_los_aware_params(self, alpha, beta, nlos_factor, error_prob, add_imperfections):
        """Update NLOS-Aware parameters"""
        self.los_aware_alpha = alpha
        self.los_aware_beta = beta
        self.los_aware_nlos_factor = nlos_factor
        self.los_aware_probabilite_erreur = error_prob
        self.add_imperfections = add_imperfections
    
    def on_channel_changed(self, channel):
        """Handle channel changes"""
        try:
            self.channel_conditions.set_uwb_channel(int(channel))
            self.restart_simulation()
        except ValueError as e:
            self.show_error_message("Channel Error", str(e))
    
    def on_tx_gain_changed(self, gain):
        """Handle TX gain changes"""
        new_params = UWBParameters(
            tx_antenna_gain_dbi=gain,
            rx_antenna_gain_dbi=self.channel_conditions.uwb_params.rx_antenna_gain_dbi,
            noise_figure_db=self.channel_conditions.uwb_params.noise_figure_db
        )
        self.channel_conditions.update_uwb_parameters(new_params)
    
    def on_rx_gain_changed(self, gain):
        """Handle RX gain changes"""
        new_params = UWBParameters(
            tx_antenna_gain_dbi=self.channel_conditions.uwb_params.tx_antenna_gain_dbi,
            rx_antenna_gain_dbi=gain,
            noise_figure_db=self.channel_conditions.uwb_params.noise_figure_db
        )
        self.channel_conditions.update_uwb_parameters(new_params)
    
    def on_noise_figure_changed(self, noise):
        """Handle noise figure changes"""
        new_params = UWBParameters(
            tx_antenna_gain_dbi=self.channel_conditions.uwb_params.tx_antenna_gain_dbi,
            rx_antenna_gain_dbi=self.channel_conditions.uwb_params.rx_antenna_gain_dbi,
            noise_figure_db=noise
        )
        self.channel_conditions.update_uwb_parameters(new_params)
    
    def on_path_loss_changed(self, pl_exp):
        """Handle path loss exponent changes"""
        new_params = PathLossParams(
            path_loss_exponent=pl_exp,
            reference_loss_db=self.channel_conditions.los_path_loss_params.reference_loss_db,
            shadow_fading_std=self.channel_conditions.los_path_loss_params.shadow_fading_std
        )
        self.channel_conditions.update_path_loss_parameters(new_params)
    
    def on_shadow_std_changed(self, shadow_std):
        """Handle shadow fading changes"""
        new_params = PathLossParams(
            path_loss_exponent=self.channel_conditions.los_path_loss_params.path_loss_exponent,
            reference_loss_db=self.channel_conditions.los_path_loss_params.reference_loss_db,
            shadow_fading_std=shadow_std
        )
        self.channel_conditions.update_path_loss_parameters(new_params)
    
    def on_noise_model_changed(self, model_name):
        """Handle noise model changes (all parameters are dynamic)"""
        self.channel_conditions.set_noise_model(model_name)
    
    def on_freq_decay_changed(self, kappa):
        """Handle frequency decay factor (κ) changes"""
        self.channel_conditions.los_path_loss_params.frequency_decay_factor = kappa
        self.channel_conditions.current_path_loss_params.frequency_decay_factor = kappa
    
    def on_cluster_decay_changed(self, gamma_ns):
        """Handle S-V cluster decay constant (Γ) changes - input in ns"""
        gamma_s = gamma_ns * 1e-9  # Convert ns to seconds
        self.channel_conditions.cluster_decay = gamma_s
        self.channel_conditions.los_sv_params.cluster_decay = gamma_s
    
    def on_ray_decay_changed(self, gamma_ns):
        """Handle S-V ray decay constant (γ) changes - input in ns"""
        gamma_s = gamma_ns * 1e-9  # Convert ns to seconds
        self.channel_conditions.ray_decay = gamma_s
        self.channel_conditions.los_sv_params.ray_decay = gamma_s
    
    def on_toa_threshold_changed(self, threshold):
        """Handle ToA detection threshold changes"""
        self.channel_conditions.detection_threshold_factor = threshold
    
    def on_environment_changed(self, env_name):
        """Handle IEEE 802.15.3a environment preset changes (CM1-CM4)"""
        env_config = None
        los_params = None
        
        # 1. Check loaded configs (CSV) - these are now EnvironmentConfig objects
        if hasattr(self, 'loaded_configs') and env_name in self.loaded_configs:
            env_config = self.loaded_configs[env_name]
            
        # 2. Fallback to hardcoded defaults (wrapped in EnvironmentConfig behavior)
        if env_config is None:
            from src.core.uwb.uwb_types import CM1_LOS_0_4M, CM2_NLOS_0_4M, CM3_NLOS_4_10M, CM4_EXTREME_NLOS, EnvironmentConfig
            
            # For backward compatibility, map old names to partial configs
            # Ideally we should define these properly as EnvironmentConfig in uwb_types.py
            env_map = {
                "CM1 (LOS 0-4m)": CM1_LOS_0_4M,
                "CM2 (NLOS 0-4m)": CM2_NLOS_0_4M,
                "CM3 (NLOS 4-10m)": CM3_NLOS_4_10M,
                "CM4 (Extreme NLOS)": CM4_EXTREME_NLOS
            }
            
            if env_name in env_map:
                # If these are just SVModelParams, we treat them as LOS params for simplicity
                # or create a dummy config
                params = env_map[env_name]
                # Default behavior: use this params for both LOS/NLOS or specific
                env_config = EnvironmentConfig(name=env_name, los_params=params, nlos_params=params)

        if env_config:
            # We have an EnvironmentConfig with potentially different LOS/NLOS params
            # Update the ChannelConditions with BOTH sets
            self.channel_conditions.los_sv_params = env_config.los_params
            self.channel_conditions.nlos_sv_params = env_config.nlos_params
            
            # Also update the current active params based on current state (usually starts as LOS)
            # Typically user selection implies resetting to this environment's default (LOS) state
            # until a zone is entered.
            self.channel_conditions.current_sv_params = env_config.los_params
            
            # Update Path Loss Params (derived from LOS params usually, or we should have separate PL params)
            # SVModelParams has 'path_loss_exponent' and 'shadow_fading_std'
            
            # LOS Path Loss
            self.channel_conditions.los_path_loss_params.path_loss_exponent = env_config.los_params.path_loss_exponent
            self.channel_conditions.los_path_loss_params.shadow_fading_std = env_config.los_params.shadow_fading_std
            
            # We should also update NLOS path loss params if possible, but ChannelModel might not expose them directly
            # The ChannelModel switches to 'active_zone.path_loss_params' or generates them.
            # Ideally ChannelModel should have 'nlos_path_loss_params' storage.
            # For now, we update the GUI spinboxes to show the LOS values (User requested "only the los case")
            
            # Update UI spinboxes to reflect preset (LOS values)
            params = env_config.los_params
            
            if hasattr(self, 'cluster_decay_spin'):
                self.cluster_decay_spin.blockSignals(True)
                self.cluster_decay_spin.setValue(params.cluster_decay * 1e9)
                self.cluster_decay_spin.blockSignals(False)
            if hasattr(self, 'ray_decay_spin'):
                self.ray_decay_spin.blockSignals(True)
                self.ray_decay_spin.setValue(params.ray_decay * 1e9)
                self.ray_decay_spin.blockSignals(False)
            if hasattr(self, 'los_pl_spin'):
                self.los_pl_spin.blockSignals(True)
                self.los_pl_spin.setValue(params.path_loss_exponent)
                self.los_pl_spin.blockSignals(False)
            if hasattr(self, 'los_shadow_spin'):
                self.los_shadow_spin.blockSignals(True)
                self.los_shadow_spin.setValue(params.shadow_fading_std)
                self.los_shadow_spin.blockSignals(False)
    
    def on_fixed_noise_changed(self, noise_std):
        """Handle fixed hardware implementation noise changes"""
        self.channel_conditions.uwb_params.fixed_noise_std = noise_std
    
    # ==================== Utility Methods ====================
    
    def add_anchor_by_coords(self):
        """Add anchor by coordinates"""
        x = self.x_coord.value()
        y = self.y_coord.value()
        new_anchor = Anchor(Position(x, y))
        self.anchors.append(new_anchor)
        
        # Renormalize IDs to ensure sequence (A1, A2...)
        self.renormalize_anchors()

    def renormalize_anchors(self):
        """Renormalize anchor IDs to be sequential (A1, A2...)"""
        for i, anchor in enumerate(self.anchors):
            anchor.id = f"A{i+1}"
            
        self.update_anchor_list()
        self.plot_manager.update_anchor_visualization(
            self.position_plot, self.anchors, self.channel_conditions, self.tag)
    
    def update_anchor_list(self):
        """Update anchor list display"""
        if self.anchor_list is None:
            return  # Panel not opened yet
        text = "Current Anchors:\n"
        for anchor in self.anchors:
            text += f"{anchor.id}: ({anchor.position.x:.1f}, {anchor.position.y:.1f})\n"
        try:
            self.anchor_list.setText(text)
        except RuntimeError:
            self.anchor_list = None
        
        # Also update windows that need anchor list
        if self.distance_plots_window is not None:
            self.distance_plots_window.update_anchors(self.anchors)
            
        if self.cir_window is not None:
            self.cir_window.update_anchors(self.anchors)
    

    
    def show_error_message(self, title, message):
        """Show error message dialog"""
        dialog = CustomMessageBox(title, message, "error", self)
        dialog.exec_()
    
    def show_info_message(self, title, message):
        """Show info message dialog"""
        dialog = CustomMessageBox(title, message, "info", self)
        dialog.exec_()
    
    def show_question_dialog(self, title, message):
        """Show question dialog"""
        from PyQt5.QtWidgets import QDialog
        dialog = CustomMessageBox(title, message, "question", self)
        return dialog.exec_() == QDialog.Accepted
    
    # ========== Simulation Settings Handlers ==========
    
    def _on_duration_changed(self, text):
        """Handle duration combo box change"""
        if text == "Custom":
            duration = self.timeline_widget.custom_duration_spin.value()
        elif text == "∞ Infinite":
            duration = None  # Infinite mode
        else:
            # Parse duration from text (e.g., "30s" -> 30)
            duration = float(text.replace('s', ''))
        
        self.simulation_manager.set_duration(duration)
    
    def _on_custom_duration_changed(self, value):
        """Handle custom duration spinner change"""
        self.simulation_manager.set_duration(float(value))
    
    def _on_snapshot_interval_changed(self, text):
        """Handle snapshot interval combo change"""
        interval_map = {
            "Every Frame": 1,
            "Every 3rd": 3,
            "Every 5th": 5,
            "Every 10th": 10
        }
        interval = interval_map.get(text, 5)
        self.simulation_manager.set_snapshot_interval(interval)
    
    def toggle_history_mode(self, checked):
        """Toggle between full history and tail history view"""
        # If timeline is active (simulation finished), update view immediately
        if hasattr(self, 'timeline_widget') and self.timeline_widget.isVisible():
            current_time = self.timeline_widget.current_time
            self._on_timeline_scrub(current_time)

    def _on_timeline_scrub(self, time_value):
        """Handle timeline slider scrubbing - restore state at given time"""
        snapshot = self.simulation_manager.recorder.get_snapshot_at_time(time_value)
        if not snapshot:
            print(f"No snapshot found at time {time_value:.2f}s")
            return
        
        # 1. Update tag object position (Crucial for visualization methods to work)
        self.tag.position.x = snapshot.tag_position[0]
        self.tag.position.y = snapshot.tag_position[1]
        
        # 2. Update anchor positions if recorded
        if snapshot.anchor_states:
            anchor_map = {a.id: a for a in self.anchors}
            for anchor_state in snapshot.anchor_states:
                a_id = anchor_state.get('id')
                # Try to match ID (handle potential string/int mismatch if any)
                if a_id in anchor_map:
                    pos = anchor_state.get('position')
                    if pos:
                        anchor_map[a_id].position.x = pos[0]
                        anchor_map[a_id].position.y = pos[1]

        # 3. Get trajectory history up to this time
        # Check toggle state for history length
        max_points = 100
        if hasattr(self, 'history_toggle') and self.history_toggle.isChecked():
            max_points = None
            
        tag_x, tag_y, est_x, est_y = self.simulation_manager.recorder.get_trajectory_up_to_time(time_value, max_points=max_points)
        
        # 4. Update trajectory plots
        if len(tag_x) > 0:
            self.plot_items['tag_trajectory'].setData(tag_x, tag_y)
            self.plot_items['estimated_trajectory'].setData(est_x, est_y)
        
        # 5. Update current position markers
        self.plot_items['tag_point'].setData([snapshot.tag_position[0]], [snapshot.tag_position[1]])
        self.plot_items['estimated_point'].setData([snapshot.estimated_position[0]], [snapshot.estimated_position[1]])
        
        # 6. Update visual elements (Lines, Anchors)
        # This redraws lines to the NEW tag position and updates colors based on geometry
        self.plot_manager.update_measurement_lines(
            self.position_plot, self.anchors, self.tag, self.channel_conditions)
            
        self.plot_manager.update_anchor_visualization(
            self.position_plot, self.anchors, self.channel_conditions, self.tag)
            
        # 7. Update status display
        self.simulation_manager.update_status(snapshot.error)

    # ========== Algorithm Creation Wizard ==========

    def open_algorithm_wizard(self):
        """Open the wizard to create a new custom algorithm"""
        import os
        
        # Determine user_algorithms directory
        from src.utils.resource_loader import get_data_path
        user_algo_dir = get_data_path(os.path.join('src', 'user_algorithms'))
        
        wizard = AlgorithmCreationWindow(self, user_algo_dir)
        if wizard.exec_():
            # If saved, refresh the algorithm list
            self.refresh_algorithm_list()
            
    def refresh_algorithm_list(self):
        """Reload algorithms and update the combobox"""
        from src.core.localization import Alghortimes_doc
        
        # Force reload by clearing cache
        Alghortimes_doc._cached_algorithms = None
        doc = Alghortimes_doc()
        algos = list(doc.get_algorithm_methods().keys())
        
        # Update combo box if it exists
        if self.algo_combo:
            current = self.algo_combo.currentText()
            self.algo_combo.clear()
            self.algo_combo.addItems(algos)
            
            # Restore selection if possible
            index = self.algo_combo.findText(current)
            if index >= 0:
                self.algo_combo.setCurrentIndex(index)
            else:
                # If current algo disappeared (unlikely unless deleted), select first
                if self.algo_combo.count() > 0:
                    self.algo_combo.setCurrentIndex(0)
                    
        # Update status
        if self.status_display:
            self.status_display.append("Algorithms reloaded.")


    def export_simulation_data(self):
        """Export simulation data to files"""
        try:
            # Generate default name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"export_{timestamp}"
            
            # Ask user for location and name
            export_dir, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Simulation Data", 
                os.path.join(os.getcwd(), default_name),
                "Directory (*)"
            )
            
            if not export_dir:
                return  # User cancelled
                
            # Create export directory
            os.makedirs(export_dir, exist_ok=True)
            
            # 1. Export Configuration
            config_file = os.path.join(export_dir, "configuration.txt")
            with open(config_file, "w") as f:
                f.write(f"Simulation Export - {timestamp}\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"Algorithm: {self.algorithm}\n")
                f.write(f"Duration: {self.simulation_manager.simulation_time:.2f}s\n")
                f.write(f"Movement Pattern: {self.movement_pattern}\n")
                f.write(f"Movement Speed: {self.movement_speed} m/s\n\n")
                
                f.write("UWB Channel Parameters:\n")
                f.write("-" * 20 + "\n")
                params = self.channel_conditions.los_path_loss_params
                f.write(f"Path Loss Exponent: {params.path_loss_exponent}\n")
                f.write(f"Shadow Fading Std: {params.shadow_fading_std}\n")
                f.write(f"Frequency: {self.channel_conditions.uwb_params.center_frequency/1e9} GHz\n\n")
                
                f.write("Anchors:\n")
                f.write("-" * 20 + "\n")
                for anchor in self.anchors:
                    f.write(f"ID: {anchor.id}, Pos: ({anchor.position.x}, {anchor.position.y})\n")
            
            # 2. Export IMU Data
            imu_file = os.path.join(export_dir, "imu_data.csv")
            with open(imu_file, "w") as f:
                f.write("Timestamp,Acc_X,Acc_Y,Acc_Z,Gyro_X,Gyro_Y,Gyro_Z\n")
                if hasattr(self.tag, 'imu_data'):
                    data = self.tag.imu_data
                    for i in range(len(data.timestamps)):
                        f.write(f"{data.timestamps[i]},{data.acc_x[i]},{data.acc_y[i]},{data.acc_z[i]},"
                                f"{data.gyro_x[i]},{data.gyro_y[i]},{data.gyro_z[i]}\n")
            
            # 3. Export Simulation Results
            results_file = os.path.join(export_dir, "simulation_results.csv")
            with open(results_file, "w") as f:
                # Header
                header = "Timestamp,True_X,True_Y,Est_X,Est_Y,Error"
                # Add columns for each anchor
                for anchor in self.anchors:
                    header += f",Dist_{anchor.id},NLOS_{anchor.id}"
                f.write(header + "\n")
                
                # Data
                snapshots = self.simulation_manager.recorder.snapshots
                for snap in snapshots:
                    line = f"{snap.timestamp},{snap.tag_position[0]},{snap.tag_position[1]}," \
                           f"{snap.estimated_position[0]},{snap.estimated_position[1]},{snap.error}"
                    
                    # Add anchor data
                    # Create a map for quick lookup
                    anchor_map = {astate['id']: astate for astate in snap.anchor_states}
                    
                    for anchor in self.anchors:
                        if anchor.id in anchor_map:
                            # distance is stored in 'measurements' dict in snapshot, not anchor_states
                            dist = snap.measurements.get(anchor.id, "")
                            is_nlos = not anchor_map[anchor.id].get('is_los', True)
                            line += f",{dist},{1 if is_nlos else 0}"
                        else:
                            line += ",,"
                    
                    f.write(line + "\n")

            CustomMessageBox("Export Successful", f"Data exported to:\n{export_dir}", "info", self).exec_()
            
        except Exception as e:
            CustomMessageBox("Export Failed", f"Error exporting data: {str(e)}", "error", self).exec_()
    def keyPressEvent(self, event):
        """Handle key press events"""
        # Delegate to selection manager first
        if hasattr(self, 'selection_manager') and self.selection_manager.handle_key_press(event):
            return

        # Delegate to anchor selection manager
        if hasattr(self, 'anchor_selection_manager') and self.anchor_selection_manager.handle_key_press(event):
            return
            
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release events"""
        # Delegate to selection manager first
        if hasattr(self, 'selection_manager') and self.selection_manager.handle_key_release(event):
            return

        # Delegate to anchor selection manager
        if hasattr(self, 'anchor_selection_manager') and self.anchor_selection_manager.handle_key_release(event):
            return
            
        super().keyReleaseEvent(event)
    def delete_selected_algorithm(self):
        """Delete user-selected custom algorithm"""
        current_algo = self.algo_combo.currentText()
        
        # Check if it's a built-in algorithm
        from src.core.localization import Alghortimes_doc
        doc = Alghortimes_doc()
        # Create temporary instance just to check keys might be heavy, better check against cached
        if current_algo in [
            "Trilateration",
            "Extended Kalman Filter",
            "Unscented Kalman Filter",
            "Adaptive Extended Kalman Filter",
            "NLOS-Aware AEKF",
            "Improved Adaptive EKF",
            "IMU Only",
            "IMU assisted NLOS-Aware AEKF",
        ]:
            QMessageBox.warning(self, "Cannot Delete", 
                              f"'{current_algo}' is a built-in algorithm and cannot be deleted.")
            return

        # Confirm deletion
        reply = QMessageBox.question(self, "Confirm Deletion", 
                                   f"Are you sure you want to delete the custom algorithm '{current_algo}'?\n\nThis will permanently delete the python file.",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                   
        if reply == QMessageBox.Yes:
            try:
                # Find the file
                import os
                from src.utils.resource_loader import get_data_path
                user_algo_dir = get_data_path(os.path.join("src", "user_algorithms"))
                
                # We need to find which file corresponds to this algorithm name
                # This is tricky because the name is in the class, not necessarily the filename
                # But typically for this app, they match or we can infer.
                # Let's search for the file containing the class name or algorithm name
                
                # Simpler approach: Reload algorithms to check where it came from? 
                # Or just assume filename matches normalized name?
                # The AlgorithmLoader registers them.
                
                # Let's iterate files in user_algo_dir
                target_file = None
                for filename in os.listdir(user_algo_dir):
                    if filename.endswith(".py") and filename != "__init__.py":
                        # Check file content or name
                        # For now, let's assume the user knows what they are deleting 
                        # functionality is "Delete Custom Algorithm".
                        # If the system can't easily map name -> file without metadata.
                        
                        # Let's check `Alghortimes_doc._cached_algorithms` values?
                        # They are functions. `func.__code__.co_filename` might have the path!
                        pass
                
                # Algorithm is bound method or function
                algo_func = doc.get_algorithm_methods().get(current_algo)
                if algo_func:
                    # It's likely a bound method or partial, or function
                    # If it's a custom algo loaded from file
                    import inspect
                    try:
                        file_path = inspect.getfile(algo_func)
                        if "user_algorithms" in file_path:
                            os.remove(file_path)
                            
                            # Reload algorithms
                            Alghortimes_doc.reload_custom_algorithms()
                            
                            # Refresh combo box
                            self.refresh_algorithm_list()
                            
                            QMessageBox.information(self, "Success", f"Algorithm '{current_algo}' deleted.")
                        else:
                            QMessageBox.warning(self, "Error", f"Could not determine if '{current_algo}' is a user file.\nPath: {file_path}")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to identify file for algorithm: {e}")
                else:
                    QMessageBox.warning(self, "Error", "Algorithm function not found.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete algorithm: {str(e)}")

    def open_algorithm_folder(self):
        """Open the folder containing custom algorithms"""
        from src.utils.resource_loader import get_data_path
        import os
        
        folder = get_data_path(os.path.join("src", "user_algorithms"))
        os.makedirs(folder, exist_ok=True)
        
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            QMessageBox.warning(self, "Error", "User algorithms folder does not exist.")


