"""
Control Panels Module
Handles creation of all control panels and UI widgets
"""
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QPushButton, QCheckBox, QTextEdit, QSlider, QWidget)
from PyQt5.QtCore import Qt
from src.core.localization import LocalizationAlgorthimes
from src.gui.theme import COLORS
from src.gui.widgets import ModernGroupBox, ActionButton


class ControlPanelFactory:
    """Factory class for creating control panels"""
    
    @staticmethod
    def create_file_buttons():
        """Create file operation buttons (Save, Load, Clean)"""
        layout = QHBoxLayout()
        
        save_map_btn = ActionButton("💾", variant="secondary")
        save_map_btn.setToolTip("Save Map")
        
        load_map_btn = ActionButton("📂", variant="secondary")
        load_map_btn.setToolTip("Load Map")
        
        clean_map_btn = ActionButton("🗑️", variant="secondary")
        clean_map_btn.setToolTip("Clean Map")
        
        layout.addWidget(save_map_btn)
        layout.addWidget(load_map_btn)
        layout.addWidget(clean_map_btn)
        layout.setSpacing(4)
        
        return layout, save_map_btn, load_map_btn, clean_map_btn
    
    @staticmethod
    def create_main_controls():
        """Create main simulation controls (Restart, Start/Pause) - Centered"""
        layout = QHBoxLayout()
        
        restart_button = ActionButton("🔄 Reset", variant="secondary")
        restart_button.setToolTip("Restart Simulation")
        
        # Button starts as "Start" - simulation is paused by default
        pause_button = ActionButton("▶️ Start", variant="secondary")
        pause_button.setToolTip("Start Simulation")
        pause_button.setCheckable(True)
        pause_button.setChecked(True)  # Checked = paused initially
        
        layout.addWidget(restart_button)
        layout.addWidget(pause_button)
        layout.setSpacing(10)
        
        return layout, restart_button, pause_button
    
    @staticmethod
    def create_sensor_controls():
        """Create sensor control buttons (IMU, Distance, AI)"""
        layout = QHBoxLayout()
        
        # IMU button
        imu_button = ActionButton("📡 IMU", variant="secondary")
        imu_button.setToolTip("Show IMU Data Window")
        
        # Distance button
        distance_button = ActionButton("📊 Dist", variant="secondary")
        distance_button.setToolTip("Show Distance Plots Window")
        
        # AI Training Data button
        ai_data_button = ActionButton("🤖 AI", variant="secondary")
        ai_data_button.setToolTip("Toggle AI Training Data Collection")
        ai_data_button.setCheckable(True)
        
        layout.addWidget(imu_button)
        layout.addWidget(distance_button)
        layout.addWidget(ai_data_button)
        layout.setSpacing(6)
        
        return layout, imu_button, distance_button, ai_data_button

    @staticmethod
    def create_view_controls():
        """Create view control buttons (Path, Lines)"""
        layout = QHBoxLayout()
        
        # Path toggle button
        path_toggle_btn = ActionButton("📍 Path", variant="secondary")
        path_toggle_btn.setToolTip("Toggle Path Visibility")
        path_toggle_btn.setCheckable(True)
        path_toggle_btn.setChecked(True)
        
        # Lines toggle button
        lines_toggle_btn = ActionButton("📏 Lines", variant="secondary")
        lines_toggle_btn.setToolTip("Toggle Measurement Lines")
        lines_toggle_btn.setCheckable(True)
        lines_toggle_btn.setChecked(True)
        
        layout.addWidget(path_toggle_btn)
        layout.addWidget(lines_toggle_btn)
        layout.setSpacing(6)
        
        return layout, path_toggle_btn, lines_toggle_btn
    
    
    @staticmethod
    def create_anchor_config_panel():
        """Create anchor configuration panel"""
        anchor_group = ModernGroupBox("Anchor Configuration")
        
        anchor_layout = QVBoxLayout()
        anchor_layout.setSpacing(5)
        anchor_layout.setContentsMargins(5, 5, 5, 5)
        
        # Coordinate input
        coord_input_layout = QHBoxLayout()
        coord_section = QWidget()
        coord_grid = QGridLayout(coord_section)
        coord_grid.setSpacing(4)
        coord_grid.setContentsMargins(0, 0, 0, 0)
        
        
        x_label = QLabel("X:")
        x_coord = QDoubleSpinBox()
        x_coord.setDecimals(5)
        x_coord.setRange(-100.0, 100.0)
        
        y_label = QLabel("Y:")
        y_coord = QDoubleSpinBox()
        y_coord.setDecimals(5)
        y_coord.setRange(-100.0, 100.0)
        
        coord_grid.addWidget(x_label, 0, 0)
        coord_grid.addWidget(x_coord, 0, 1)
        coord_grid.addWidget(y_label, 0, 2)
        coord_grid.addWidget(y_coord, 0, 3)
        
        coord_input_layout.addWidget(coord_section)
        
        add_coord_btn = ActionButton("Add", variant="secondary")
        coord_input_layout.addWidget(add_coord_btn)
        anchor_layout.addLayout(coord_input_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(4)
        add_anchor_btn = ActionButton("Click to Add", variant="secondary")
        add_anchor_btn.setCheckable(True)
        
        delete_anchor_btn = ActionButton("Delete", variant="secondary")
        delete_anchor_btn.setCheckable(True)
        
        action_layout.addWidget(add_anchor_btn)
        action_layout.addWidget(delete_anchor_btn)
        anchor_layout.addLayout(action_layout)
        
        # Anchor list
        anchor_list = QTextEdit()
        anchor_list.setReadOnly(True)
        anchor_list.setMaximumHeight(80)
        anchor_layout.addWidget(anchor_list)
        
        anchor_group.setLayout(anchor_layout)
        
        widgets = {
            'x_coord': x_coord,
            'y_coord': y_coord,
            'add_coord_btn': add_coord_btn,
            'add_anchor_btn': add_anchor_btn,
            'delete_anchor_btn': delete_anchor_btn,
            'anchor_list': anchor_list
        }
        
        return anchor_group, widgets
    
    @staticmethod
    def create_nlos_panel():
        """Create NLOS regions panel - compact design"""
        from PyQt5.QtWidgets import QGridLayout
        
        nlos_group = ModernGroupBox("NLOS Zones")
        
        nlos_layout = QVBoxLayout()
        nlos_layout.setContentsMargins(4, 8, 4, 4)
        nlos_layout.setSpacing(4)
        
        # Grid layout for buttons - 2x2 grid
        grid = QGridLayout()
        grid.setSpacing(4)
        
        draw_mode_btn = ActionButton("🖊️ Line", variant="secondary")
        draw_mode_btn.setCheckable(True)
        
        draw_polygon_btn = ActionButton("⬡ Polygon", variant="secondary")
        draw_polygon_btn.setCheckable(True)
        
        add_moving_btn = ActionButton("🏃 Moving", variant="secondary")
        add_moving_btn.setCheckable(True)
        
        delete_mode_btn = ActionButton("🗑️ Delete", variant="secondary")
        delete_mode_btn.setCheckable(True)
        delete_mode_btn.setToolTip("Enable delete mode to remove specific zones/obstacles")
        
        grid.addWidget(draw_mode_btn, 0, 0)
        grid.addWidget(draw_polygon_btn, 0, 1)
        grid.addWidget(add_moving_btn, 1, 0)
        grid.addWidget(delete_mode_btn, 1, 1)
        nlos_layout.addLayout(grid)
        
        # Import button - full width but compact
        import_image_btn = ActionButton("🖼️ Import Floor Plan", variant="secondary")
        import_image_btn.setToolTip("Import binary image (black=walls, white=open)")
        nlos_layout.addWidget(import_image_btn)
        
        # Compact instructions - collapsible style
        instructions_label = QLabel("💡 Draw regions • Right-click to finish • Click 'Delete' to remove items")
        instructions_label.setWordWrap(True)
        instructions_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; padding: 4px;")
        nlos_layout.addWidget(instructions_label)
        
        nlos_group.setLayout(nlos_layout)
        
        widgets = {
            'draw_mode_btn': draw_mode_btn,
            'draw_polygon_btn': draw_polygon_btn,
            'delete_mode_btn': delete_mode_btn,
            'import_image_btn': import_image_btn,
            'add_moving_btn': add_moving_btn
        }
        
        return nlos_group, widgets
    
    @staticmethod
    def create_uwb_channel_panel():
        """Create UWB channel configuration panel - Tabbed Design"""
        from PyQt5.QtWidgets import QTabWidget, QWidget
        
        channel_group = ModernGroupBox("UWB Channel Configuration")
        
        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(4, 12, 4, 4)
        
        # Create Tab Widget
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {COLORS['border']}; }}
            QTabBar::tab {{ background: #2d2d30; color: #d4d4d4; padding: 6px; }}
            QTabBar::tab:selected {{ background: #3e3e42; border-bottom: 2px solid {COLORS['accent']}; }}
        """)
        
        # --- Tab 1: Signal & Link ---
        signal_tab = QWidget()
        signal_layout = QGridLayout()
        signal_layout.setSpacing(6)
        signal_layout.setContentsMargins(8, 8, 8, 8)
        
        # Channel
        signal_layout.addWidget(QLabel("Channel:"), 0, 0)
        channel_combo = QComboBox()
        channel_combo.addItems(["1", "2", "3", "5", "9", "10"])
        channel_combo.setCurrentText("5")
        channel_combo.setToolTip("UWB Channel (Center Frequency & Bandwidth)")
        signal_layout.addWidget(channel_combo, 0, 1)
        
        # Preset
        signal_layout.addWidget(QLabel("Preset:"), 1, 0)
        environment_combo = QComboBox()
        environment_combo.addItems(["CM1 (LOS 0-4m)", "CM2 (NLOS 0-4m)", "CM3 (NLOS 4-10m)", "CM4 (Extreme NLOS)", "Custom"])
        environment_combo.setCurrentText("CM1 (LOS 0-4m)")
        signal_layout.addWidget(environment_combo, 1, 1)

        # Gains
        signal_layout.addWidget(QLabel("Tx Gain (dBi):"), 2, 0)
        tx_gain_spin = QDoubleSpinBox()
        tx_gain_spin.setRange(-100, 100)
        tx_gain_spin.setValue(0.0)
        tx_gain_spin.setSingleStep(0.5)
        signal_layout.addWidget(tx_gain_spin, 2, 1)
        
        signal_layout.addWidget(QLabel("Rx Gain (dBi):"), 3, 0)
        rx_gain_spin = QDoubleSpinBox()
        rx_gain_spin.setRange(-100, 100)
        rx_gain_spin.setValue(0.0)
        rx_gain_spin.setSingleStep(0.5)
        signal_layout.addWidget(rx_gain_spin, 3, 1)

        # Path Loss
        signal_layout.addWidget(QLabel("Path Loss (n):"), 4, 0)
        los_pl_spin = QDoubleSpinBox()
        los_pl_spin.setRange(0, 10)
        los_pl_spin.setValue(2.0)
        los_pl_spin.setSingleStep(0.1)
        signal_layout.addWidget(los_pl_spin, 4, 1)
        
        signal_layout.addWidget(QLabel("Shadow (dB):"), 5, 0)
        los_shadow_spin = QDoubleSpinBox()
        los_shadow_spin.setRange(0, 20)
        los_shadow_spin.setValue(2.0)
        los_shadow_spin.setSingleStep(0.1)
        signal_layout.addWidget(los_shadow_spin, 5, 1)
        
        signal_layout.addWidget(QLabel("Freq Decay (κ):"), 6, 0)
        freq_decay_spin = QDoubleSpinBox()
        freq_decay_spin.setRange(0.0, 5.0)
        freq_decay_spin.setValue(1.0)
        freq_decay_spin.setSingleStep(0.1)
        signal_layout.addWidget(freq_decay_spin, 6, 1)
        
        # Keep layout tight at top
        signal_layout.setRowStretch(7, 1)
        signal_tab.setLayout(signal_layout)
        tabs.addTab(signal_tab, "Signal")
        
        # --- Tab 2: Multipath ---
        mp_tab = QWidget()
        mp_layout = QGridLayout()
        mp_layout.setSpacing(6)
        mp_layout.setContentsMargins(8, 8, 8, 8)
        
        mp_layout.addWidget(QLabel("Cluster Decay (Γ):"), 0, 0)
        cluster_decay_spin = QDoubleSpinBox()
        cluster_decay_spin.setRange(0.1, 100)
        cluster_decay_spin.setValue(7.1)
        cluster_decay_spin.setSuffix(" ns")
        mp_layout.addWidget(cluster_decay_spin, 0, 1)
        
        mp_layout.addWidget(QLabel("Ray Decay (γ):"), 1, 0)
        ray_decay_spin = QDoubleSpinBox()
        ray_decay_spin.setRange(0.1, 100)
        ray_decay_spin.setValue(4.3)
        ray_decay_spin.setSuffix(" ns")
        mp_layout.addWidget(ray_decay_spin, 1, 1)
        
        mp_layout.setRowStretch(2, 1)
        mp_tab.setLayout(mp_layout)
        tabs.addTab(mp_tab, "Multipath")

        # --- Tab 3: Receiver ---
        rx_tab = QWidget()
        rx_layout = QGridLayout()
        rx_layout.setSpacing(6)
        rx_layout.setContentsMargins(8, 8, 8, 8)
        
        rx_layout.addWidget(QLabel("Noise Fig (dB):"), 0, 0)
        noise_spin = QDoubleSpinBox()
        noise_spin.setRange(0, 50)
        noise_spin.setValue(6.0)
        rx_layout.addWidget(noise_spin, 0, 1)
        
        rx_layout.addWidget(QLabel("Jitter (m):"), 1, 0)
        fixed_noise_spin = QDoubleSpinBox()
        fixed_noise_spin.setRange(0.0, 1.0)
        fixed_noise_spin.setValue(0.1)
        fixed_noise_spin.setSingleStep(0.01)
        rx_layout.addWidget(fixed_noise_spin, 1, 1)
        
        rx_layout.addWidget(QLabel("ToA Thresh:"), 2, 0)
        toa_threshold_spin = QDoubleSpinBox()
        toa_threshold_spin.setRange(0.001, 1.0)
        toa_threshold_spin.setValue(0.1)
        toa_threshold_spin.setSingleStep(0.05)
        rx_layout.addWidget(toa_threshold_spin, 2, 1)
        
        rx_layout.addWidget(QLabel("Noise Model:"), 3, 0)
        noise_model_combo = QComboBox()
        noise_model_combo.addItems(["Gaussian", "Non-Centralized Gaussian", "Uniform", "Laplace", "Mixed Gaussian", "Student's t"])
        noise_model_combo.setCurrentText("Gaussian")
        rx_layout.addWidget(noise_model_combo, 3, 1)
        
        rx_layout.setRowStretch(4, 1)
        rx_tab.setLayout(rx_layout)
        tabs.addTab(rx_tab, "Receiver")
        
        main_layout.addWidget(tabs)
        channel_group.setLayout(main_layout)
        
        widgets = {
            'channel_combo': channel_combo,
            'tx_gain_spin': tx_gain_spin,
            'rx_gain_spin': rx_gain_spin,
            'noise_spin': noise_spin,
            'los_pl_spin': los_pl_spin,
            'los_shadow_spin': los_shadow_spin,
            'noise_model_combo': noise_model_combo,
            'freq_decay_spin': freq_decay_spin,
            'cluster_decay_spin': cluster_decay_spin,
            'ray_decay_spin': ray_decay_spin,
            'toa_threshold_spin': toa_threshold_spin,
            'environment_combo': environment_combo,
            'fixed_noise_spin': fixed_noise_spin
        }
        
        return channel_group, widgets
    
    @staticmethod
    def create_movement_panel():
        """Create movement pattern panel"""
        movement_group = ModernGroupBox("Movement Pattern")
        movement_layout = QGridLayout()
        movement_layout.setSpacing(4)
        
        # Pattern selection
        pattern_label = QLabel("Pattern:")
        pattern_combo = QComboBox()
        pattern_combo.addItems(["Circular", "Figure 8", "Square", "Random Walk", "Fixed Point", "Foot Mounted"])
        movement_layout.addWidget(pattern_label, 0, 0)
        movement_layout.addWidget(pattern_combo, 0, 1, 1, 2)
        
        # Time step control
        timestep_label = QLabel("Time Step:")
        timestep_slider = QSlider(Qt.Horizontal)
        timestep_slider.setMinimum(1)
        timestep_slider.setMaximum(100)
        timestep_slider.setValue(5)
        timestep_value_label = QLabel("5 ms (200 Hz)")
        timestep_value_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        
        
        # Speed control (definitions only, added to layout later)
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setMinimum(1)
        speed_slider.setMaximum(1000)
        speed_slider.setValue(10)
        speed_value_label = QLabel("1.0 m/s")
        speed_value_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")

        
        # Custom trajectory buttons
        traj_btn_layout = QHBoxLayout()
        draw_trajectory_btn = ActionButton("➕ New", variant="secondary")
        draw_trajectory_btn.setToolTip("Create/Draw Custom Trajectory")
        # draw_trajectory_btn.setFixedWidth(60) # Removed fixed width
        
        import_traj_btn = ActionButton("📂 Import", variant="secondary")
        import_traj_btn.setToolTip("Import trajectory from CSV")
        
        delete_traj_btn = ActionButton("🗑️", variant="danger")
        delete_traj_btn.setToolTip("Delete selected custom trajectory")
        # delete_traj_btn.setFixedWidth(40) # Removed fixed width
        
        open_traj_folder_btn = ActionButton("📂", variant="secondary")
        open_traj_folder_btn.setToolTip("Open trajectories folder")
        # open_traj_folder_btn.setFixedWidth(40) # Removed fixed width
        
        play_exact_btn = ActionButton("⏱️ Play Exact", variant="secondary")
        play_exact_btn.setToolTip("Play exact trajectory points exactly once based on time step")
        
        # Row 1: Trajectory Actions (New, Import / Open, Delete)
        # Split into two sub-rows for better fit
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(8) # Increased spacing
        
        # Sub-row 1: Creation/Import
        row1 = QHBoxLayout()
        row1.addWidget(draw_trajectory_btn, 1) # Stretch factor 1
        row1.addWidget(import_traj_btn, 1) # Stretch factor 1
        actions_layout.addLayout(row1)
        
        # Sub-row 2: Management
        row2 = QHBoxLayout()
        row2.addWidget(open_traj_folder_btn, 1) # Stretch factor 1
        row2.addWidget(delete_traj_btn, 1) # Stretch factor 1
        actions_layout.addLayout(row2)
        
        row3 = QHBoxLayout()
        row3.addWidget(play_exact_btn, 1)
        actions_layout.addLayout(row3)
        
        # Add actions layout at Row 1 (pushing others down)
        movement_layout.addLayout(actions_layout, 1, 0, 1, 3)
        
        # Adjust indices of subsequent rows
        movement_layout.addWidget(QLabel("Timestep:"), 2, 0)
        movement_layout.addWidget(timestep_slider, 2, 1)
        movement_layout.addWidget(timestep_value_label, 2, 2)
        
        movement_layout.addWidget(QLabel("Speed:"), 3, 0)
        movement_layout.addWidget(speed_slider, 3, 1)
        movement_layout.addWidget(speed_value_label, 3, 2)
        
        # Target point and fixed point config
        fp_label = QLabel("Fixed Pt:")
        fp_label.setToolTip("Target X/Y for Fixed Point Pattern")
        
        fp_x_spin = QDoubleSpinBox()
        fp_x_spin.setRange(-1000.0, 1000.0)
        fp_x_spin.setDecimals(3)
        fp_x_spin.setPrefix("X: ")
        fp_x_spin.setValue(0.0)
        
        fp_y_spin = QDoubleSpinBox()
        fp_y_spin.setRange(-1000.0, 1000.0)
        fp_y_spin.setDecimals(3)
        fp_y_spin.setPrefix("Y: ")
        fp_y_spin.setValue(0.0)
        
        fp_layout = QHBoxLayout()
        fp_layout.setSpacing(4)
        fp_layout.addWidget(fp_x_spin, 1)
        fp_layout.addWidget(fp_y_spin, 1)
        
        target_point_btn = ActionButton("🎯 Set on Map", variant="secondary")
        target_point_btn.setCheckable(True)
        # Remove Fixed Pt: label from the grid to save space, rely on prefixes
        movement_layout.addLayout(fp_layout, 4, 0, 1, 2)
        movement_layout.addWidget(target_point_btn, 4, 2)
        
        movement_group.setLayout(movement_layout)
        
        widgets = {
            'pattern_combo': pattern_combo,
            'timestep_slider': timestep_slider,
            'timestep_value_label': timestep_value_label,
            'speed_slider': speed_slider,
            'speed_value_label': speed_value_label,
            'draw_trajectory_btn': draw_trajectory_btn,
            'import_traj_btn': import_traj_btn,
            'delete_traj_btn': delete_traj_btn,
            'open_traj_folder_btn': open_traj_folder_btn,
            'play_exact_btn': play_exact_btn,
            'target_point_btn': target_point_btn,
            'fp_x_spin': fp_x_spin,
            'fp_y_spin': fp_y_spin
        }
        
        return movement_group, widgets
    
    @staticmethod
    def create_algorithm_panel():
        """Create localization algorithm panel - Compact Design"""
        algo_group = ModernGroupBox("Localization Algorithm")
        
        # Main vertical layout
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(8)
        algo_layout.setContentsMargins(5, 8, 5, 5)
        
        # Row 1: Algorithm Selection + Add Button
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(4)
        
        algo_combo = QComboBox()
        # Get algorithms dynamically including custom ones
        from src.core.localization import Alghortimes_doc
        doc = Alghortimes_doc()
        algos = list(doc.get_algorithm_methods().keys())
        algo_combo.addItems(algos)
        algo_combo.setToolTip("Select Localization Algorithm")
        
        # Add "New" button
        add_algo_btn = ActionButton("➕ New", variant="secondary")
        # add_algo_btn.setFixedWidth(60)
        add_algo_btn.setToolTip("Create Custom Algorithm")
        
        # Add Delete button - Increased size for visibility
        delete_algo_btn = ActionButton("🗑️", variant="danger")
        # delete_algo_btn.setFixedWidth(40)
        delete_algo_btn.setToolTip("Delete selected custom algorithm")
        
        # Add Open Folder button - Increased size for visibility
        open_algo_folder_btn = ActionButton("📂", variant="secondary")
        # open_algo_folder_btn.setFixedWidth(40)
        open_algo_folder_btn.setToolTip("Open algorithms folder")
        
        # Layout:
        # Row 1: Algo Combo
        # Row 2: Actions (New, Open, Delete)
        # Row 3: Parameters
        
        # Row 1: Combo
        algo_layout.addWidget(algo_combo)
        
        # Row 2: Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(5)
        
        # Remove fixed widths to allow expansion
        add_algo_btn.setMinimumWidth(60) # Set minimum instead of fixed
        open_algo_folder_btn.setMinimumWidth(40)
        delete_algo_btn.setMinimumWidth(40)
        
        btn_layout.addWidget(add_algo_btn, 1) # Stretch factor 1
        btn_layout.addWidget(open_algo_folder_btn, 1) # Stretch factor 1
        btn_layout.addWidget(delete_algo_btn, 1) # Stretch factor 1
        
        algo_layout.addLayout(btn_layout)
        
        # Row 2: Parameters (MA Window)
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(8)
        
        ma_label = QLabel("Smoothing (MA):")
        ma_label.setToolTip("Moving Average Window Size")
        ma_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        
        ma_window_spin = QSpinBox()
        ma_window_spin.setRange(1, 100)
        ma_window_spin.setValue(20)
        ma_window_spin.setSuffix(" samples")
        ma_window_spin.setToolTip("Moving Average Window Size")
        
        row2_layout.addWidget(ma_label)
        row2_layout.addWidget(ma_window_spin)
        row2_layout.addStretch() # Push to left
        
        algo_layout.addLayout(row2_layout)
        
        algo_group.setLayout(algo_layout)
        
        widgets = {
            'algo_combo': algo_combo,
            'add_algo_btn': add_algo_btn,
            'delete_algo_btn': delete_algo_btn,
            'open_algo_folder_btn': open_algo_folder_btn,
            'ma_window_spin': ma_window_spin
        }
        
        return algo_group, widgets
    
    @staticmethod
    def create_status_panel():
        """Create status display panel"""
        status_group = ModernGroupBox("Status")
        status_layout = QVBoxLayout()
        
        status_display = QTextEdit()
        status_display.setReadOnly(True)
        status_display.setMinimumHeight(200)
        status_display.setMaximumHeight(200)
        status_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        status_layout.addWidget(status_display)
        status_group.setLayout(status_layout)
        
        return status_group, status_display
    


