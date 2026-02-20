# Refactored Localization App - Module Documentation

## Overview

The application has been refactored into a modular architecture to improve maintainability, scalability, and code organization. The GUI components are structured into managers, panels, and windows.

## New File Structure

```
GUI/
├── managers/                  # Manager classes
│   ├── plot_manager.py           # Plot creation and visualization management
│   ├── event_handlers.py         # Mouse events and user interactions
│   ├── simulation_manager.py     # Simulation update loop and state
│   ├── nlos_manager.py           # NLOS zone creation and editing
│   ├── trajectory_manager.py     # Custom trajectory management
│   ├── file_manager.py           # Save/load operations
│   └── simulation_recorder.py    # Simulation data recording and playback
├── panels/                    # UI panel components
│   ├── control_panels.py         # UI control panel factory
│   ├── Plots.py                  # Localization error plots
│   ├── dockable_panel.py         # Docking system for panels
│   └── timeline_widget.py        # Timeline slider for playback
├── windows/                   # Window and dialog components
│   ├── Distance_plot_window.py    # Distance measurements window
│   ├── imu_window.py              # IMU data visualization window
│   ├── nlos_config_window.py      # NLOS configuration dialog
│   ├── comparison_window.py       # Kalman filter comparison window
│   ├── filter_selection_dialog.py # Filter selection dialog
│   ├── Nlos_aware_params_window.py # NLOS-aware parameters window
│   ├── localization_error_plot.py # Error plot window
│   ├── cir_window.py              # Channel Impulse Response visualization
│   ├── algorithm_creation_window.py # Custom algorithm creator
│   ├── image_import_window.py     # Floor plan image importer
│   └── moving_nlos_window.py      # Moving NLOS zone configuration
├── __init__.py                # Package initialization
├── theme.py                   # Theme configuration
└── README.md                  # This file

Localization_app.py             # Main application file
```

## Module Descriptions

### Managers Module (`managers/`)

#### 1. `plot_manager.py` (PlotManager)
**Purpose:** Manages all plotting operations using PyQtGraph.
**Key Features:**
-   Creates and configures position plots.
-   Manages plot items (trajectories, points, markers).
-   Updates anchor visualizations with LOS/NLOS coloring.
-   Handles measurement line drawing.

#### 2. `event_handlers.py` (EventHandler)
**Purpose:** Handles all mouse events and user interactions on the plot area.
**Key Features:**
-   Manages interaction modes (dragging, drawing, selecting).
-   Coordinates signals between UI and managers.

#### 3. `simulation_manager.py` (SimulationManager)
**Purpose:** Manages the core simulation loop and logic.
**Key Features:**
-   Main simulation update loop (timer-based).
-   Position estimation using selected algorithms.
-   Measurement collection and validation.
-   Updates visualizations via PlotManager.

#### 4. `nlos_manager.py` (NLOSManager)
**Purpose:** Manages NLOS (Non-Line-of-Sight) zone creation and logic.
**Key Features:**
-   Creates and edits NLOS zones (Rectangles, Polygons).
-   Calculates signal attenuation and delays based on zone parameters.
-   Manages static and moving obstacles.

#### 5. `trajectory_manager.py` (TrajectoryManager)
**Purpose:** Handles custom trajectory creation and management.
**Key Features:**
-   Manages trajectory planning and drawing.
-   Interpolates paths for smooth movement.

#### 6. `file_manager.py` (FileManager)
**Purpose:** Manages file operations for map configurations.
**Key Features:**
-   Saves/loads complete map configurations (JSON).
-   Serializes anchors, zones, and simulation settings.

#### 7. `simulation_recorder.py` (SimulationRecorder) **[NEW]**
**Purpose:** Records simulation states for timeline playback and analysis.
**Key Features:**
-   Snapshot-based recording of tag position, estimates, errors, and anchor states.
-   Circular buffer for memory efficiency during infinite simulations.
-   Supports time-based retrieval for trajectory reconstruction.

### Panels Module (`panels/`)

#### 1. `control_panels.py` (ControlPanelFactory)
**Purpose:** Factory class for creating UI control panels.
**Key Features:**
-   Generates standard UI widgets for simulation controls.

#### 2. `Plots.py` (LocalizationErrorPlot)
**Purpose:** Real-time visualization of localization errors.

#### 3. `dockable_panel.py` (DockablePanel, PanelManager) **[NEW]**
**Purpose:** Provides a flexible docking system for UI panels.
**Key Features:**
-   Allows panels to be docked in the main window or floated as separate windows.
-   Manages panel state (open, closed, floating).
-   Persists layout preferences (potentially).

#### 4. `timeline_widget.py` (TimelineWidget) **[NEW]**
**Purpose:** Timeline slider for navigating recorded simulation history.
**Key Features:**
-   Play/Pause replay of simulation.
-   Step forward/backward controls.
-   Scrubbing through recorded history.
-   Playback speed control.

### Windows Module (`windows/`)

#### 1. `cir_window.py` (CIRWindow) **[NEW]**
**Purpose:** Visualizes the Channel Impulse Response (CIR) for all anchors.
**Key Features:**
-   Grid view of CIR plots (paginated).
-   Visualizes First Path (ToA) and Detection Threshold.
-   Color-coded curves for LOS/NLOS status.
-   Real-time updates.

#### 2. `algorithm_creation_window.py` (AlgorithmCreationWindow) **[NEW]**
**Purpose:** GUI for creating and saving custom localization algorithms.
**Key Features:**
-   Integrated code editor with syntax highlighting.
-   Template generation based on selected features (IMU, Kalman, NLOS).
-   Sanitizes filenames and class names.
-   Saves directly to the user algorithms directory.

#### 3. `image_import_window.py` (ImageImportWindow) **[NEW]**
**Purpose:** Tool for importing floor plan images and extracting walls.
**Key Features:**
-   Loads images (PNG, JPG).
-   Line detection using Hough Transform with tunable parameters.
-   Scaling and origin offset configuration.
-   Converts detected lines into NLOS zones/walls.

#### 4. `moving_nlos_window.py` (MovingNLOSWindow) **[NEW]**
**Purpose:** Configuration dialog for moving NLOS obstacles.
**Key Features:**
-   Defines trajectory (Start/End) and movement dynamics (Speed, Rotation).
-   Configures shape (Circle, Square, Polygon) and dimensions.
-   Sets channel parameters (Path Loss, Shadowing, Multipath).

#### 5. Existing Windows:
-   **Distance_plot_window.py**: Real-time distance measurements from anchors.
-   **imu_window.py**: IMU sensor data visualization.
-   **nlos_config_window.py**: Setup for static NLOS zones.
-   **comparison_window.py**: Compare different algorithms/filters side-by-side.
-   **filter_selection_dialog.py**: Select active estimation filters.
-   **Nlos_aware_params_window.py**: Tune parameters for NLOS-aware algorithms.

## Main Application Flow

The `Localization_app.py` initializes the Core components and the GUI Managers.
1.  **Initialization**: Sets up `PlotManager`, `SimulationManager`, and `PanelManager`.
2.  **UI Setup**: Creates the main window with a central plot area and dockable panels.
3.  **Simulation Loop**: `SimulationManager` triggers updates, `SimulationRecorder` captures state, and `PlotManager` refreshes the display.

## Adding New Features

### To add a new Window:
1.  Create the window class in `src/gui/windows/`.
2.  Add a method in `Localization_app.py` to open it (e.g., menu action).

### To add a new Algorithm:
1.  Use the `AlgorithmCreationWindow` from the GUI.
2.  Or manually add a file to `src/user_algorithms/`.

### To add a new Panel:
1.  Create the panel content widget.
2.  Register it with `PanelManager` in `Localization_app.py`.
