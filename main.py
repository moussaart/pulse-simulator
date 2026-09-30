#!/usr/bin/env python
"""
UWB Localization Simulation - Entry Point

This script serves as the main entry point for the UWB localization simulation application.
It sets up the Python path correctly and launches the application.

Usage:
    python main.py
"""

import multiprocessing
import sys
from pathlib import Path

# Add the project root to Python path to enable proper imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Must precede any launch: in a frozen build a child process re-executes this
# executable, and without freeze_support() it starts a second GUI instead of a
# worker. See src/api/pulse_freeze_hook.py.
multiprocessing.freeze_support()

# Seed writable user data on first frozen run (no-op in development)
from src.utils.resource_loader import seed_user_data
seed_user_data()


def _run_headless_worker() -> int:
    """Run the training worker instead of the GUI.

    Imported lazily and before the Qt GUI so that a headless or charts-only run
    never loads a display stack it does not need.
    """
    from src.api.ai_training_worker import WORKER_FLAG, main as worker_main, strip_worker_flag

    argv = strip_worker_flag(sys.argv[1:])
    if not argv:
        print(f"{WORKER_FLAG} needs arguments; see --help")
        return 2
    return worker_main(argv)


def _is_worker_invocation() -> bool:
    from src.api.ai_training_worker import is_worker_invocation

    return is_worker_invocation()


# Set AppUserModelID for Windows taskbar icon
import ctypes
import os 
try:
    # Explicitly tell Windows this is a unique application
    # This allows the taskbar icon to be different from the default Python icon
    myappid = 'irisa.pulse.simulator.v1' 
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass


if __name__ == "__main__":
    # Headless worker mode short-circuits the GUI entirely.
    if _is_worker_invocation():
        sys.exit(_run_headless_worker())

    # Now import and run the application
    from src.app.localization_main import main

    main()
