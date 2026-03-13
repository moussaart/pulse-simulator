#!/usr/bin/env python
"""
UWB Localization Simulation - Entry Point

This script serves as the main entry point for the UWB localization simulation application.
It sets up the Python path correctly and launches the application.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add the project root to Python path to enable proper imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Seed writable user data on first frozen run (no-op in development)
from src.utils.resource_loader import seed_user_data
seed_user_data()

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

# Now import and run the application
from src.app.localization_main import main

if __name__ == "__main__":
    main()
