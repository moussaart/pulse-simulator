import sys
import os
import shutil
from pathlib import Path


def is_frozen() -> bool:
    """Check if running as a PyInstaller frozen executable."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def get_resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a bundled read-only resource.
    Works for both development and PyInstaller frozen executables.

    Args:
        relative_path: Path relative to the project root (e.g., 'assets/logo.svg')

    Returns:
        Absolute path to the resource.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # In development, use the project root
        # This assumes this file is in src/utils/
        # So project root is ../../
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    return os.path.join(base_path, relative_path)


def get_data_path(relative_path: str) -> str:
    """
    Get the absolute path for writable user data files.

    In development mode: resolves relative to the project root (unchanged behavior).
    In frozen exe mode:  resolves relative to %APPDATA%/PULSE_Simulator/ so that
                         user-created trajectories, configs, and exports persist.

    Args:
        relative_path: Path relative to the project root (e.g., 'data/trajectories')

    Returns:
        Absolute path to the writable data location.
    """
    if is_frozen():
        base_path = get_writable_user_dir("PULSE_Simulator")
    else:
        # In development, use the project root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    full_path = os.path.join(base_path, relative_path)
    return full_path


def get_writable_user_dir(app_name: str = "PULSE_Simulator") -> str:
    """
    Get a safe writable directory for user data (logs, saved maps).
    Uses %APPDATA%/app_name on Windows.
    """
    home = Path.home()
    if sys.platform == "win32":
        app_data = os.getenv('APPDATA')
        if app_data:
            path = os.path.join(app_data, app_name)
        else:
            path = os.path.join(home, 'AppData', 'Roaming', app_name)
    else:
        path = os.path.join(home, f'.{app_name.lower()}')

    os.makedirs(path, exist_ok=True)
    return path


def seed_user_data():
    """
    On first run of a frozen executable, copy the bundled 'data/' directory
    from the read-only _MEIPASS temp folder to the writable user data directory.

    This ensures sample trajectories, configs, and export directories are
    available for the user to read and write to.

    In development mode this is a no-op.
    """
    if not is_frozen():
        return

    user_data_dir = get_writable_user_dir("PULSE_Simulator")
    bundled_data_dir = os.path.join(sys._MEIPASS, 'data')

    if not os.path.exists(bundled_data_dir):
        return

    target_data_dir = os.path.join(user_data_dir, 'data')

    # Only seed if the target doesn't exist yet (first run)
    if not os.path.exists(target_data_dir):
        try:
            shutil.copytree(bundled_data_dir, target_data_dir)
            print(f"Seeded user data directory: {target_data_dir}")
        except Exception as e:
            print(f"Warning: Could not seed user data: {e}")
            # Create at least the essential subdirectories
            for subdir in ['trajectories', 'configs', 'exports', 'maps']:
                os.makedirs(os.path.join(target_data_dir, subdir), exist_ok=True)
