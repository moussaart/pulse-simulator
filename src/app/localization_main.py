import sys
from PyQt5.QtWidgets import QApplication
from src.app.Localization_app import LocalizationApp

from src.core.parallel.gpu_backend import gpu_manager
from src.utils.resource_loader import seed_user_data

def main():
    # Seed writable user data on first frozen run (no-op in development)
    seed_user_data()

    # Print hardware acceleration status
    print("\n" + "="*60)
    print("  UWB Localization Simulation - Hardware Status")
    print("="*60)
    print(f"  {gpu_manager.get_status_string()}")
    print("="*60 + "\n")

    app = QApplication(sys.argv)
    
    # Set application icon
    try:
        from PyQt5.QtGui import QIcon
        from src.utils.resource_loader import get_resource_path
        import os
        icon_path = get_resource_path(os.path.join('assets', 'logo.ico'))
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception as e:
        print(f"Error setting app icon: {e}")

    window = LocalizationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 