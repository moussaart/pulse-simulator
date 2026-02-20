"""
build_installer.py
==================
Helper script to build PULSE_Installer.exe using PyInstaller.

Usage (from project root):
    python build_installer.py

Output:
    dist/PULSE_Installer.exe   ← distribute this to users
"""

import os
import subprocess
import shutil
import sys


def build_installer():
    print("=" * 60)
    print("  PULSE Simulator – Building Installer EXE")
    print("=" * 60)

    spec_path = os.path.join("installer", "PULSE_Installer.spec")

    if not os.path.exists(spec_path):
        print(f"✘  Spec file not found: {spec_path}")
        sys.exit(1)

    # Clean old build artefacts (only installer-related ones)
    for folder in ("build", "dist"):
        if os.path.exists(folder):
            # Only clean if it contains our target; avoid removing app build
            target = os.path.join(folder, "PULSE_Installer" if folder == "dist" else "install_PULSE")
            if os.path.exists(target):
                print(f"  Removing {target}…")
                shutil.rmtree(target)

    print("\nRunning PyInstaller…")
    try:
        subprocess.run(
            ["pyinstaller", spec_path, "--clean", "--noconfirm"],
            check=True
        )
    except FileNotFoundError:
        print("✘  pyinstaller not found. Install it with:  pip install pyinstaller")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"✘  Build failed: {e}")
        sys.exit(1)

    # Verify
    exe_path = os.path.join("dist", "PULSE_Installer.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n✔  Installer built successfully!")
        print(f"   Location : {os.path.abspath(exe_path)}")
        print(f"   Size     : {size_mb:.1f} MB")
        print("\n   Distribute 'PULSE_Installer.exe' to your users.")
        print("   When they run it, it will set up PULSE and create a Desktop shortcut.")
    else:
        print("✘  Expected output not found after build.")
        sys.exit(1)


if __name__ == "__main__":
    build_installer()
