"""
PULSE Simulator – Windows Installer GUI
=========================================
This installer:
  1. Verifies Python 3.9+ is installed
  2. Lets the user choose an install directory
  3. Copies the PULSE source tree into that directory
  4. Creates a virtual environment
  5. Installs all required packages via pip (live progress)
  6. Creates a .bat launcher + Windows Desktop shortcut
  7. Optionally launches the app immediately

Build into a standalone exe with:
    pyinstaller installer/PULSE_Installer.spec
"""

import sys
import os
import subprocess
import shutil
import threading
import winreg
import time
import multiprocessing

# ── Bootstrap: make sure we can import PyQt5 even from frozen exe ──────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QFileDialog,
    QLineEdit, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
    QStackedWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPixmap, QIcon, QLinearGradient,
    QPainter, QPainterPath, QBrush
)

# ── Bundled requirements (embedded so installer is self-contained) ──────────
REQUIREMENTS = """PyQt5>=5.15.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pyqtgraph>=0.12.0
filterpy>=1.4.5
opencv-python>=4.5.0
"""

# ── Style constants ─────────────────────────────────────────────────────────
DARK_BG      = "#0D1117"
PANEL_BG     = "#161B22"
CARD_BG      = "#21262D"  # Lighter for better card separation
ACCENT       = "#58A6FF"  # Softer blue
ACCENT_HOVER = "#79C0FF"
SUCCESS      = "#3FB950"  # GitHub success green
ERROR        = "#F85149"  # GitHub error red
WARNING      = "#D29922"
TEXT_PRI     = "#C9D1D9"
TEXT_SEC     = "#8B949E"
BORDER       = "#30363D"
SHADOW       = "#000000"

STYLESHEET = f"""
QMainWindow, QWidget#root {{
    background-color: {DARK_BG};
}}
QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRI};
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
}}
QFrame#card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#header {{
    background-color: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
}}
QLabel {{
    background: transparent;
}}
QPushButton#primary {{
    background-color: {ACCENT};
    color: #0D1117;

    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton#primary:pressed {{
    background-color: {ACCENT};
    padding-top: 11px; /* slight press effect */
}}
QPushButton#primary:disabled {{
    background-color: #30363D;
    color: {TEXT_SEC};
}}
QPushButton#secondary {{
    background: transparent;
    color: {ACCENT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton#secondary:hover {{
    border-color: {TEXT_SEC};
    background-color: {PANEL_BG};
}}
QPushButton#browse {{
    background-color: {PANEL_BG};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 13px;
}}
QPushButton#browse:hover {{
    border-color: {TEXT_SEC};
    background-color: #30363D;
}}
QLineEdit {{
    background-color: {DARK_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    color: {TEXT_PRI};
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QProgressBar {{
    background-color: {BORDER};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}
QTextEdit {{
    background-color: {DARK_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    color: {TEXT_SEC};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}}
QScrollBar:vertical {{
    background: {DARK_BG};
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    min-height: 20px;
    border-radius: 5px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""

# ── Helper: Find Python - No changes needed here

# ── Helper: Get Source Dir - No changes needed here

# ── Custom Widgets ──────────────────────────────────────────────────────────

# ── Helper: find Python executable ─────────────────────────────────────────
def find_python():
    """Return the path to a Python 3.9+ executable, or None."""
    # When running as a frozen exe, sys.executable is the installer itself.
    # Including it would cause the installer to re-launch itself infinitely.
    if getattr(sys, 'frozen', False):
        candidates = ["python", "python3", "py"]
    else:
        candidates = [sys.executable, "python", "python3", "py"]
    for cand in candidates:
        try:
            result = subprocess.run(
                [cand, "--version"],
                capture_output=True, text=True, timeout=5
            )
            ver = result.stdout.strip() or result.stderr.strip()
            if "Python 3." in ver:
                parts = ver.split(".")
                minor = int(parts[1])
                if minor >= 9:
                    return cand, ver
        except Exception:
            continue
    return None, None


def get_source_dir():
    """Return the root of the PULSE source tree (works frozen + dev + installer subfolder)."""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe — source is bundled in _MEIPASS
        return sys._MEIPASS
    else:
        # Running as script — check if we're in installer/ subfolder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(script_dir)
        # If the parent contains src/ and assets/, it's the project root
        if os.path.isdir(os.path.join(parent, 'src')) and os.path.isdir(os.path.join(parent, 'assets')):
            return parent
        # Otherwise assume the script itself is in the project root
        if os.path.isdir(os.path.join(script_dir, 'src')):
            return script_dir
        return parent


# ── Worker thread ───────────────────────────────────────────────────────────
class InstallWorker(QThread):
    log        = pyqtSignal(str)        # log line
    progress   = pyqtSignal(int)        # 0-100
    step_done  = pyqtSignal(str, bool)  # (step_name, success)
    finished   = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, install_dir, python_exe, source_dir):
        super().__init__()
        self.install_dir = install_dir
        self.python_exe  = python_exe
        self.source_dir  = source_dir

    # ── helpers ──────────────────────────────────────────────────────
    def _run(self, cmd, cwd=None, env=None):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=cwd, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self.log.emit(line)
        proc.wait()
        return proc.returncode

    def _emit_step(self, name, ok):
        self.step_done.emit(name, ok)

    # ── main logic ───────────────────────────────────────────────────
    def run(self):
        install_dir = self.install_dir
        venv_dir    = os.path.join(install_dir, ".venv")
        src_dir     = self.source_dir

        # ── STEP 1: Copy source ──────────────────────────────────────
        self.log.emit("► Copying PULSE files to install directory…")
        try:
            if os.path.abspath(src_dir) != os.path.abspath(install_dir):
                for item in os.listdir(src_dir):
                    # Skip the installer folder itself, build, dist artefacts
                    if item in ('installer', 'build', 'dist', '__pycache__', '.git', '.venv'):
                        continue
                    s = os.path.join(src_dir, item)
                    d = os.path.join(install_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
            self.log.emit("  ✔ Files copied.")
            self._emit_step("Copy files", True)
        except Exception as e:
            self.log.emit(f"  ✘ Copy failed: {e}")
            self._emit_step("Copy files", False)
            self.finished.emit(False, f"File copy failed: {e}")
            return
        self.progress.emit(15)

        # ── STEP 2: Create virtual environment ───────────────────────
        self.log.emit("► Creating virtual environment…")
        rc = self._run([self.python_exe, "-m", "venv", venv_dir])
        if rc != 0:
            self.log.emit("  ✘ venv creation failed.")
            self._emit_step("Create virtual environment", False)
            self.finished.emit(False, "Could not create virtual environment.")
            return
        self.log.emit("  ✔ Virtual environment created.")
        self._emit_step("Create virtual environment", True)
        self.progress.emit(30)

        # ── STEP 3: Upgrade pip ──────────────────────────────────────
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        self.log.emit("► Upgrading pip…")
        self._run([pip_exe, "install", "--upgrade", "pip", "--quiet"])
        self.log.emit("  ✔ pip upgraded.")
        self.progress.emit(40)

        # ── STEP 4: Write requirements to temp file ──────────────────
        req_path = os.path.join(install_dir, "_requirements_temp.txt")
        with open(req_path, "w") as f:
            f.write(REQUIREMENTS)

        # ── STEP 5: Install packages ─────────────────────────────────
        self.log.emit("► Installing packages (this may take a few minutes)…")
        packages = [l.strip() for l in REQUIREMENTS.splitlines() if l.strip() and not l.startswith('#')]
        total = len(packages)
        base_progress = 40
        progress_range = 50  # 40→90

        rc = self._run([
            pip_exe, "install", "-r", req_path,
            "--progress-bar", "off"
        ])
        os.remove(req_path)

        if rc != 0:
            self.log.emit("  ✘ Package installation failed.")
            self._emit_step("Install packages", False)
            self.finished.emit(False, "Package installation failed. Check the log for details.")
            return
        self.log.emit("  ✔ All packages installed.")
        self._emit_step("Install packages", True)
        self.progress.emit(90)

        # ── STEP 6: Create launcher .bat ────────────────────────────
        python_in_venv = os.path.join(venv_dir, "Scripts", "python.exe")
        # run.py is the true entry point – it sets up sys.path and calls main()
        launcher_py    = os.path.join(install_dir, "run.py")
        bat_path       = os.path.join(install_dir, "PULSE_Simulator.bat")

        bat_content = f"""@echo off
cd /d "{install_dir}"
"{python_in_venv}" "{launcher_py}" %*
"""
        with open(bat_path, "w") as f:
            f.write(bat_content)
        self.log.emit(f"  ✔ Launcher created: {bat_path}")
        self._emit_step("Create launcher", True)
        self.progress.emit(95)

        # ── STEP 7: Desktop shortcut ─────────────────────────────────
        try:
            desktop       = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop, "PULSE Simulator.lnk")
            icon_path     = os.path.join(install_dir, "assets", "logo.ico")

            # Use pythonw.exe (no console window) – designed for GUI/PyQt5 apps.
            # Windows .lnk shortcuts REQUIRE backslashes in TargetPath/Arguments.
            pythonw_exe = os.path.join(venv_dir, "Scripts", "pythonw.exe")
            if not os.path.exists(pythonw_exe):
                pythonw_exe = python_in_venv  # fallback to python.exe

            # Write PowerShell script to a temp .ps1 file to avoid all
            # quoting / special-character issues with -Command.
            ps1_path = os.path.join(install_dir, "_create_shortcut.ps1")
            icon_line = ""
            if os.path.exists(icon_path):
                icon_line = f'$sc.IconLocation = "{icon_path}"'

            ps1_content = (
                f'$ws = New-Object -ComObject WScript.Shell\n'
                f'$sc = $ws.CreateShortcut("{shortcut_path}")\n'
                f'$sc.TargetPath       = "{pythonw_exe}"\n'
                f'$sc.Arguments        = \'"{launcher_py}"\'\n'
                f'$sc.WorkingDirectory = "{install_dir}"\n'
                f'$sc.Description      = "PULSE UWB Simulator"\n'
                f'$sc.WindowStyle      = 1\n'
                f'{icon_line}\n'
                f'$sc.Save()\n'
            )
            with open(ps1_path, "w", encoding="utf-8") as f:
                f.write(ps1_content)

            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", ps1_path],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            try:
                os.remove(ps1_path)   # clean up temp script
            except Exception:
                pass

            if result.returncode == 0:
                self.log.emit(f"  ✔ Desktop shortcut created: {shortcut_path}")
                self._emit_step("Create Desktop shortcut", True)
            else:
                self.log.emit(f"  ⚠ Shortcut PowerShell error: {result.stderr.strip()}")
                self._emit_step("Create Desktop shortcut", False)
        except Exception as e:
            self.log.emit(f"  ⚠ Could not create shortcut: {e}")
            self._emit_step("Create Desktop shortcut", False)

        self.progress.emit(100)
        self.finished.emit(True, bat_path)


# ── Step indicator widget ───────────────────────────────────────────────────
class StepRow(QWidget):
    def __init__(self, text, is_last=False, parent=None):
        super().__init__(parent)
        self.is_last = is_last
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Icon area (fixed width)
        self.icon_frame = QFrame()
        self.icon_frame.setFixedSize(24, 24)
        self.icon_frame.setStyleSheet(f"border-radius: 12px; border: 2px solid {BORDER}; background: {DARK_BG};")
        
        self.icon_lbl = QLabel("", self.icon_frame)
        self.icon_lbl.setAlignment(Qt.AlignCenter)
        self.icon_lbl.setGeometry(0, 0, 24, 24)
        self.icon_lbl.setStyleSheet("border: none; background: transparent; color: white;")

        # Container for the vertical line and the icon
        self.step_graphic = QWidget()
        self.step_graphic.setFixedWidth(24)
        vbox = QVBoxLayout(self.step_graphic)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.icon_frame)
        
        if not is_last:
            self.line = QFrame()
            self.line.setFixedWidth(2)
            self.line.setStyleSheet(f"background-color: {BORDER};")
            # Center the line
            line_cont = QWidget()
            line_layout = QHBoxLayout(line_cont)
            line_layout.setContentsMargins(11, 0, 0, 0) # 11px padding to center 2px line in 24px width
            line_layout.setSpacing(0)
            line_layout.addWidget(self.line)
            line_layout.addStretch()
            vbox.addWidget(line_cont, 1) # Expand to fill vertical space
        else:
            vbox.addStretch(1)

        layout.addWidget(self.step_graphic)
        
        self.text_lbl = QLabel(text)
        self.text_lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 14px; padding-top: 2px;")
        layout.addWidget(self.text_lbl)
        
        self.setStyleSheet("background: transparent;")
        self.setFixedHeight(50) # Taller rows

    def set_running(self):
        self.icon_frame.setStyleSheet(f"border-radius: 12px; border: 2px solid {ACCENT}; background: {DARK_BG};")
        self.text_lbl.setStyleSheet(f"color: {TEXT_PRI}; font-size: 14px; font-weight: 600; padding-top: 2px;")
        # Simple pulse animation could go here, for now just change color

    def set_done(self, success=True):
        color = SUCCESS if success else ERROR
        self.icon_frame.setStyleSheet(f"border-radius: 12px; border: none; background: {color};")
        self.icon_lbl.setText("✓" if success else "✕")
        self.text_lbl.setStyleSheet(f"color: {TEXT_PRI}; font-size: 14px; padding-top: 2px;")
        if not self.is_last and success:
            self.line.setStyleSheet(f"background-color: {SUCCESS};")


# ── Main window ─────────────────────────────────────────────────────────────
class InstallerWindow(QMainWindow):
    STEPS = [
        "Copy files",
        "Create virtual environment",
        "Install packages",
        "Create launcher",
        "Create Desktop shortcut",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PULSE Simulator – Setup")
        self.setFixedSize(700, 680) # Slightly taller for better spacing
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)

        # Icon
        icon_path = os.path.join(get_source_dir(), "assets", "logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.worker    = None
        self.bat_path  = None
        self.install_dir = os.path.join(os.path.expanduser("~"), "PULSE_Simulator")

        self.python_exe, self.python_ver = find_python()

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_header())

        # Use standard QStackedWidget for now over FadeStackWidget to ensure stability
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_setup_page())   # 0
        self.stack.addWidget(self._build_install_page()) # 1
        self.stack.addWidget(self._build_done_page())    # 2
        main_layout.addWidget(self.stack, 1)
        
        # Center the window on screen
        self.adjustSize()
        center = QApplication.desktop().availableGeometry().center()
        self.move(int(center.x() - self.width() * 0.5), int(center.y() - self.height() * 0.5))


    # ── Header ───────────────────────────────────────────────────────
    def _build_header(self):
        frame = QFrame()
        frame.setObjectName("header")
        frame.setFixedHeight(80)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(40, 0, 40, 0)
        layout.setSpacing(16)

        # Modern Icon Placeholder (or real icon if loaded)
        icon_lbl = QLabel("◈") 
        icon_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 32px; font-weight: bold;")
        
        # Try to load existing icon if available for better look
        icon_path = os.path.join(get_source_dir(), "assets", "logo.ico")
        if os.path.exists(icon_path):
             pix = QPixmap(icon_path)
             if not pix.isNull():
                icon_lbl.setPixmap(pix.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                icon_lbl.setText("")

        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title_lbl = QLabel("PULSE Simulator")
        title_lbl.setStyleSheet(f"color: {TEXT_PRI}; font-size: 20px; font-weight: 700;")
        sub_lbl = QLabel("Installation Wizard")
        sub_lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px;")
        title_block.addWidget(title_lbl)
        title_block.addWidget(sub_lbl)

        layout.addWidget(icon_lbl)
        layout.addLayout(title_block)
        layout.addStretch()

        ver_lbl = QLabel("v1.0.0")
        ver_lbl.setStyleSheet(f"color: {BORDER}; font-size: 12px; font-weight: 600; background: {PANEL_BG}; padding: 4px 8px; border-radius: 4px; border: 1px solid {BORDER};")
        layout.addWidget(ver_lbl)
        
        return frame

    # ── Page 0: Setup ─────────────────────────────────────────────────
    def _build_setup_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(24)

        # ── Python status card ───────────────────────────────────────
        py_card = self._card()
        py_layout = QVBoxLayout(py_card)
        py_layout.setContentsMargins(24, 20, 24, 20)
        
        py_heading = QLabel("Python Environment")
        py_heading.setStyleSheet(f"color: {TEXT_PRI}; font-weight: 600; font-size: 15px;")
        
        if self.python_exe:
            py_status = QLabel(f"Detected: {self.python_ver}")
            py_status.setStyleSheet(f"color: {SUCCESS}; font-size: 13px;")
            py_sub = QLabel(f"<span style='color:{TEXT_SEC}'>Using:</span> {self.python_exe}")
            py_sub.setStyleSheet("font-family: Consolas; font-size: 11px;")
        else:
            py_status = QLabel("❌ Python 3.9+ not found")
            py_status.setStyleSheet(f"color: {ERROR}; font-size: 13px; font-weight: 600;")
            py_sub = QLabel("Please install Python 3.9 or newer from python.org before proceeding.")
            py_sub.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px;")

        py_layout.addWidget(py_heading)
        py_layout.addSpacing(4)
        py_layout.addWidget(py_status)
        py_layout.addWidget(py_sub)
        layout.addWidget(py_card)

        # ── Packages card ────────────────────────────────────────────
        pkg_card = self._card()
        pkg_layout = QVBoxLayout(pkg_card)
        pkg_layout.setContentsMargins(24, 20, 24, 20)
        
        pkg_heading = QLabel("Dependencies")
        pkg_heading.setStyleSheet(f"color: {TEXT_PRI}; font-weight: 600; font-size: 15px;")
        pkg_layout.addWidget(pkg_heading)
        
        reqs_text = ", ".join([l.strip().split('>')[0] for l in REQUIREMENTS.splitlines() if l.strip()])
        pkg_desc = QLabel(f"The following packages will be installed in a virtual environment:\n\n{reqs_text}")
        pkg_desc.setWordWrap(True)
        pkg_desc.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px; line-height: 1.4;")
        pkg_layout.addWidget(pkg_desc)
        layout.addWidget(pkg_card)

        # ── Install dir card ─────────────────────────────────────────
        dir_card = self._card()
        dir_layout = QVBoxLayout(dir_card)
        dir_layout.setContentsMargins(24, 20, 24, 20)
        
        dir_heading = QLabel("Installation Directory")
        dir_heading.setStyleSheet(f"color: {TEXT_PRI}; font-weight: 600; font-size: 15px;")
        dir_layout.addWidget(dir_heading)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit(self.install_dir)
        self.dir_edit.setFixedHeight(36)
        browse_btn = QPushButton("Change Folder...")
        browse_btn.setObjectName("browse")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setFixedHeight(36)
        browse_btn.clicked.connect(self._browse)
        
        dir_row.addWidget(self.dir_edit, 1)
        dir_row.addWidget(browse_btn)
        dir_layout.addLayout(dir_row)
        layout.addWidget(dir_card)

        layout.addStretch()

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.install_btn = QPushButton("Start Installation")
        self.install_btn.setObjectName("primary")
        self.install_btn.setCursor(Qt.PointingHandCursor)
        self.install_btn.setEnabled(bool(self.python_exe))
        self.install_btn.clicked.connect(self._start_install)
        btn_row.addWidget(self.install_btn)
        layout.addLayout(btn_row)

        return page

    # ── Page 1: Installing ───────────────────────────────────────────
    def _build_install_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        status_lbl = QLabel("Installing PULSE Simulator…")
        status_lbl.setStyleSheet(f"color: {TEXT_PRI}; font-size: 15px; font-weight: 600;")
        layout.addWidget(status_lbl)

        # Steps list
        steps_card = self._card()
        steps_layout = QVBoxLayout(steps_card)
        steps_layout.setContentsMargins(24, 24, 24, 24)
        steps_layout.setSpacing(0) # Spacing handled by rows themselves
        
        self.step_rows = {}
        for i, step in enumerate(self.STEPS):
            is_last = (i == len(self.STEPS) - 1)
            row = StepRow(step, is_last=is_last)
            self.step_rows[step] = row
            steps_layout.addWidget(row)
        layout.addWidget(steps_card)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(14)
        layout.addWidget(self.progress_bar)

        # Log
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(200)
        layout.addWidget(self.log_box)

        layout.addStretch()
        return page

    # ── Page 2: Done ────────────────────────────────────────────────
    def _build_done_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 48, 40, 32)
        layout.setSpacing(24)
        layout.setAlignment(Qt.AlignCenter)

        # Success Card
        card = self._card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 40, 32, 40)
        card_layout.setSpacing(16)
        card_layout.setAlignment(Qt.AlignCenter)

        self.done_icon = QLabel("✔")
        self.done_icon.setAlignment(Qt.AlignCenter)
        self.done_icon.setStyleSheet(f"color: {SUCCESS}; font-size: 72px;")

        self.done_title = QLabel("Installation Complete!")
        self.done_title.setAlignment(Qt.AlignCenter)
        self.done_title.setStyleSheet(f"color: {TEXT_PRI}; font-size: 24px; font-weight: 700;")

        self.done_sub = QLabel(
            "PULSE Simulator has been successfully installed.\n"
            "You can now find a shortcut on your Desktop."
        )
        self.done_sub.setAlignment(Qt.AlignCenter)
        self.done_sub.setWordWrap(True)
        self.done_sub.setStyleSheet(f"color: {TEXT_SEC}; font-size: 14px; line-height: 1.5;")
        
        card_layout.addWidget(self.done_icon)
        card_layout.addWidget(self.done_title)
        card_layout.addWidget(self.done_sub)
        
        layout.addStretch()
        layout.addWidget(card)
        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondary")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.close)

        self.launch_btn = QPushButton("🚀  Launch PULSE")
        self.launch_btn.setObjectName("primary")
        self.launch_btn.setCursor(Qt.PointingHandCursor)
        self.launch_btn.clicked.connect(self._launch_app)

        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        btn_row.addWidget(self.launch_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        return page

    # ── Helper: card frame ───────────────────────────────────────────
    def _card(self):
        f = QFrame()
        f.setObjectName("card")
        
        # Add slight shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        f.setGraphicsEffect(shadow)
        
        return f

    # ── Browse ───────────────────────────────────────────────────────
    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose install directory", self.dir_edit.text()
        )
        if path:
            self.dir_edit.setText(path)

    # ── Start install ────────────────────────────────────────────────
    def _start_install(self):
        install_dir = self.dir_edit.text().strip()
        if not install_dir:
            return
        os.makedirs(install_dir, exist_ok=True)

        self.stack.setCurrentIndex(1)  # show install page

        self.worker = InstallWorker(
            install_dir=install_dir,
            python_exe=self.python_exe,
            source_dir=get_source_dir()
        )
        self.worker.log.connect(self._on_log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.step_done.connect(self._on_step_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

        # Animate first step as running
        if self.STEPS:
            self.step_rows[self.STEPS[0]].set_running()

    # ── Slots ─────────────────────────────────────────────────────────
    def _on_log(self, line: str):
        self.log_box.append(line)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_step_done(self, step: str, ok: bool):
        if step in self.step_rows:
            self.step_rows[step].set_done(ok)
        # Mark next as running
        steps = self.STEPS
        try:
            idx = steps.index(step)
            if idx + 1 < len(steps):
                self.step_rows[steps[idx + 1]].set_running()
        except ValueError:
            pass

    def _on_finished(self, success: bool, info: str):
        if success:
            self.bat_path = info
            QTimer.singleShot(600, lambda: self.stack.setCurrentIndex(2))
        else:
            # Show error in log and stay on page
            self.log_box.append(f"\n✘ Installation failed: {info}")
            self.progress_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {ERROR}; border-radius: 6px; }}"
            )

    def _launch_app(self):
        if self.bat_path and os.path.exists(self.bat_path):
            subprocess.Popen(
                ["cmd", "/c", "start", "", self.bat_path],
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        self.close()


# ── Entry point ─────────────────────────────────────────────────────────────
def main():
    # Required when frozen with PyInstaller to prevent infinite re-spawn
    multiprocessing.freeze_support()

    # HiDPI - Must be set before QCoreApplication is created
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setApplicationName("PULSE Installer")

    win = InstallerWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
