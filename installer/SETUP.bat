@echo off
:: ============================================================
:: PULSE Simulator – Setup Launcher
:: Double-click this file to start the installer.
:: ============================================================
setlocal EnableDelayedExpansion
title PULSE Simulator Setup

echo.
echo  ============================================================
echo    PULSE Simulator – Setup
echo  ============================================================
echo.

:: ── Find Python ─────────────────────────────────────────────
set PYTHON_CMD=

:: Try 'python' command
python --version >nul 2>&1
if !errorlevel! == 0 (
    set PYTHON_CMD=python
    goto :found_python
)

:: Try 'py' launcher (Windows py.exe)
py --version >nul 2>&1
if !errorlevel! == 0 (
    set PYTHON_CMD=py
    goto :found_python
)

:: Try 'python3'
python3 --version >nul 2>&1
if !errorlevel! == 0 (
    set PYTHON_CMD=python3
    goto :found_python
)

:: Not found
echo  [ERROR] Python was not found on this system.
echo.
echo  Please install Python 3.9 or higher from:
echo    https://www.python.org/downloads/
echo.
echo  Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found_python
echo  Python found: %PYTHON_CMD%
for /f "tokens=*" %%V in ('!PYTHON_CMD! --version 2^>^&1') do set PYTHON_VER=%%V
echo  Version: %PYTHON_VER%
echo.

:: ── Check PyQt5 is available for the installer GUI ──────────
%PYTHON_CMD% -c "import PyQt5" >nul 2>&1
if !errorlevel! neq 0 (
    echo  Installing PyQt5 for the setup wizard...
    %PYTHON_CMD% -m pip install --quiet PyQt5
    if !errorlevel! neq 0 (
        echo  [ERROR] Could not install PyQt5. Check your internet connection.
        pause
        exit /b 1
    )
)

:: ── Launch the graphical installer ──────────────────────────
echo  Launching PULSE Setup Wizard...
echo.
%PYTHON_CMD% "%~dp0install_PULSE.py"

endlocal
