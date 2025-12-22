@echo off
REM VISCA Camera Control Application Launcher
REM Double-click this file to launch the app

echo ============================================================
echo VISCA Camera Video Tracking Application
echo ============================================================
echo.

REM Launch the Python application
python app.py

REM Pause so user can see any error messages
if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    pause
)
