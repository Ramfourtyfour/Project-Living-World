@echo off
cd /d "%~dp0"
if not exist "build\livingworld.dll" (
  python tools\build.py
  if errorlevel 1 goto failed
)
python -S tools\inspector.py --social-baseline
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Living World could not start. Install Python 3.11+ with Tkinter, then retry.
pause
exit /b 1
