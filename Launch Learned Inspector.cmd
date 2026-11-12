@echo off
cd /d "%~dp0"
if not exist "build\livingworld.dll" goto build
if not exist "build\livingworld_policy.dll" goto build
goto launch
:build
python tools\build.py
if errorlevel 1 goto failed
:launch
python -S tools\inspector.py --native-policy inference/artifacts/curriculum-v3.lwgru
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Living World could not start. Install Python 3.11+ with Tkinter, then retry.
echo A source checkout also requires building both native libraries with a C++20 compiler.
pause
exit /b 1
