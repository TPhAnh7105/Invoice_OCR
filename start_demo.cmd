@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay .venv. Xem README_DEMO.md de cai dat.
  pause
  exit /b 1
)
echo Mo trinh duyet tai http://127.0.0.1:8000
".venv\Scripts\python.exe" run_demo.py
if errorlevel 1 pause
