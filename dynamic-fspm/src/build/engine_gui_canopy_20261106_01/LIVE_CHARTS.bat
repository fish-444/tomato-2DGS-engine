@echo off
rem Fallback chart window: reads the newest gui\gui_runs\<time>\daily.csv every 5 s (read-only).
chcp 65001 >nul
cd /d "%~dp0gui"
python live_charts.py
if errorlevel 1 pause
