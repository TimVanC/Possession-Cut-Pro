@echo off
rem Stops the background engine and removes it from the programs that start when you log in.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\autostart.ps1" remove
pause
