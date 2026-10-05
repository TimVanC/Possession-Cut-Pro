@echo off
rem Makes the Possession Cut engine start quietly every time you log in, and starts it now.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\autostart.ps1" install
pause
