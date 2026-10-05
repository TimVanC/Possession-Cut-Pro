@echo off
rem Starts the Possession Cut engine. Leave this window open while you use the app.
rem First run sets everything up, which takes a few minutes.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev.ps1" --no-frontend
if errorlevel 1 pause
