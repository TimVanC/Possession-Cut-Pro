@echo off
rem Windows launcher: runs dev.ps1 regardless of the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev.ps1" %*
