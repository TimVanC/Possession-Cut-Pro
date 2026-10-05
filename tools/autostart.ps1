# Start the Possession Cut engine in the background at login (Windows).
#   tools\autostart.ps1 install   add the login item and start the engine now
#   tools\autostart.ps1 remove    stop the engine and remove the login item
#   tools\autostart.ps1 status
# The login item is an ordinary shortcut in your Startup folder (Win+R, shell:startup);
# deleting it by hand does the same as "remove".
param([ValidateSet("install", "remove", "status")][string]$Action = "install")
$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$link = Join-Path ([Environment]::GetFolderPath("Startup")) "Possession Cut.lnk"
$py = Join-Path $root ".venv\Scripts\python.exe"
$launch = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$root\dev.ps1`" --no-frontend --background"

if ($Action -eq "status") {
    if (Test-Path $link) { Write-Host "Autostart is installed: $link" } else { Write-Host "Autostart is not installed." }
    exit 0
}

if ($Action -eq "remove") {
    if (Test-Path $py) { & $py -m possession_cut.dev --stop }
    if (Test-Path $link) { Remove-Item $link }
    Write-Host "Possession Cut no longer starts at login."
    exit 0
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = (Get-Command powershell.exe).Source
$shortcut.Arguments = $launch
$shortcut.WorkingDirectory = $root
$shortcut.WindowStyle = 7
$shortcut.Description = "Possession Cut engine"
$shortcut.Save()

Write-Host "Starting the Possession Cut engine (first run installs what it needs; give it a few minutes)..."
Start-Process -FilePath $shortcut.TargetPath -ArgumentList $launch -WorkingDirectory $root -WindowStyle Hidden
Write-Host ""
Write-Host "Done. The engine now starts by itself whenever you log in."
Write-Host "Open the Possession Cut page in your browser and upload a game."
Write-Host "Its log is in data\engine.log. To undo this, run remove-autostart.cmd."
