# Removes the "OptionsBot" scheduled task and stops the running bot.
# Run from an elevated PowerShell:  .\uninstall_autostart.ps1
$ErrorActionPreference = "SilentlyContinue"
Stop-ScheduledTask -TaskName "OptionsBot"
Unregister-ScheduledTask -TaskName "OptionsBot" -Confirm:$false
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*options_bot.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Write-Host "OptionsBot task removed and the bot stopped."
