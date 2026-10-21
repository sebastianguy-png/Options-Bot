# Registers a Windows Scheduled Task "OptionsBot" that starts the bot (headless, no
# console window) every time you log on, and starts it right now.
#
#   Right-click PowerShell → Run as administrator, then:
#       cd C:\Users\sebas\Options-Bot
#       Set-ExecutionPolicy -Scope Process Bypass
#       .\install_autostart.ps1
#
# Remove it again with .\uninstall_autostart.ps1.  The dashboard stays at
# http://127.0.0.1:5000 ; console output goes to logs\bot_<date>.log.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vbs  = Join-Path $here "run_bot_hidden.vbs"
if (-not (Test-Path $vbs)) { throw "run_bot_hidden.vbs not found next to this script" }

$action   = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`"" -WorkingDirectory $here
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "OptionsBot" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Options Bot dashboard + autotrader (headless; restarts itself)" -Force | Out-Null
Start-ScheduledTask -TaskName "OptionsBot"
Write-Host "Installed and started. The bot now starts at every logon (hidden). Dashboard: http://127.0.0.1:5000"
Write-Host "Logs: $here\logs    Kill switch for the autotrader: create a file named STOP_AUTOTRADE in $here"
