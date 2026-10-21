@echo off
REM Runs setup_tailscale.ps1 as administrator (one UAC click) — see the .ps1 for what it does.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -NoExit -File \"%~dp0setup_tailscale.ps1\"'"
