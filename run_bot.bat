@echo off
REM Launch Options Bot at Above Normal priority
cd /d "%~dp0"
start "Options Bot" /ABOVENORMAL python options_bot.py
