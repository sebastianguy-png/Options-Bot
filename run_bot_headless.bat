@echo off
REM Options Bot — unattended mode: no browser window, restarts itself if it ever exits.
REM Exit code 3 = another copy is already running on the port; then we stop instead of looping.
cd /d "%~dp0"
if not exist logs mkdir logs
set OPTIONS_BOT_HEADLESS=1
:loop
py options_bot.py --headless
set CODE=%errorlevel%
echo [%date% %time%] bot exited with code %CODE% >> logs\restarts.log
if "%CODE%"=="3" goto end
timeout /t 30 /nobreak > nul
goto loop
:end
