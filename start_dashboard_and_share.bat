@echo off
REM start_dashboard_and_share.bat
REM
REM Double-click this whenever you want to open the dashboard AND get a
REM shareable link to it. It starts two things, each in its own window:
REM   1. The Streamlit dashboard itself, listening on localhost:8501.
REM   2. An ngrok tunnel, which exposes that local port as a public
REM      https:// URL - this is the link people use to view your
REM      dashboard live, from anywhere, while both windows stay open.
REM
REM SECURITY NOTE: while these two windows stay open, ANYONE who has the
REM link can open your dashboard - there is no login on it. That's fine
REM for showing a professor or classmates on purpose, but don't leave this
REM running unattended for long stretches if you don't want the link
REM working the whole time. Closing the "Public Link (ngrok)" window (or
REM the "Dashboard" window) shuts that half down; close both to fully stop
REM sharing.
REM
REM ONE-TIME SETUP REQUIRED before this works (see the setup notes
REM delivered alongside this file):
REM   1. Put ngrok.exe in THIS SAME folder (next to this .bat file).
REM   2. Run `ngrok config add-authtoken <your token>` once, from Command
REM      Prompt, in this folder.
REM   3. Replace YOUR-DEV-DOMAIN-HERE below with the exact free dev domain
REM      shown in your ngrok dashboard under Domains (looks something like
REM      abc123xyz.ngrok-free.app or .ngrok-free.dev - copy it exactly as
REM      shown there, don't guess the ending).

cd /d "%~dp0"

echo Starting Streamlit dashboard...
start "Dashboard" python -m streamlit run app.py --server.headless true

echo Waiting 15 seconds for Streamlit to finish starting...
timeout /t 15 /nobreak >nul

echo Starting ngrok tunnel...
start "Public Link (ngrok)" ngrok.exe http 8501 --url https://dioxide-false-declared.ngrok-free.dev

echo.
echo Both are starting in their own windows.
echo Your shareable link is: https://dioxide-false-declared.ngrok-free.dev
echo (Also shown inside the ngrok window itself, under "Forwarding".)
echo.
echo Close the "Dashboard" and "Public Link (ngrok)" windows when you are
echo done sharing - closing THIS window does not stop them.
pause
