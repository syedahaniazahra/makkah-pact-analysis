@echo off
REM run_pipeline.bat - convenience wrapper around run_pipeline.py.
REM
REM This is the "or a .bat wrapper" option: point Windows Task Scheduler's
REM Action directly at THIS file (Program/script: full path to this .bat,
REM leave Arguments blank), or just double-click it for a quick manual run.
REM It is NOT required - pointing Task Scheduler straight at python.exe
REM with run_pipeline.py as the Argument works exactly the same (see the
REM Task Scheduler setup notes) - this file exists only to make the
REM "Program/script" field a single, simple, double-click-able thing.
REM
REM %~dp0 = the folder THIS .bat file lives in, with a trailing backslash -
REM so this always runs from the right folder no matter what Task
REM Scheduler's own "Start in" field is set to (or left blank), and no
REM matter where this .bat happens to get moved to alongside the rest of
REM the project.
cd /d "%~dp0"

REM Uses whatever "python" resolves to on PATH for whatever account runs
REM this (your own login, if Task Scheduler is set to "Run only when user
REM is logged on" - see the setup notes for the alternative). Run
REM `where python` in a Command Prompt once to confirm this is the SAME
REM interpreter your project already uses (the one with streamlit/pandas/
REM networkx/etc. installed) - if it isn't, replace "python" below with
REM that command's full path instead, e.g.:
REM   "C:\Users\zahra\AppData\Local\Programs\Python\Python314\python.exe" run_pipeline.py
python run_pipeline.py >> pipeline_run_console.log 2>&1
