@echo off
setlocal
cd /d "%~dp0"

where conda >nul 2>nul
if %errorlevel%==0 (
  set "CONDA=conda"
) else if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
  set "CONDA=%USERPROFILE%\anaconda3\Scripts\conda.exe"
) else if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
  set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
) else (
  echo Could not find conda. Install Anaconda or Miniconda first.
  pause
  exit /b 1
)

echo Starting the provrep API on http://127.0.0.1:8000  (close this window to stop it)
start "" "http://127.0.0.1:8000/docs"
call "%CONDA%" run -n provrep --no-capture-output provrep -c configs/default.yaml serve --port 8000
pause
