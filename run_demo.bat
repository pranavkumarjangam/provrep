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

echo === Triage: rheumatoid arthritis (a held-out test disease) ===
call "%CONDA%" run -n provrep --no-capture-output provrep -c configs/default.yaml triage "rheumatoid arthritis" --show 10
echo.
echo === Triage: migraine (a held-out test disease) ===
call "%CONDA%" run -n provrep --no-capture-output provrep -c configs/default.yaml triage "migraine" --show 10
echo.
echo Full briefs (evidence, verified claims, verdicts) are saved in artifacts\triage\
echo To triage another disease, run in an Anaconda Prompt:
echo    conda activate provrep
echo    cd /d "%~dp0"
echo    provrep -c configs/default.yaml triage "disease name"
pause
