@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  provrep setup: creates a conda env "provrep" (Python 3.11),
echo  installs CPU PyTorch + the project, then runs the tests.
echo  Takes about 5-10 minutes and ~1 GB of disk.
echo ============================================================

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

call "%CONDA%" env list | findstr /b /c:"provrep " >nul
if errorlevel 1 (
  echo [1/4] Creating environment...
  call "%CONDA%" create -y -n provrep -c conda-forge --override-channels python=3.11
  if errorlevel 1 goto :fail
) else (
  echo [1/4] Environment "provrep" already exists, reusing it.
)
echo [2/4] Installing CPU PyTorch...
call "%CONDA%" run -n provrep --no-capture-output python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
echo [3/4] Installing provrep and its dependencies...
call "%CONDA%" run -n provrep --no-capture-output python -m pip install -e ".[dev,llm]"
if errorlevel 1 goto :fail
echo [4/4] Running the test suite...
call "%CONDA%" run -n provrep --no-capture-output python -m pytest
if errorlevel 1 goto :fail
echo.
echo Setup complete. Double-click run_demo.bat next.
pause
exit /b 0
:fail
echo.
echo Setup FAILED at the step above. Copy the error text and send it to Claude.
pause
exit /b 1
