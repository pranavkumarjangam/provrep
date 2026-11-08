@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  provrep GPU setup for your RTX 3050: conda env "provrep"
echo  with CUDA PyTorch. Downloads about 2.5 GB, 10-20 minutes.
echo  (Run this INSTEAD of setup_windows.bat, or after it.)
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
  echo [1/5] Creating environment...
  call "%CONDA%" create -y -n provrep -c conda-forge --override-channels python=3.11
  if errorlevel 1 goto :fail
) else (
  echo [1/5] Environment "provrep" already exists, reusing it.
)
echo [2/5] Removing any CPU-only PyTorch...
call "%CONDA%" run -n provrep --no-capture-output python -m pip uninstall -y torch >nul 2>nul
echo [3/5] Installing CUDA PyTorch...
call "%CONDA%" run -n provrep --no-capture-output python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :fail
echo [4/5] Installing provrep and its dependencies...
call "%CONDA%" run -n provrep --no-capture-output python -m pip install -e ".[dev,llm]"
if errorlevel 1 goto :fail
echo [5/5] Checking the GPU and running the tests...
call "%CONDA%" run -n provrep --no-capture-output python scripts\check_gpu.py
call "%CONDA%" run -n provrep --no-capture-output python -m pytest
if errorlevel 1 goto :fail
echo.
echo GPU setup complete. Double-click train_gpu.bat to train on the full graph.
pause
exit /b 0
:fail
echo.
echo Setup FAILED at the step above. Copy the error text and send it to Claude.
pause
exit /b 1
