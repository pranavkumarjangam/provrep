@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Full-graph training on the GPU (all 2.25M Hetionet edges).
echo  Results go to artifacts\gpu - the CPU model in artifacts\ is kept.
echo  Expect roughly 20-60 minutes. Keep the laptop plugged in.
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

if not exist data\raw mkdir data\raw
for %%F in (hetionet-v1.0-nodes.tsv hetionet-v1.0-edges.sif.gz indications.tsv) do (
  if not exist "data\raw\%%F" if exist "%USERPROFILE%\%%F" copy /y "%USERPROFILE%\%%F" "data\raw\%%F" >nul
)
if not exist "data\raw\hetionet-v1.0-edges.sif.gz" (
  echo Raw Hetionet files not found - downloading them...
  call "%CONDA%" run -n provrep --no-capture-output provrep -c configs/hetionet_rtx3050.yaml download
  if errorlevel 1 goto :fail
)
set "RUN=call "%CONDA%" run -n provrep --no-capture-output provrep -c configs/hetionet_rtx3050.yaml"
echo [1/4] Building the full knowledge graph...
%RUN% build-kg
if errorlevel 1 goto :fail
echo [2/4] Training the R-GCN on the GPU...
%RUN% train
if errorlevel 1 goto :fail
echo [3/4] Fitting calibrators...
%RUN% calibrate
if errorlevel 1 goto :fail
echo [4/4] Evaluating on the held-out test diseases...
%RUN% eval
if errorlevel 1 goto :fail
echo.
echo === Demo triage with the GPU-trained model ===
%RUN% triage "rheumatoid arthritis" --show 10
echo.
echo Done. Report: artifacts\gpu\eval\eval_report.md  - send it to Claude to compare with the CPU run.
pause
exit /b 0
:fail
echo.
echo FAILED at the step above. If it says "CUDA out of memory", edit configs\hetionet_rtx3050.yaml
echo and set hidden_dim: 64, then run this again. Otherwise send the error text to Claude.
pause
exit /b 1
