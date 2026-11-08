@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Push provrep to GitHub
echo  Edit GHUSER below if your username is not pranavkumarjangam.
echo ============================================================
set "GHUSER=pranavkumarjangam"
set "REPO=provrep"

where git >nul 2>nul || (echo Git is not installed. Install from https://git-scm.com/download/win & pause & exit /b 1)

if not exist .git (
  git init
  git branch -M main
)
git config user.name  >nul 2>nul || git config user.name  "%GHUSER%"
git config user.email >nul 2>nul || git config user.email "%GHUSER%@users.noreply.github.com"

git add .
git commit -m "provrep: provenance-grounded drug repurposing (full pipeline, UI, docs)" || echo (nothing new to commit)

REM create the repo on GitHub if the GitHub CLI is installed and logged in
where gh >nul 2>nul
if %errorlevel%==0 (
  gh repo view %GHUSER%/%REPO% >nul 2>nul || gh repo create %GHUSER%/%REPO% --public --source . --remote origin
)

git remote get-url origin >nul 2>nul || git remote add origin https://github.com/%GHUSER%/%REPO%.git
echo Pushing to https://github.com/%GHUSER%/%REPO% ...
git push -u origin main
echo.
echo If it asked you to sign in, run this file again after signing in.
echo Done. Open: https://github.com/%GHUSER%/%REPO%
pause
