# Putting provrep on GitHub

This session can't push to your account directly (its GitHub access is locked to other
repositories). Do it from your PC, where you're signed in as yourself. Two ways:

## Easiest — double-click
1. Create an **empty** repo on github.com: New repository, name it `provrep`, Public,
   do NOT add a README/.gitignore/license, Create.
2. Double-click `push_to_github.bat` in this folder.
   - If your username isn't `pranavkumarjangam`, open the .bat in Notepad and change `GHUSER`.
   - If a sign-in window appears, finish it, then run the file again.
3. Open https://github.com/pranavkumarjangam/provrep

## Manual (Git Bash / PowerShell, run inside this folder)
```
git init
git branch -M main
git add .
git commit -m "provrep: provenance-grounded drug repurposing"
git remote add origin https://github.com/pranavkumarjangam/provrep.git
git push -u origin main
```

Your trained model and raw data are intentionally **not** pushed (see `.gitignore`:
`artifacts/`, `data/`). The repo stays small — just code, configs, tests and docs.
To share the trained model too, attach `provrep_colab_results.zip` to a GitHub Release.
