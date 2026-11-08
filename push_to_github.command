#!/bin/bash
# Double-click to push provrep to GitHub, or run:  bash push_to_github.command
cd "$(dirname "$0")"
GHUSER="pranavkumarjangam"
REPO="provrep"
VISIBILITY="--public"    # change to --private for a private repo

command -v git >/dev/null || { echo "Git not installed. Install Xcode command line tools: xcode-select --install"; read -n1; exit 1; }

[ -d .git ] || { git init; git branch -M main; }
git config user.name  >/dev/null 2>&1 || git config user.name  "$GHUSER"
git config user.email >/dev/null 2>&1 || git config user.email "$GHUSER@users.noreply.github.com"

git add .
git commit -m "provrep: provenance-grounded drug repurposing (full pipeline, UI, docs)" || echo "(nothing new to commit)"

if command -v gh >/dev/null 2>&1; then
  gh auth status >/dev/null 2>&1 || gh auth login
  gh repo view "$GHUSER/$REPO" >/dev/null 2>&1 || gh repo create "$GHUSER/$REPO" $VISIBILITY --source . --remote origin
fi

git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$GHUSER/$REPO.git"
echo "Pushing to https://github.com/$GHUSER/$REPO ..."
git push -u origin main
echo
echo "Done (if no errors above). Open: https://github.com/$GHUSER/$REPO"
read -n1 -p "Press any key to close..."
