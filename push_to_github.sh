#!/usr/bin/env bash
# Create a GitHub repo from this folder and push it.
# Usage (run from INSIDE the kd-lab-onpolicy/ directory):
#   bash push_to_github.sh [repo-name] [public|private]
# Defaults: repo-name = kd-lab-onpolicy-distillation, visibility = public
set -euo pipefail

REPO_NAME="${1:-kd-lab-onpolicy-distillation}"
VISIBILITY="${2:-public}"   # public | private

# basic git identity check
if ! git config --get user.name >/dev/null || ! git config --get user.email >/dev/null; then
  echo "Set your git identity first, e.g.:"
  echo '  git config --global user.name "Parv Patodia"'
  echo '  git config --global user.email "you@example.com"'
  exit 1
fi

git init -q
git add -A
git commit -qm "feat: on-policy distillation branch scaffold for kd-lab"
git branch -M main

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
  echo "Done: $(gh repo view "$REPO_NAME" --json url -q .url 2>/dev/null || echo "pushed to $REPO_NAME")"
else
  echo "gh CLI not found or not authenticated. Finish manually:"
  echo "  1) Create an EMPTY repo named '$REPO_NAME' at https://github.com/new"
  echo "     (do not add a README, .gitignore, or license; this folder already has them)."
  echo "  2) Then run:"
  echo "       git remote add origin git@github.com:<your-username>/$REPO_NAME.git"
  echo "       git push -u origin main"
fi
