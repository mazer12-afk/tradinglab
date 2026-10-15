#!/usr/bin/env bash
# Check out the experiment's records branch into ./records (created as an orphan branch on first use).
# Usage: scripts/experiment_records.sh [branch]   (default: experiment.records_branch in config/experiment.yaml)
# Needs GITHUB_TOKEN and GITHUB_REPOSITORY; writes branch= and finished= to $GITHUB_OUTPUT when set.
set -euo pipefail
branch="${1:-}"
if [ -z "$branch" ]; then
  branch=$(sed -n 's/^[[:space:]]*records_branch:[[:space:]]*\([^[:space:]#]*\).*/\1/p' config/experiment.yaml | head -1)
fi
[ -n "$branch" ] || { echo "no records branch configured" >&2; exit 1; }
url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if git ls-remote --exit-code --heads "$url" "$branch" >/dev/null 2>&1; then
  git clone -q --depth 1 --single-branch -b "$branch" "$url" records
else
  git init -q -b "$branch" records
  git -C records remote add origin "$url"
fi
git -C records config user.name "tradinglab-bot"
git -C records config user.email "tradinglab-bot@users.noreply.github.com"
finished=false
[ -f records/FINISHED.json ] && finished=true
echo "records branch: $branch (finished=$finished)"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "branch=$branch" >> "$GITHUB_OUTPUT"
  echo "finished=$finished" >> "$GITHUB_OUTPUT"
fi
