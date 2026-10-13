#!/usr/bin/env bash
# One live cycle + report, then commit the journal (cron-friendly).
#   */15 * * * * /opt/tradinglab/scripts/cycle_and_commit.sh >> /var/log/tradinglab.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/tradinglab cycle --report
.venv/bin/tradinglab status || true
if [ "${COMMIT_STATE:-1}" = "1" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  git add state reports/live.html 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -q -m "live: cycle $(date -u +%Y-%m-%dT%H:%MZ)"
    [ "${PUSH_STATE:-1}" = "1" ] && git push -q || true
  fi
fi
