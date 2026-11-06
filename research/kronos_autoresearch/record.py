"""Log the last run (run.log) to results.tsv and keep or discard strategy.py.

    python research/kronos_autoresearch/record.py [--control] "<one-line description>"
keep: the score beats the best so far, so best_strategy.py becomes strategy.py; discard or crash: strategy.py is
restored from best_strategy.py. The very first run is the baseline and is always kept. --control logs a comparison
run (for example the best rule without Kronos's inputs); it is never kept.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG, TSV = HERE.parent.parent / "run.log", HERE / "results.tsv"
STRAT, BEST, RUNS = HERE / "strategy.py", HERE / "best_strategy.py", HERE / "runs"


def field(text: str, name: str):
    m = re.search(rf"^{name}:\s*(\S+)", text, re.M)
    return float(m.group(1)) if m else None


def main() -> None:
    control = "--control" in sys.argv
    desc = " ".join(a for a in sys.argv[1:] if a != "--control").strip() or "(no description)"
    text = LOG.read_text() if LOG.exists() else ""
    score = field(text, "score")
    rows = list(csv.DictReader(TSV.open(), delimiter="\t")) if TSV.exists() else []
    kept = [float(r["score"]) for r in rows if r["status"] == "keep"]
    best = max(kept) if kept else None
    if score is None:
        status = "crash"
    elif control:
        status = "control"
    elif best is None or score > best:
        status = "keep"
    else:
        status = "discard"
    n = len(rows) + 1
    RUNS.mkdir(exist_ok=True)
    shutil.copy(STRAT, RUNS / f"{n:03d}_{status}.py")
    code_hash = hashlib.sha1(STRAT.read_bytes()).hexdigest()[:7]
    if status == "keep":
        shutil.copy(STRAT, BEST)
    elif BEST.exists():
        shutil.copy(BEST, STRAT)
    new = not TSV.exists()
    with TSV.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(["run", "code", "score", "total_return", "max_drawdown", "turnover_per_day", "status", "description"])
        w.writerow([n, code_hash, f"{score:.4f}" if score is not None else "", field(text, "total_return") or "",
                    field(text, "max_drawdown") or "", field(text, "turnover_per_day") or "", status, desc])
    print(f"run {n}: {status} (score {score}, best before {best})")


if __name__ == "__main__":
    main()
