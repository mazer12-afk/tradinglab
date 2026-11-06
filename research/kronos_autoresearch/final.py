"""The one-time holdout test (Apr - Oct 2026), run after the loop stopped: the loop's best strategy, the same rule
without Kronos (control) and the baseline. Writes holdout.json.

    python research/kronos_autoresearch/final.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import evaluate  # noqa: E402

RULES = {"best (Kronos volatility forecast)": "best_strategy.py",
         "control (realised volatility, no Kronos)": "runs/015_control.py",
         "baseline (sign of Kronos's median path)": "runs/001_keep.py"}


def main() -> None:
    out = {}
    for name, f in RULES.items():
        spec = importlib.util.spec_from_file_location("s", HERE / f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        dev, hold = evaluate(m.fit, m.predict, verbose=False), evaluate(m.fit, m.predict, final=True, verbose=False)
        out[name] = {"file": f, "dev": dev, "holdout": hold}
        print(f"{name:42s} dev Sharpe {dev['sharpe']:+.2f} ({dev['total_return']:+.1%})   "
              f"holdout Sharpe {hold['sharpe']:+.2f} ({hold['total_return']:+.1%}, worst drawdown {hold['max_drawdown']:.1%})")
    (HERE / "holdout.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
