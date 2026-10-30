# Year-1 experiment: development and reference scripts

Run from the repository root with the project's Python. Everything before step 5 uses data before
2025-10-29 only; steps 5–7 read the test year and were run after the roster was frozen.

1. `fetch5m.py`: download spot 5-minute bars since 2022-07 into the data cache.
2. `daily_dev.py OUT.csv`: daily candidates (kept, improved, new) walk-forward out of sample, 2021 → 2025-10.
   `daily_fund.py`: the funding filter with the funding a live run sees. `ml_dev.py OUT.csv`: daily LightGBM.
3. `intra_wf.py OUTDIR` then `intra_eval.py OUTDIR [variants]`: 5-minute LightGBM walk-forward forecasts
   (live recipe; 1-week / 1-month / 3-month retraining) and the policies built on them, net of costs.
4. `daily_select.py`: the `main` blend (research selection on 2021 → 2025-10) → `models/year1/selection.json`.
5. `tradinglab --config config/year1.yaml simulate --books daily --start 2025-10-29 --end 2026-10-29 --out DIR`,
   then `wf_summary.py DIR`.
6. `reference_daily.py OUT.csv`, `reference_intra.py OUT.csv`: test-year backtests of every strategy, pruned ones
   included.
7. `prune_rolling.py [OUT.csv]`: the pruning rule applied at each year start since 2020.
