# vault-2w analysis

Report: `report.html` (published at https://claude.ai/artifact/XL5CKjvXnK83tQc4567pHb).

- `fetch_klines.py <data_dir> [5m|5m30|1h]` — spot closes for the window from data-api.binance.vision.
- `analyse.py <state_dir> <data_dir> <experiment.json> <out_dir>` — per-account results, vault twin vs control,
  market context, attribution, regressions, bootstrap ranges; run on `state/` as of the close-out (commit 9875ea4,
  identical to the final state apart from one flat mark per account written at 01:29 UTC).
- `out/accounts.csv`, `out/vault_vs_control.csv`, `out/results.json` — the outputs the report is built from.

Every account figure was reproduced independently from the raw tables to the cent.
