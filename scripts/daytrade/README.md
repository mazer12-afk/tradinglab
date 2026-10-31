# Day-trading research scripts

Run from the repository root (`load.py` must be importable: pass this folder as the first argument).

1. `fetch1m.py`: download 1-minute USD-M perpetual bars since 2022-07 into the data cache.
2. `intra_edge.py DIR`: edge per trade of the lab's 5-minute models before costs (development data).
3. `events_a.py DIR [OUT.csv]`: event studies on Aug 2023 – Aug 2024.
4. `grid_a.py DIR OUT.csv` (`GRID=2` for the extended grid), `grid_b.py`: fading extreme moves, both directions.
5. `diag.py DIR`: results split by direction and by market-wide vs coin-specific moves.
6. `crash_grid.py DIR A|B OUT.csv`: the long-only (buy-the-crash) grid per period.
7. `test_year.py DIR OUTDIR`: the single test-year run of `config/daytrade_frozen.json`, plus reference runs.
