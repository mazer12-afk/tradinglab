# Research agenda (autonomous)

The standing goal: a method that passes every line of `PROTOCOL.md`, on unseen data, and then its forward test.
Every rule tried goes in `registry.csv`. Work on one item at a time; commit each result.

## Running

- **Forward test of R2, R3, R4 (funding carry)**, from 8 Nov 2026, daily (`carry-forward.yml`, branch
  `experiment/carry-forward`). Check every day:
  - the job ran;
  - `archive_check` shows the live and archive books agree (`max_daily_net_diff` ~ 0, `max_weight_diff` ~ 0);
  - returns and costs look sane.
  On or after 6 Dec 2026, write the forward verdict per PROTOCOL.md section 5, then tell the user.

## Status of rules

- **Passed the backtest bar:** R2, R3, R4 (funding carry). All have thin yields in 2025-26 (about 0-2 % a year) because
  funding falls in down markets.
- **Failed:** R1 (low volatility: a disguised short-market bet), R5 (trend), R6 (cross-sectional funding factor:
  positive alpha but uneven, drawdown 60 % on fresh coins).

## Queue (each goes through PROTOCOL.md; fresh confirmation data must be pre-registered first)

1. **Pre-register fresh set B:** the next 60 USDT perpetuals by volume that also trade on spot, beyond the 57 in
   `fresh_universe.json`. It is the confirmation data for new carry and cross-sectional rules (all 2020-2026 data of
   the 87 coins used so far has been seen).
2. **Settlement drift (day trading):** after settlements whose rate is above 30 %/yr, prices fell about 14 bp over
   the next 3 hours (12 coins, 2020-2023, design data). A short from settlement + 1 h to + 4 h barely covers 12 bp
   of costs. Look for a sharper version: bigger rates, coin-relative returns. Confirm on other coins and later
   years.
3. **Quarterly futures basis:** Binance delivery contracts (e.g. BTCUSDT_YYMMDD) lock in the basis at entry. Test
   whether rolling the richest contract beats perpetual funding carry.
4. **R6 repair:** the funding factor had alpha everywhere (t 0.7 / 3.0 / 2.0), but its drawdowns came from a few
   coins. Try per-coin risk caps and inverse-volatility sizing. Design on the 30 known coins; confirm on fresh set B.
5. **Carry capital efficiency:** the 1.5x capital per notional is conservative. Measure the worst intraday spike
   against the short leg in all data to justify a tighter margin (it scales returns, not the test outcome).
