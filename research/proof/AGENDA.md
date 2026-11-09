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
  positive alpha but uneven, drawdown 60 % on fresh coins), R7 (R6 on coins listed >= 1 year: positive everywhere
  but weak on set B, Sharpe 0.56, alpha t 1.6). Fresh set B is now used; pre-register set C before the next
  confirmation.

## Queue (each goes through PROTOCOL.md; fresh confirmation data must be pre-registered first)

1. **Fresh set B: pre-registered** (`fresh_universe_b.json`, 60 coins, 7 Nov 2026). Use it only to confirm a frozen rule.
2. ~~Settlement drift~~ (8 Nov): dead end. Minus the market, high-funding coins trail by only 4-8 bp over 4 h
   (costs ~24 bp hedged); raw drops are market-wide and reverse within 8 h. Registry id 12.
3. ~~Quarterly futures basis~~ (8 Nov): not better now. BTC/ETH quarterlies lock 4.6-5.6 %/yr against +6.5 %/yr
   of BTC perpetual funding over the last 30 days. Revisit only if quarterly basis runs well above funding.
   Registry id 13.
4. ~~R6 repair~~ (8 Nov): R7 (age >= 1 year, rank, inverse vol) failed on set B (registry 15).
6. ~~Basis mean reversion~~ (8 Nov): real convergence (gaps > 60 bp close ~50 bp in 4 h vs ~34 bp costs) but the
   opportunities vanished after 2022 (~20 hours in all of 2023). Registry 16.
7. ~~Pairs mean reversion~~ (9 Nov): no edge (+4.9 bp at 4 h vs ~12 bp costs; negative at 12-24 h). Registry 17.
5. ~~Carry capital efficiency~~ (9 Nov): keep 1.5x. Daily spikes against the short leg reach 46-66 % on 1 coin-day in
   1,000 (12 coins / fresh coins), with extremes above 100 %. **Deployment requirement:** run carry where spot counts
   as collateral for the short (unified / portfolio margin); otherwise the short can be liquidated in a squeeze and
   leave the spot leg unhedged.

## Next ideas (to be added as the queue empties)

- Pre-register **fresh set C** (the next 60 coins by volume) before the next confirmation.
- Carry: predicted funding (Binance publishes the next rate continuously) to enter or leave a day earlier.
- Cross-exchange funding differences (OKX is reachable): needs a history source before it can be tested.
