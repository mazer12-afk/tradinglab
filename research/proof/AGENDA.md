# Research agenda (autonomous)

The standing goal: a method that passes every line of `PROTOCOL.md`, on unseen data, and then its forward test.
Every rule tried goes in `registry.csv`.

## Where things stand (10 Nov 2026)

- **Passed the backtest bar: funding carry** (R2, R3, R4). Seven tests on unseen data: Binance later years, other
  coins, fresh coins, the survivorship-free universe (8 of 9 checks; 9 of 9 with announced-delisting exits) and Deribit.
  Report: `reports/proof/report.html`.
- **Failed:** R1 low volatility, R5 trend, R6 / R7 / R8 funding-factor books. Nine more ideas showed no edge on design
  data (registry 12, 13, 16 - 18, 20 - 22, 27).
- The search for a second method is closed for now: every family with a plausible mechanism and testable data has
  been through the protocol.

## Running

- **Forward test of R2, R3, R4**, from 8 Nov 2026, daily (`carry-forward.yml`, branch `experiment/carry-forward`).
  Check every day:
  - the job ran;
  - `archive_check`: price returns must match the archive exactly (`max_price_return_diff` 0). Funding, positions
    and daily P&L are compared once the month's funding file is published (early December);
  - returns and costs look sane.
- **On or after 6 Dec 2026:** write the forward verdict per PROTOCOL.md section 5 into `results/forward_verdict.json`,
  update the report and README, and tell the user.

## Only if new data or a new mechanism appears

- Reverse carry (spot short, perpetual long when funding is negative): needs margin borrow-rate history.
- Options (volatility premium): needs option price history.
- Any new rule: all Binance history through Oct 2026 has now been used (the wide panel covers every coin), so a new
  rule can only be confirmed on data that does not exist yet (freeze it, then wait) or on another exchange.
