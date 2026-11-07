# When is a trading method "proven"?

Written on 7 Nov 2026, before any of the confirmatory tests below were run. A method counts as proven only if it
passes **every** test here, on data that played no part in designing it. Every rule tested is logged in
`registry.csv` (one line per rule and data set), so the statistics can account for how many were tried.

## 1. Freeze before testing

The rule is fully specified in code and committed (the commit hash is recorded in the registry) **before** the
confirmatory data is touched. After that, nothing about it may change. A changed rule is a new rule and starts
again at step 1.

## 2. Confirmatory data

Data that played no part in designing the rule:
- **other periods:** for rules designed on Jul 2025 – Oct 2026, the years 2021 – 2023;
- **other coins:** the lab's 18 coins outside the 12 traded ones (TRX, XLM, ATOM, NEAR, UNI, ETC, FIL, AAVE, HBAR,
  ALGO, VET, INJ, SUI, APT, ARB, OP, ICP, LDO), each from its listing.

Each method's registry line names its design data and its confirmatory data.

## 3. Costs

Perpetuals: 5 bp taker fee plus 1 bp slippage on every unit traded, and the actual funding paid or received. Spot:
10 bp fee plus 1 bp slippage. No maker rebates, no fill assumptions that a 1-minute bar cannot support.

## 4. The bar

On the confirmatory data, after costs:

| Test | Requirement |
|---|---|
| Overall | Sharpe ≥ 1.0, and probabilistic Sharpe vs 0 ≥ 0.99 |
| Selection | Deflated Sharpe ≥ 0.95, with the number of trials in the registry for this family |
| Consistency | Profitable in at least 75 % of calendar years, and on both the original and the other coins |
| Drawdown | Worst drawdown no deeper than −20 % at the tested size |
| Robustness | Every neighbouring setting (each parameter halved and doubled) profitable; doubled costs still profitable |
| Not a market bet | Alpha against the equal-weight market positive with t ≥ 2; a market-neutral rule must have \|beta\| < 0.2 |

## 5. Forward test

A rule that passes runs as a live paper-trading test for at least 4 weeks, through the same code path, and must
trade as the simulator says it would (same positions, costs within 10 %). Four weeks cannot prove profit; they prove
the implementation.

## 6. Reporting

Every result is reported, including failures. A method that fails any line is reported as failed, with the line it
failed.
