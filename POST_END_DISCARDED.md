# Post-end activity discarded (vault-2w close-out)

The experiment ended 2026-10-29 23:47:45 UTC. The first close-out (shift 20261029T1919, records commit
8befff9) liquidated every account correctly: intraday at 23:48:15 UTC, daily at 23:48:30 UTC, at live
quotes. Writing the final status then failed (pandas could not parse the close-out mark's timestamp
next to the whole-second bar marks), so FINISHED.json was not written.

Each retry shift (20261029T2353 … 20261030T0100) re-ran an intraday trading cycle before closing out
again, so the intraday accounts traded bars after the end (4,952 orders; about $9.6k of fees and
$3.4k of spread/impact across the 24 intraday accounts). The daily book did not trade again.

Those post-end trades are not part of the experiment. state/, csv/ and status/ are restored to the
first close-out (commit 8befff9); the discarded state stays in this branch's history at commit a9c14e7.
The bug is fixed on main: the close-out only liquidates (it never runs a trading cycle) and timestamps
are parsed as ISO 8601.
