"""Stateful risk overlay shared by the backtester and the live engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from tradinglab.config import RiskConfig


@dataclass
class DrawdownGuard:
    """Scales exposure down as drawdown deepens; flattens at the halt level.

    After a halt the book stays flat for ``cooldown_bars`` bars, then the
    high-water mark is reset to current equity and trading resumes. This keeps
    an autonomous system from either blowing up or staying switched off forever.
    """

    cfg: RiskConfig
    cooldown_bars: int = 20
    peak: float = field(default=float("nan"))
    halted_bars_left: int = 0
    halts: int = 0

    def update(self, equity: float) -> float:
        """Register the latest equity and return the exposure multiplier in [0, 1]."""
        if self.peak != self.peak:  # NaN → first observation
            self.peak = equity
        if self.halted_bars_left > 0:
            self.halted_bars_left -= 1
            if self.halted_bars_left == 0:
                self.peak = equity
            return 0.0
        self.peak = max(self.peak, equity)
        dd = 1.0 - equity / self.peak if self.peak > 0 else 0.0
        if dd >= self.cfg.dd_halt:
            self.halted_bars_left = self.cooldown_bars
            self.halts += 1
            return 0.0
        if dd <= self.cfg.dd_derisk_start:
            return 1.0
        frac = (dd - self.cfg.dd_derisk_start) / (self.cfg.dd_halt - self.cfg.dd_derisk_start)
        return 1.0 - frac * (1.0 - self.cfg.dd_floor_scale)

    def state(self) -> dict:
        return {"peak": self.peak, "halted_bars_left": self.halted_bars_left, "halts": self.halts}

    def restore(self, s: dict) -> None:
        self.peak = float(s.get("peak", float("nan")))
        self.halted_bars_left = int(s.get("halted_bars_left", 0))
        self.halts = int(s.get("halts", 0))
