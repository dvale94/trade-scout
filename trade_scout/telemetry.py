from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import DefaultDict, Optional

from . import journal, config


@dataclass
class TelemetryTracker:
    strategy_profile: str
    timeframe_min: int
    flush_every: timedelta = timedelta(hours=1)

    # counters
    bars_seen: int = 0
    anchor_bars_seen: int = 0
    insufficient_history: int = 0
    indicator_missing: int = 0
    below_score: int = 0
    alerts_sent: int = 0
    skipped_tight_stop: int = 0

    # score stats
    scores_n: int = 0
    scores_sum: float = 0.0
    score_min: Optional[float] = None
    score_max: Optional[float] = None

    tier_selected: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    cooldown_blocked: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    alerts_by_tier: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
    next_flush: datetime = field(init=False)

    def __post_init__(self):
        self.next_flush = self.window_start + self.flush_every


    # ---- record helpers ----
    def record_bar(self, is_anchor: bool = False) -> None:
        self.bars_seen += 1
        if is_anchor:
            self.anchor_bars_seen += 1


    def record_insufficient_history(self) -> None:
        self.insufficient_history += 1


    def record_indicator_missing(self) -> None:
        self.indicator_missing += 1


    def record_below_score(self) -> None:
        self.below_score += 1


    def record_score(self, score: float) -> None:
        self.scores_n += 1
        self.scores_sum += float(score)
        self.score_min = float(score) if self.score_min is None else min(self.score_min, float(score))
        self.score_max = float(score) if self.score_max is None else max(self.score_max, float(score))


    def record_tier_selected(self, tier: str) -> None:
        self.tier_selected[tier] += 1


    def record_cooldown_blocked(self, tier: str) -> None:
        self.cooldown_blocked[tier] += 1


    def record_alert_sent(self, tier: str) -> None:
        self.alerts_sent += 1
        self.alerts_by_tier[tier] += 1


    def record_skipped_tight_stop(self) -> None:
        self.skipped_tight_stop += 1


    # ---- flush ----
    def maybe_flush(self, now_utc: datetime) -> None:
        if now_utc < self.next_flush:
            return

        avg_score = (self.scores_sum / self.scores_n) if self.scores_n > 0 else None

        row = {
            "timestamp_utc": self.next_flush.astimezone(timezone.utc).isoformat(),
            "market_type": config.MARKET_TYPE,
            "strategy_profile": self.strategy_profile,
            "timeframe_min": self.timeframe_min,

            "bars_seen": self.bars_seen,
            "anchor_bars_seen": self.anchor_bars_seen,
            "insufficient_history": self.insufficient_history,
            "indicator_missing": self.indicator_missing,
            "below_score": self.below_score,
            "alerts_sent": self.alerts_sent,
            "skipped_tight_stop": self.skipped_tight_stop,

            "scores_n": self.scores_n,
            "score_min": self.score_min,
            "score_avg": avg_score,
            "score_max": self.score_max,

            "tier_selected": json.dumps(dict(self.tier_selected), separators=(",", ":")),
            "cooldown_blocked": json.dumps(dict(self.cooldown_blocked), separators=(",", ":")),
            "alerts_by_tier": json.dumps(dict(self.alerts_by_tier), separators=(",", ":")),
        }

        journal.log_telemetry(row)

        # Optional: single hourly print (safe and not noisy)
        print(
            f"[TELEMETRY] bars={row['bars_seen']} alerts={row['alerts_sent']} "
            f"below_score={row['below_score']} ind_missing={row['indicator_missing']} "
            f"score_min={row['score_min']} score_avg={row['score_avg']} score_max={row['score_max']} "
            f"tiers={row['tier_selected']} cooldown={row['cooldown_blocked']}"
        )

        self._reset_window()


    def _reset_window(self) -> None:
        # advance window
        self.window_start = self.next_flush
        self.next_flush = self.window_start + self.flush_every

        # reset counters
        self.bars_seen = 0
        self.anchor_bars_seen = 0
        self.insufficient_history = 0
        self.indicator_missing = 0
        self.below_score = 0
        self.alerts_sent = 0
        self.skipped_tight_stop = 0

        self.scores_n = 0
        self.scores_sum = 0.0
        self.score_min = None
        self.score_max = None

        self.tier_selected.clear()
        self.cooldown_blocked.clear()
        self.alerts_by_tier.clear()