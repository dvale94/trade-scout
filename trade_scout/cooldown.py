from __future__ import annotations

from datetime import datetime, timedelta


class Cooldown:
    def __init__(self, minutes: int):
        self.cooldown = timedelta(minutes=minutes)
        self.last_alert: dict[tuple[str | None, str, str], datetime] = {}

    def allowed(self, symbol: str, tier: str, now: datetime, strategy_id: str | None = None) -> bool:
        key = (strategy_id, symbol, tier)
        last = self.last_alert.get(key)
        if not last:
            return True
        return (now - last) >= self.cooldown

    def mark(self, symbol: str, tier: str, now: datetime, strategy_id: str | None = None):
        self.last_alert[(strategy_id, symbol, tier)] = now
