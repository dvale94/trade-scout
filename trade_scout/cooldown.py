from __future__ import annotations

from datetime import datetime, timedelta


class Cooldown:
    def __init__(self, minutes: int):
        self.cooldown = timedelta(minutes=minutes)
        self.last_alert: dict[tuple[str, str], datetime] = {}

    def allowed(self, symbol: str, tier: str, now: datetime) -> bool:
        key = (symbol, tier)
        last = self.last_alert.get(key)
        if not last:
            return True
        return (now - last) >= self.cooldown

    def mark(self, symbol: str, tier: str, now: datetime):
        self.last_alert[(symbol, tier)] = now
