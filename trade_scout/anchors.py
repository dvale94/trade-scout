from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from . import market


@dataclass
class AnchorState:
    regime: Optional[str] = None
    close: Optional[float] = None
    ma50: Optional[float] = None
    ma200: Optional[float] = None
    ts: Optional[datetime] = None


class AnchorTracker:
    """
    Market-agnostic anchor tracker.
    market.py decides which anchor symbol(s) exist and how to classify regime.
    """

    def __init__(self):
        self.anchor_symbol: str = market.anchor_symbol()
        self.state = AnchorState()


    def is_anchor(self, symbol: str) -> bool:
        return self.anchor_symbol is not None and symbol == self.anchor_symbol


    def update_from_computed(self, computed: dict, ts: datetime) -> None:
        # computed comes from score_candidate() on the anchor's df
        self.state.close = computed.get("close")
        self.state.ma50 = computed.get("ma50")
        self.state.ma200 = computed.get("ma200")
        self.state.regime = market.classify_anchor_regime(computed)
        self.state.ts = ts


    def inject_into_computed(self, computed: dict) -> dict:
        """
        Returns a copy of computed with anchor_* fields injected for logging.
        """
        out = dict(computed)
        if not self.anchor_symbol:
            return out

        out["anchor_symbol"] = self.anchor_symbol
        out["anchor_regime"] = self.state.regime
        out["anchor_close"] = self.state.close
        out["anchor_ma50"] = self.state.ma50
        out["anchor_ma200"] = self.state.ma200
        return out