from __future__ import annotations

from typing import Any, Mapping

from trade_scout.execution.alpaca_adapter import get_order_by_client_order_id, submit_market_order
from trade_scout.execution.models import BrokerOrderResult


def submit_validated_order(
    *,
    request_id: str,
    alert_row: Mapping[str, Any],
) -> BrokerOrderResult:
    return submit_market_order(request_id=request_id, alert_row=alert_row)


def find_existing_broker_order(request_id: str) -> BrokerOrderResult | None:
    return get_order_by_client_order_id(request_id)
