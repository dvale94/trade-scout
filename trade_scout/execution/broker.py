from __future__ import annotations

from typing import Any, Mapping

from trade_scout.execution.alpaca_adapter import (
    cancel_order,
    get_order_by_broker_order_id,
    get_order_by_client_order_id,
    submit_limit_order,
    submit_market_order,
    submit_stop_order,
)
from trade_scout.execution.models import BrokerOrderResult


def submit_validated_order(
    *,
    request_id: str,
    alert_row: Mapping[str, Any],
) -> BrokerOrderResult:
    return submit_market_order(request_id=request_id, alert_row=alert_row)


def find_existing_broker_order(request_id: str) -> BrokerOrderResult | None:
    return get_order_by_client_order_id(request_id)


def find_existing_broker_order_by_client_order_id(client_order_id: str) -> BrokerOrderResult | None:
    return get_order_by_client_order_id(client_order_id)


def submit_protective_stop_order(
    *,
    client_order_id: str,
    symbol: str,
    side: str,
    qty: float,
    stop_price: float,
    limit_price: float | None = None,
) -> BrokerOrderResult:
    return submit_stop_order(
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        stop_price=stop_price,
        limit_price=limit_price,
    )


def cancel_broker_order(broker_order_id: str) -> BrokerOrderResult | None:
    return cancel_order(broker_order_id)


def submit_take_profit_order(
    *,
    client_order_id: str,
    symbol: str,
    side: str,
    qty: float,
    limit_price: float,
) -> BrokerOrderResult:
    return submit_limit_order(
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        limit_price=limit_price,
    )


def fetch_broker_order_by_id(broker_order_id: str) -> BrokerOrderResult | None:
    return get_order_by_broker_order_id(broker_order_id)


def find_broker_order_by_client_order_id(client_order_id: str) -> BrokerOrderResult | None:
    return get_order_by_client_order_id(client_order_id)
