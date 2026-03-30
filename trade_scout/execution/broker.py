from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trade_scout.execution.alpaca_adapter import (
    cancel_order_by_broker_order_id as cancel_alpaca_order_by_broker_order_id,
    get_order_by_broker_order_id,
    get_order_by_client_order_id,
    submit_market_order,
    submit_limit_order as submit_alpaca_limit_order,
    submit_stop_limit_order as submit_alpaca_stop_limit_order,
    submit_stop_order as submit_alpaca_stop_order,
)
from trade_scout.execution.models import BrokerOrderResult


CRYPTO_STOP_LIMIT_OFFSET_BPS = 10.0


@dataclass(frozen=True)
class ProtectiveStopOrderSpec:
    order_type: str
    time_in_force: str
    stop_price: float
    limit_price: float | None = None


def submit_validated_order(
    *,
    request_id: str,
    alert_row: Mapping[str, Any],
) -> BrokerOrderResult:
    return submit_market_order(request_id=request_id, alert_row=alert_row)


def find_existing_broker_order(request_id: str) -> BrokerOrderResult | None:
    return get_order_by_client_order_id(request_id)


def fetch_broker_order_by_id(broker_order_id: str) -> BrokerOrderResult | None:
    return get_order_by_broker_order_id(broker_order_id)


def find_broker_order_by_client_order_id(client_order_id: str) -> BrokerOrderResult | None:
    return get_order_by_client_order_id(client_order_id)


def submit_stop_order(
    *,
    client_order_id: str,
    market_type: str | None,
    symbol: str,
    side: str,
    qty: float,
    stop_price: float,
) -> BrokerOrderResult:
    spec = build_protective_stop_order_spec(
        market_type=market_type,
        side=side,
        stop_price=stop_price,
    )
    if spec.order_type == "stop_limit":
        return submit_alpaca_stop_limit_order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            stop_price=spec.stop_price,
            limit_price=spec.limit_price if spec.limit_price is not None else spec.stop_price,
            time_in_force=spec.time_in_force,
        )
    return submit_alpaca_stop_order(
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        stop_price=spec.stop_price,
    )


def submit_limit_order(
    *,
    client_order_id: str,
    symbol: str,
    side: str,
    qty: float,
    limit_price: float,
) -> BrokerOrderResult:
    return submit_alpaca_limit_order(
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        limit_price=limit_price,
    )


def cancel_broker_order(broker_order_id: str) -> BrokerOrderResult:
    return cancel_alpaca_order_by_broker_order_id(broker_order_id)


def build_protective_stop_order_spec(
    *,
    market_type: str | None,
    side: str,
    stop_price: float,
) -> ProtectiveStopOrderSpec:
    normalized_market_type = str(market_type or "").strip().lower()
    if normalized_market_type == "crypto":
        return ProtectiveStopOrderSpec(
            order_type="stop_limit",
            time_in_force="gtc",
            stop_price=stop_price,
            limit_price=derive_crypto_stop_limit_price(stop_price=stop_price, side=side),
        )
    return ProtectiveStopOrderSpec(
        order_type="stop",
        time_in_force="gtc",
        stop_price=stop_price,
    )


def derive_crypto_stop_limit_price(*, stop_price: float, side: str) -> float:
    """Derive a conservative crypto stop-limit price from the protective exit side.

    SELL protective stops use a slightly lower limit than the stop price.
    BUY protective stops use a slightly higher limit than the stop price.
    """
    if stop_price <= 0:
        raise ValueError("stop_price must be greater than 0")

    normalized_side = str(side).strip().upper()
    offset_multiplier = CRYPTO_STOP_LIMIT_OFFSET_BPS / 10000.0
    if normalized_side == "SHORT":
        return round(stop_price * (1.0 - offset_multiplier), 6)
    if normalized_side == "LONG":
        return round(stop_price * (1.0 + offset_multiplier), 6)
    raise ValueError(f"Unsupported protective stop side: {side}")
