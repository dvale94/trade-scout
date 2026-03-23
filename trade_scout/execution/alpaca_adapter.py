from __future__ import annotations

import json
from typing import Any, Mapping

import requests

from trade_scout import config
from trade_scout.execution.models import BrokerOrderResult
from trade_scout.execution.queue import utc_now_iso


ALPACA_BROKER_NAME = "alpaca_paper"


def _map_side(side: str) -> str:
    normalized = str(side).strip().upper()
    if normalized == "LONG":
        return "buy"
    if normalized == "SHORT":
        return "sell"
    raise ValueError(f"Unsupported execution side for Alpaca: {side}")


def _require_alpaca_config() -> None:
    if not config.ALPACA_API_KEY.strip():
        raise RuntimeError("Missing ALPACA_API_KEY for paper execution submission")
    if not config.ALPACA_SECRET_KEY.strip():
        raise RuntimeError("Missing ALPACA_SECRET_KEY for paper execution submission")
    if not config.ALPACA_BASE_URL.strip():
        raise RuntimeError("Missing ALPACA_BASE_URL for paper execution submission")
    if config.EXECUTION_ORDER_QTY <= 0:
        raise RuntimeError("EXECUTION_ORDER_QTY must be greater than 0")


def _auth_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


def _parse_response_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw_text": response.text}


def _normalize_order_result(response_json: Mapping[str, Any]) -> BrokerOrderResult:
    broker_order_id = str(response_json.get("id", "")).strip()
    submitted_ts_utc = str(response_json.get("submitted_at", "")).strip() or utc_now_iso()
    if not broker_order_id:
        raise RuntimeError("Alpaca order payload did not include broker order id")

    return BrokerOrderResult(
        broker=ALPACA_BROKER_NAME,
        broker_order_id=broker_order_id,
        submitted_ts_utc=submitted_ts_utc,
        raw_response_json=json.dumps(response_json, sort_keys=True),
    )


def submit_market_order(
    *,
    request_id: str,
    alert_row: Mapping[str, Any],
) -> BrokerOrderResult:
    _require_alpaca_config()

    symbol = str(alert_row.get("symbol", "")).strip()
    side = _map_side(str(alert_row.get("side", "")))
    if not symbol:
        raise RuntimeError("Alert row is missing symbol for Alpaca submission")

    payload = {
        "symbol": symbol,
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
        "qty": config.EXECUTION_ORDER_QTY,
        "client_order_id": request_id,
    }

    url = f"{config.ALPACA_BASE_URL.rstrip('/')}/v2/orders"
    response = requests.post(url, json=payload, headers=_auth_headers(), timeout=15)
    response_json = _parse_response_json(response)

    if response.status_code >= 400:
        detail = response_json if isinstance(response_json, dict) else {"response": response_json}
        raise RuntimeError(
            f"Alpaca submission failed status={response.status_code} detail={json.dumps(detail, sort_keys=True)}"
        )

    return _normalize_order_result(response_json)


def get_order_by_client_order_id(request_id: str) -> BrokerOrderResult | None:
    _require_alpaca_config()

    url = f"{config.ALPACA_BASE_URL.rstrip('/')}/v2/orders:by_client_order_id"
    response = requests.get(
        url,
        params={"client_order_id": request_id},
        headers=_auth_headers(),
        timeout=15,
    )
    response_json = _parse_response_json(response)

    if response.status_code == 404:
        return None

    if response.status_code >= 400:
        detail = response_json if isinstance(response_json, dict) else {"response": response_json}
        raise RuntimeError(
            f"Alpaca order lookup failed status={response.status_code} detail={json.dumps(detail, sort_keys=True)}"
        )

    return _normalize_order_result(response_json)
