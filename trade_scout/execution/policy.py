from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    is_expired: bool = False
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None


def parse_iso_utc(value: Any) -> datetime:
    if value is None:
        raise ValueError("missing timestamp")

    text = str(value).strip()
    if not text:
        raise ValueError("blank timestamp")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_alert_field(alert_row: Mapping[str, Any], candidates: Iterable[str]) -> Any:
    for name in candidates:
        value = alert_row.get(name)
        if value is not None:
            return value
    return None


def validate_execution_request(
    request_row,
    alert_row,
    *,
    max_alert_age_seconds: int,
    now_utc: Optional[datetime] = None,
) -> ValidationResult:
    del request_row

    if alert_row is None:
        return ValidationResult(
            is_valid=False,
            rejection_code="ALERT_NOT_FOUND",
            rejection_reason="alert_id not found in simulation truth",
        )

    symbol = get_alert_field(alert_row, ["symbol"])
    if symbol is None or not str(symbol).strip():
        return ValidationResult(
            is_valid=False,
            rejection_code="INVALID_SYMBOL",
            rejection_reason="alert symbol is missing or blank",
        )

    side = get_alert_field(alert_row, ["side"])
    if side is None or not str(side).strip():
        return ValidationResult(
            is_valid=False,
            rejection_code="INVALID_SIDE",
            rejection_reason="alert side is missing or blank",
        )

    created_ts_raw = get_alert_field(
        alert_row,
        ["created_ts_utc", "alert_created_ts_utc", "created_at_utc", "created_at"],
    )
    try:
        created_ts_utc = parse_iso_utc(created_ts_raw)
    except Exception:
        return ValidationResult(
            is_valid=False,
            rejection_code="INVALID_TIMESTAMP",
            rejection_reason="unable to parse alert timestamp",
        )

    now = now_utc.astimezone(timezone.utc) if now_utc is not None else datetime.now(timezone.utc)
    age_seconds = max(0.0, (now - created_ts_utc).total_seconds())
    if age_seconds > float(max_alert_age_seconds):
        return ValidationResult(
            is_valid=False,
            is_expired=True,
            rejection_code="ALERT_STALE",
            rejection_reason=(
                f"alert is stale: age_seconds={age_seconds:.0f} "
                f"window_seconds={max_alert_age_seconds}"
            ),
        )

    return ValidationResult(is_valid=True)
