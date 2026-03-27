from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"
    SUBMITTED = "SUBMITTED"


class PaperTradePlanStatus(str, Enum):
    CREATED = "CREATED"


class PaperTradePlanEntryStatus(str, Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"


class PaperTradePlanPositionStatus(str, Enum):
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class PaperOrderRole(str, Enum):
    ENTRY = "ENTRY"
    STOP = "STOP"
    TP1 = "TP1"


class PaperFillRole(str, Enum):
    ENTRY = "ENTRY"
    STOP = "STOP"
    TP1 = "TP1"


class PaperOrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REPLACED = "REPLACED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PaperTradeOutcomeStatus(str, Enum):
    CLOSED = "CLOSED"


class PaperTradePlanPartialExitStatus(str, Enum):
    NONE = "NONE"
    TP1_PARTIAL = "TP1_PARTIAL"
    FULLY_EXITED = "FULLY_EXITED"


@dataclass(frozen=True)
class ExecutionRequest:
    request_id: str
    alert_id: str
    requested_ts_utc: str
    status: ExecutionStatus
    policy_name: Optional[str] = None
    policy_version: Optional[str] = None
    requested_by: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class BrokerOrderResult:
    broker: str
    broker_order_id: str
    submitted_ts_utc: str
    raw_response_json: str
    broker_status_raw: Optional[str] = None
    accepted_ts_utc: Optional[str] = None
    canceled_ts_utc: Optional[str] = None
    expired_ts_utc: Optional[str] = None
    rejected_ts_utc: Optional[str] = None
    filled_ts_utc: Optional[str] = None
    filled_qty: Optional[float] = None
    filled_avg_price: Optional[float] = None
    status_reason: Optional[str] = None
    last_broker_update_ts_utc: Optional[str] = None
