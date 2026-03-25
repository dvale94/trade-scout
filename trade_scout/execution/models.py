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


class PaperOrderRole(str, Enum):
    ENTRY = "ENTRY"


class PaperOrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"


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
