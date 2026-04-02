from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional

from trade_scout import config
from trade_scout.execution.broker import find_existing_broker_order, submit_validated_order
from trade_scout.execution.models import PaperPositionActionStatus, PaperOrderRole
from trade_scout.execution.policy import validate_execution_request
from trade_scout.execution.reconciler import ExecutionReconciler
from trade_scout.execution.repository import ExecutionRepository


class ExecutionService:
    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        reconciler: ExecutionReconciler,
        alert_lookup: Callable[[str], Optional[Mapping[str, Any]]],
        max_alert_age_seconds: int,
        broker_order_finder: Callable | None = None,
        broker_order_submitter: Callable | None = None,
    ):
        self.repository = repository
        self.reconciler = reconciler
        self.alert_lookup = alert_lookup
        self.max_alert_age_seconds = max_alert_age_seconds
        self.broker_order_finder = broker_order_finder
        self.broker_order_submitter = broker_order_submitter

    def process_claimed_request(self, conn, request_row) -> None:
        request_id = self._get_request_id(request_row)
        alert_id = self._get_alert_id(request_row)
        status = str(request_row["status"])

        if status != "CLAIMED":
            print(
                f"[execution-worker] warning request_id={request_id} "
                f"expected_status=CLAIMED actual_status={status}"
            )
            return

        try:
            alert_row = self.alert_lookup(alert_id)
            validation = validate_execution_request(
                request_row,
                alert_row,
                max_alert_age_seconds=self.max_alert_age_seconds,
            )

            if validation.is_expired:
                reason = validation.rejection_reason or "execution request expired"
                self.repository.mark_expired(conn, request_id, reason=reason)
                print(f"[execution-worker] expired request_id={request_id} reason={reason}")
                return

            if not validation.is_valid:
                code = validation.rejection_code or "VALIDATION_REJECTED"
                reason = validation.rejection_reason or "execution request rejected"
                self.repository.mark_rejected(conn, request_id, code=code, reason=reason)
                print(
                    f"[execution-worker] rejected request_id={request_id} "
                    f"code={code} reason={reason}"
                )
                return

            self.repository.mark_validated(conn, request_id)
            print(f"[execution-worker] validated request_id={request_id}")
            validated_row = self.repository.get_request_by_id(conn, request_id)
            if validated_row is None:
                self.repository.mark_failed(
                    conn,
                    request_id,
                    code="REQUEST_NOT_FOUND",
                    message="validated request could not be reloaded",
                )
                print(
                    f"[execution-worker] failed request_id={request_id} "
                    "error=validated request could not be reloaded"
                )
                return
            self.process_validated_request(conn, validated_row)
        except Exception as exc:
            self.repository.mark_failed(
                conn,
                request_id,
                code="PROCESSING_ERROR",
                message=str(exc),
            )
            print(f"[execution-worker] failed request_id={request_id} error={exc}")

    def process_validated_request(self, conn, request_row) -> None:
        request_id = self._get_request_id(request_row)
        alert_id = self._get_alert_id(request_row)
        status = str(request_row["status"])

        if status != "VALIDATED":
            print(
                f"[execution-worker] warning request_id={request_id} "
                f"expected_status=VALIDATED actual_status={status}"
            )
            return

        alert_row: Mapping[str, Any] | None = None
        trade_plan = None
        try:
            alert_row = self._require_alert_row(alert_id)
            trade_plan = self._get_or_create_trade_plan(
                conn,
                request_id=request_id,
                alert_id=alert_id,
                alert_row=alert_row,
            )
            self._ensure_position_for_plan(conn, trade_plan)

            existing_order = self.repository.get_paper_order_by_request_id_and_role(
                conn,
                request_id,
                PaperOrderRole.ENTRY.value,
            )
            if existing_order is not None:
                self._ensure_position_for_plan(conn, trade_plan, existing_order)
                self._mark_submitted_from_local_order(conn, request_id, existing_order)
                print(
                    f"[execution-worker] reused local paper order request_id={request_id} "
                    f"broker_order_id={existing_order['broker_order_id']}"
                )
                return

            recovered_remote_order = self._get_broker_order_finder()(request_id)
            if recovered_remote_order is not None:
                local_order = self._persist_or_reuse_broker_result(
                    conn,
                    trade_plan=trade_plan,
                    request_id=request_id,
                    alert_id=alert_id,
                    alert_row=alert_row,
                    result=recovered_remote_order,
                )
                self._ensure_position_for_plan(conn, trade_plan, local_order)
                self._mark_submitted_from_local_order(conn, request_id, local_order)
                print(
                    f"[execution-worker] recovered remote broker order request_id={request_id} "
                    f"broker_order_id={local_order['broker_order_id']}"
                )
                return

            print(
                f"[execution-worker] submitting order request_id={request_id} "
                f"alert_id={alert_id} notional={config.EXECUTION_ORDER_NOTIONAL}"
            )
            result = self._get_broker_order_submitter()(request_id=request_id, alert_row=alert_row)
            local_order = self._persist_or_reuse_broker_result(
                conn,
                trade_plan=trade_plan,
                request_id=request_id,
                alert_id=alert_id,
                alert_row=alert_row,
                result=result,
            )
            self._ensure_position_for_plan(conn, trade_plan, local_order)
            self._mark_submitted_from_local_order(conn, request_id, local_order)
            print(
                f"[execution-worker] submitted request_id={request_id} "
                f"broker_order_id={local_order['broker_order_id']}"
            )
        except Exception as exc:
            try:
                recovered_remote_order = self._get_broker_order_finder()(request_id)
            except Exception as recovery_exc:
                recovered_remote_order = None
                exc = RuntimeError(f"{exc}; recovery_lookup_error={recovery_exc}")

            if recovered_remote_order is not None:
                try:
                    if alert_row is None:
                        alert_row = self._require_alert_row(alert_id)
                    if trade_plan is None:
                        trade_plan = self._get_or_create_trade_plan(
                            conn,
                            request_id=request_id,
                            alert_id=alert_id,
                            alert_row=alert_row,
                        )
                    local_order = self._persist_or_reuse_broker_result(
                        conn,
                        trade_plan=trade_plan,
                        request_id=request_id,
                        alert_id=alert_id,
                        alert_row=alert_row,
                        result=recovered_remote_order,
                    )
                    self._ensure_position_for_plan(conn, trade_plan, local_order)
                    self._mark_submitted_from_local_order(conn, request_id, local_order)
                    print(
                        "[execution-worker] recovered broker order after submit exception "
                        f"request_id={request_id} broker_order_id={local_order['broker_order_id']}"
                    )
                    return
                except Exception as persist_exc:
                    exc = RuntimeError(f"{exc}; recovery_persist_error={persist_exc}")

            self.repository.mark_failed(
                conn,
                request_id,
                code="BROKER_SUBMIT_FAILED",
                message=str(exc),
            )
            print(f"[execution-worker] failed request_id={request_id} error={exc}")

    def process_position_actions(self, conn, *, limit: int = 25) -> int:
        actionable_positions = self.repository.get_positions_with_pending_actions(conn, limit=limit)
        processed_count = 0
        for position_row in actionable_positions:
            self.ensure_post_entry_position_state(conn, position_row)
            processed_count += 1
        return processed_count

    def ensure_post_entry_position_state(self, conn, position_row):
        pending_action_status = str(position_row["pending_action_status"])
        if pending_action_status == PaperPositionActionStatus.PLACE_INITIAL_STOP.value:
            return position_row
        return position_row

    def _require_alert_row(self, alert_id: str) -> Mapping[str, Any]:
        alert_row = self.alert_lookup(alert_id)
        if alert_row is None:
            raise RuntimeError("alert lookup returned no row during submission")
        return alert_row

    def _persist_or_reuse_broker_result(
        self,
        conn,
        *,
        trade_plan,
        request_id: str,
        alert_id: str,
        result,
        alert_row: Mapping[str, Any] | None = None,
    ):
        existing_order = self.repository.get_paper_order_by_request_id_and_role(
            conn,
            request_id,
            PaperOrderRole.ENTRY.value,
        )
        if existing_order is not None:
            return existing_order

        if alert_row is None:
            alert_row = self._require_alert_row(alert_id)

        try:
            return self.repository.insert_paper_order(
                conn,
                plan_id=str(trade_plan["plan_id"]),
                request_id=request_id,
                alert_id=alert_id,
                broker=result.broker,
                client_order_id=request_id,
                symbol=str(alert_row["symbol"]),
                side=str(alert_row["side"]),
                qty=None,
                notional=config.EXECUTION_ORDER_NOTIONAL,
                order_type="market",
                time_in_force="gtc",
                result=result,
            )
        except sqlite3.IntegrityError:
            existing_order = self.repository.get_paper_order_by_request_id_and_role(
                conn,
                request_id,
                PaperOrderRole.ENTRY.value,
            )
            if existing_order is None:
                raise
            print(
                f"[execution-worker] reused local duplicate paper order request_id={request_id} "
                f"broker_order_id={existing_order['broker_order_id']}"
            )
            return existing_order

    def _get_or_create_trade_plan(
        self,
        conn,
        *,
        request_id: str,
        alert_id: str,
        alert_row: Mapping[str, Any],
    ):
        existing_plan = self.repository.get_paper_trade_plan_by_request_id(conn, request_id)
        if existing_plan is not None:
            return existing_plan

        timeout_seconds = self._coerce_int(alert_row.get("timeout_seconds"))
        signal_ts_utc = self._coerce_optional_str(
            alert_row.get("signal_ts_utc") or alert_row.get("created_ts_utc")
        )
        timeout_ts_utc = None
        if signal_ts_utc and timeout_seconds is not None:
            try:
                timeout_dt = self._parse_iso_datetime(signal_ts_utc)
            except Exception:
                timeout_dt = None
            if timeout_dt is not None:
                timeout_ts_utc = (timeout_dt + timedelta(seconds=timeout_seconds)).isoformat()

        return self.repository.create_paper_trade_plan(
            conn,
            request_id=request_id,
            alert_id=alert_id,
            symbol=str(alert_row["symbol"]),
            side=str(alert_row["side"]),
            market_type=self._coerce_optional_str(alert_row.get("market_type")),
            strategy_profile=self._coerce_optional_str(alert_row.get("strategy_profile")),
            strategy_version=self._coerce_optional_str(alert_row.get("strategy_version")),
            config_hash=self._coerce_optional_str(alert_row.get("config_hash")),
            strategy_id=self._coerce_optional_str(alert_row.get("strategy_id")),
            setup_family=self._coerce_optional_str(alert_row.get("setup_family")),
            strategy_params_json=self._coerce_optional_str(alert_row.get("strategy_params_json")),
            strategy_params_hash=self._coerce_optional_str(alert_row.get("strategy_params_hash")),
            signal_ts_utc=signal_ts_utc,
            tier=self._coerce_optional_str(alert_row.get("tier")),
            entry_ref_px=self._coerce_float(alert_row.get("signal_px")),
            entry_low=self._coerce_float(alert_row.get("entry_low")),
            entry_high=self._coerce_float(alert_row.get("entry_high")),
            stop_px=self._coerce_float(alert_row.get("stop_px")),
            t1_px=self._coerce_float(alert_row.get("t1_px")),
            t2_px=self._coerce_float(alert_row.get("t2_px")),
            timeout_seconds=timeout_seconds,
            timeout_ts_utc=timeout_ts_utc,
            risk_model=self._coerce_optional_str(alert_row.get("risk_model")),
            risk_usd=self._coerce_float(alert_row.get("risk_usd")),
            qty_model=self._coerce_optional_str(alert_row.get("qty_model")),
            intended_qty=self._coerce_float(alert_row.get("qty")),
            intended_notional=config.EXECUTION_ORDER_NOTIONAL,
        )

    def _mark_submitted_from_local_order(self, conn, request_id: str, local_order) -> None:
        self.repository.mark_submitted(
            conn,
            request_id,
            broker=str(local_order["broker"]),
            broker_order_id=str(local_order["broker_order_id"]),
            submitted_ts_utc=str(local_order["submitted_ts_utc"]),
        )

    def _ensure_position_for_plan(self, conn, trade_plan, local_order=None):
        entry_order_id = None
        entry_client_order_id = None
        if local_order is not None:
            entry_order_id = self._coerce_optional_str(local_order["paper_order_id"])
            entry_client_order_id = self._coerce_optional_str(local_order["client_order_id"])
        return self.repository.ensure_paper_position(
            conn,
            plan_id=str(trade_plan["plan_id"]),
            request_id=str(trade_plan["request_id"]),
            alert_id=str(trade_plan["alert_id"]),
            symbol=str(trade_plan["symbol"]),
            side=str(trade_plan["side"]),
            entry_order_id=entry_order_id,
            entry_client_order_id=entry_client_order_id,
        )

    @staticmethod
    def _get_request_id(request_row) -> str:
        return str(request_row["request_id"])

    @staticmethod
    def _get_alert_id(request_row) -> str:
        return str(request_row["alert_id"])

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(float(value))

    @staticmethod
    def _parse_iso_datetime(value: str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)

    def _get_broker_order_finder(self) -> Callable:
        return self.broker_order_finder or find_existing_broker_order

    def _get_broker_order_submitter(self) -> Callable:
        return self.broker_order_submitter or submit_validated_order
