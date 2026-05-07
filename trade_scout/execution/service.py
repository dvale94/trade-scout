from __future__ import annotations

import os
import re
from decimal import Decimal, ROUND_DOWN
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional

from trade_scout import config
from trade_scout.execution.broker import (
    cancel_broker_order,
    find_existing_broker_order,
    find_existing_broker_order_by_client_order_id,
    submit_take_profit_order,
    submit_protective_stop_order,
    submit_validated_order,
)
from trade_scout.execution.models import (
    PaperPositionActionStatus,
    PaperPositionStatus,
    PaperOrderRole,
)
from trade_scout.execution.policy import validate_execution_request
from trade_scout.execution.reconciler import ExecutionReconciler
from trade_scout.execution.repository import ExecutionRepository


class ExecutionService:
    CRYPTO_EXIT_QTY_DECIMALS = 8
    DEFAULT_RESIDUAL_DUST_NOTIONAL = 1.0
    _BROKER_REJECT_CODE_RE = re.compile(r'"code"\s*:\s*([0-9]+)|code[=\s:]+([0-9]+)', re.IGNORECASE)
    _INSUFFICIENT_BALANCE_RE = re.compile(
        r"requested:\s*([0-9.]+)\s*,\s*available:\s*([0-9.]+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        reconciler: ExecutionReconciler,
        alert_lookup: Callable[[str], Optional[Mapping[str, Any]]],
        max_alert_age_seconds: int,
        broker_order_finder: Callable | None = None,
        broker_order_submitter: Callable | None = None,
        stop_order_finder: Callable | None = None,
        stop_order_submitter: Callable | None = None,
        broker_order_canceler: Callable | None = None,
        tp1_order_finder: Callable | None = None,
        tp1_order_submitter: Callable | None = None,
    ):
        self.repository = repository
        self.reconciler = reconciler
        self.alert_lookup = alert_lookup
        self.max_alert_age_seconds = max_alert_age_seconds
        self.broker_order_finder = broker_order_finder
        self.broker_order_submitter = broker_order_submitter
        self.stop_order_finder = stop_order_finder
        self.stop_order_submitter = stop_order_submitter
        self.broker_order_canceler = broker_order_canceler
        self.tp1_order_finder = tp1_order_finder
        self.tp1_order_submitter = tp1_order_submitter

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

            symbol = str(trade_plan["symbol"])
            conflicts = self.repository.get_active_same_symbol_exposure(
                conn,
                symbol=symbol,
                exclude_plan_id=str(trade_plan["plan_id"]),
                exclude_request_id=request_id,
            )
            if conflicts:
                conflicts_text = self._format_same_symbol_exposure_conflicts(conflicts)
                reason = (
                    f"same symbol active exposure symbol={symbol} "
                    f"conflicts={conflicts_text}"
                )
                self.repository.mark_rejected(
                    conn,
                    request_id,
                    code="SAME_SYMBOL_ACTIVE_EXPOSURE",
                    reason=reason,
                )
                print(
                    f"[execution-worker] skipped request_id={request_id} "
                    f"symbol={symbol} reason=SAME_SYMBOL_ACTIVE_EXPOSURE "
                    f"conflicts={conflicts_text}"
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
                broker_reject_category=self._classify_broker_submit_failure(exc),
                broker_reject_code=self._extract_broker_reject_code(exc),
            )
            print(f"[execution-worker] failed request_id={request_id} error={exc}")

    def process_position_actions(self, conn, *, limit: int = 25) -> int:
        actionable_positions = self.repository.get_positions_with_pending_actions(conn, limit=limit)
        processed_count = 0
        for position_row in actionable_positions:
            try:
                self.ensure_post_entry_position_state(conn, position_row)
                processed_count += 1
            except Exception as exc:
                self.repository.update_paper_position_action_error(
                    conn,
                    plan_id=str(position_row["plan_id"]),
                    last_action_error=str(exc),
                )
                print(
                    f"[execution-worker] position action failed plan_id={position_row['plan_id']} "
                    f"pending_action_status={position_row['pending_action_status']} error={exc}"
                )
        residual_positions = self.repository.get_positions_eligible_for_trigger_evaluation(conn, limit=limit)
        for position_row in residual_positions:
            try:
                updated_position = self._process_residual_position_if_needed(conn, position_row)
                if updated_position is not None:
                    processed_count += 1
            except Exception as exc:
                self.repository.update_paper_position_action_error(
                    conn,
                    plan_id=str(position_row["plan_id"]),
                    last_action_error=str(exc),
                )
                print(
                    f"[execution-worker] residual handling failed plan_id={position_row['plan_id']} "
                    f"pending_action_status={position_row['pending_action_status']} error={exc}"
                )
        return processed_count

    def process_position_triggers(
        self,
        conn,
        *,
        price_by_symbol: Mapping[str, float],
        limit: int = 100,
        now_utc: datetime | None = None,
    ) -> int:
        if not price_by_symbol:
            return 0

        eligible_positions = self.repository.get_positions_eligible_for_trigger_evaluation(
            conn,
            limit=limit,
        )
        processed_count = 0
        for position_row in eligible_positions:
            if self.evaluate_take_profit_triggers(
                conn,
                position_row=position_row,
                price_by_symbol=price_by_symbol,
                now_utc=now_utc,
            ):
                processed_count += 1
        return processed_count

    def ensure_post_entry_position_state(self, conn, position_row):
        pending_action_status = str(position_row["pending_action_status"])
        if pending_action_status == PaperPositionActionStatus.PLACE_INITIAL_STOP.value:
            return self._place_initial_stop_if_needed(conn, position_row)
        if pending_action_status == PaperPositionActionStatus.CANCEL_STOP_FOR_TP1.value:
            return self._cancel_stop_for_tp1_if_needed(conn, position_row)
        if pending_action_status == PaperPositionActionStatus.SUBMIT_TP1.value:
            return self._submit_tp1_if_needed(conn, position_row)
        if pending_action_status == PaperPositionActionStatus.REARM_STOP_AFTER_TP1.value:
            return self._rearm_stop_after_tp1_if_needed(conn, position_row)
        if pending_action_status == PaperPositionActionStatus.CANCEL_STOP_FOR_TP2.value:
            return self._cancel_stop_for_tp2_if_needed(conn, position_row)
        if pending_action_status == PaperPositionActionStatus.SUBMIT_TP2.value:
            return self._submit_tp2_if_needed(conn, position_row)
        if pending_action_status == PaperPositionActionStatus.WAIT_TP2_FILL.value:
            return self._wait_tp2_fill_if_needed(conn, position_row)
        return position_row

    def evaluate_take_profit_triggers(
        self,
        conn,
        *,
        position_row,
        price_by_symbol: Mapping[str, float],
        now_utc: datetime | None = None,
    ) -> bool:
        symbol = str(position_row["symbol"])
        current_price = price_by_symbol.get(symbol)
        if current_price is None:
            return False

        if str(position_row["pending_action_status"]) != PaperPositionActionStatus.NONE.value:
            return False
        if str(position_row["position_status"]) not in {
            PaperPositionStatus.OPEN.value,
            PaperPositionStatus.PARTIAL.value,
        }:
            return False

        remaining_qty = self._coerce_float(position_row["remaining_qty"])
        if remaining_qty is None or remaining_qty <= 0:
            return False

        trade_plan = self.repository.get_paper_trade_plan_by_plan_id(conn, str(position_row["plan_id"]))
        if trade_plan is None:
            return False

        side = str(position_row["side"])
        tp1_px = self._coerce_float(trade_plan["t1_px"])
        tp2_px = self._coerce_float(trade_plan["t2_px"])
        trigger_ts_utc = self._now_iso_utc(now_utc)

        tp1_triggered_ts = self._coerce_optional_str(position_row["tp1_triggered_ts_utc"])
        tp2_triggered_ts = self._coerce_optional_str(position_row["tp2_triggered_ts_utc"])

        if tp1_triggered_ts is None and tp1_px is not None and self._is_tp_crossed(
            side=side,
            current_price=float(current_price),
            target_price=tp1_px,
        ):
            self.repository.record_paper_position_trigger(
                conn,
                plan_id=str(position_row["plan_id"]),
                trigger_name="tp1",
                triggered_ts_utc=trigger_ts_utc,
            )
            self.repository.update_paper_position_pending_action(
                conn,
                plan_id=str(position_row["plan_id"]),
                pending_action_status=PaperPositionActionStatus.CANCEL_STOP_FOR_TP1.value,
            )
            print(
                f"[execution-worker] tp1 triggered plan_id={position_row['plan_id']} "
                f"symbol={symbol} current_price={current_price}"
            )
            return True

        if (
            tp1_triggered_ts is not None
            and tp2_triggered_ts is None
            and tp2_px is not None
            and self._is_tp_crossed(
                side=side,
                current_price=float(current_price),
                target_price=tp2_px,
            )
        ):
            self.repository.record_paper_position_trigger(
                conn,
                plan_id=str(position_row["plan_id"]),
                trigger_name="tp2",
                triggered_ts_utc=trigger_ts_utc,
            )
            self.repository.update_paper_position_pending_action(
                conn,
                plan_id=str(position_row["plan_id"]),
                pending_action_status=PaperPositionActionStatus.CANCEL_STOP_FOR_TP2.value,
            )
            print(
                f"[execution-worker] tp2 triggered plan_id={position_row['plan_id']} "
                f"symbol={symbol} current_price={current_price}"
            )
            return True

        return False

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
    def _format_same_symbol_exposure_conflicts(conflicts: list[Any]) -> str:
        parts: list[str] = []
        for row in conflicts:
            parts.append(
                f"{row['plan_id']}:{row['request_id']}:{row['position_status']}"
            )
        return ",".join(parts)

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

    def _get_stop_order_finder(self) -> Callable:
        return self.stop_order_finder or find_existing_broker_order_by_client_order_id

    def _get_stop_order_submitter(self) -> Callable:
        return self.stop_order_submitter or submit_protective_stop_order

    def _get_broker_order_canceler(self) -> Callable:
        return self.broker_order_canceler or cancel_broker_order

    def _get_tp1_order_finder(self) -> Callable:
        return self.tp1_order_finder or find_existing_broker_order_by_client_order_id

    def _get_tp1_order_submitter(self) -> Callable:
        return self.tp1_order_submitter or submit_take_profit_order

    @staticmethod
    def _is_tp_crossed(*, side: str, current_price: float, target_price: float) -> bool:
        normalized_side = str(side).strip().upper()
        if normalized_side == "LONG":
            return current_price >= target_price
        if normalized_side == "SHORT":
            return current_price <= target_price
        raise RuntimeError(f"unsupported side for TP trigger evaluation: {side}")

    @staticmethod
    def _now_iso_utc(now_utc: datetime | None = None) -> str:
        if now_utc is None:
            return datetime.now(timezone.utc).isoformat()
        normalized = now_utc.astimezone(timezone.utc) if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
        return normalized.isoformat()

    def _place_initial_stop_if_needed(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        if str(position_row["position_status"]) not in {
            PaperPositionStatus.OPEN.value,
            PaperPositionStatus.PARTIAL.value,
        }:
            raise RuntimeError("initial stop action requires OPEN or PARTIAL position")

        remaining_qty = self._coerce_float(position_row["remaining_qty"])
        if remaining_qty is None or remaining_qty <= 0:
            raise RuntimeError("initial stop action requires remaining_qty > 0")

        trade_plan = self.repository.get_paper_trade_plan_by_plan_id(conn, plan_id)
        if trade_plan is None:
            raise RuntimeError("paper trade plan not found for position action")

        stop_price = self._coerce_float(trade_plan["stop_px"])
        if stop_price is None or stop_price <= 0:
            raise RuntimeError("paper trade plan stop_px is missing or invalid")
        stop_limit_price = stop_price
        stop_submission_qty = self._normalize_exit_qty(
            symbol=str(position_row["symbol"]),
            qty=remaining_qty,
        )
        if stop_submission_qty <= 0:
            raise RuntimeError("initial stop action requires normalized stop qty > 0")
        stop_client_order_id = self._build_initial_stop_client_order_id(plan_id)
        parent_paper_order_id = str(position_row["entry_order_id"] or "") or None

        local_stop_order = self.repository.get_paper_order_by_client_order_id(conn, stop_client_order_id)
        if local_stop_order is not None:
            return self._finalize_stop_state(
                conn,
                position_row=position_row,
                stop_order_row=local_stop_order,
                stop_qty=self._coerce_float(local_stop_order["qty"]) or stop_submission_qty,
                stop_price=stop_price,
                stop_limit_price=stop_limit_price,
                log_label="initial stop",
            )

        remote_stop_order = self._get_stop_order_finder()(stop_client_order_id)
        if remote_stop_order is not None:
            local_stop_order = self.repository.insert_paper_order(
                conn,
                plan_id=plan_id,
                request_id=str(position_row["request_id"]),
                alert_id=str(position_row["alert_id"]),
                broker=remote_stop_order.broker,
                client_order_id=stop_client_order_id,
                symbol=str(position_row["symbol"]),
                side=str(position_row["side"]),
                qty=stop_submission_qty,
                notional=None,
                order_type="stop_limit",
                time_in_force="gtc",
                order_role=PaperOrderRole.STOP.value,
                parent_paper_order_id=parent_paper_order_id,
                limit_price=stop_limit_price,
                stop_price=stop_price,
                result=remote_stop_order,
            )
            return self._finalize_stop_state(
                conn,
                position_row=position_row,
                stop_order_row=local_stop_order,
                stop_qty=stop_submission_qty,
                stop_price=stop_price,
                stop_limit_price=stop_limit_price,
                log_label="initial stop",
            )

        stop_result, stop_submission_qty = self._submit_stop_order_with_balance_handling(
            conn,
            position_row=position_row,
            client_order_id=stop_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=stop_submission_qty,
            stop_price=stop_price,
            limit_price=stop_limit_price,
            log_label="initial stop",
        )
        if stop_result is None:
            return self.repository.get_paper_position_by_plan_id(conn, plan_id)
        local_stop_order = self.repository.insert_paper_order(
            conn,
            plan_id=plan_id,
            request_id=str(position_row["request_id"]),
            alert_id=str(position_row["alert_id"]),
            broker=stop_result.broker,
            client_order_id=stop_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=stop_submission_qty,
            notional=None,
            order_type="stop_limit",
            time_in_force="gtc",
            order_role=PaperOrderRole.STOP.value,
            parent_paper_order_id=parent_paper_order_id,
            limit_price=stop_limit_price,
            stop_price=stop_price,
            result=stop_result,
        )
        print(
            f"[execution-worker] placed initial stop plan_id={plan_id} "
            f"paper_order_id={local_stop_order['paper_order_id']} broker_order_id={local_stop_order['broker_order_id']}"
        )
        return self._finalize_stop_state(
            conn,
            position_row=position_row,
            stop_order_row=local_stop_order,
            stop_qty=stop_submission_qty,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
            log_label="initial stop",
        )

    def _submit_stop_order_with_balance_handling(
        self,
        conn,
        *,
        position_row,
        client_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float | None,
        log_label: str,
    ) -> tuple[object | None, float]:
        requested_qty = qty
        try:
            return (
                self._get_stop_order_submitter()(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    qty=requested_qty,
                    stop_price=stop_price,
                    limit_price=limit_price,
                ),
                requested_qty,
            )
        except Exception as exc:
            available_qty = self._extract_available_qty_from_insufficient_balance_error(exc)
            if available_qty is not None:
                retry_qty = self._normalize_exit_qty(symbol=symbol, qty=available_qty)
                if 0 < retry_qty < requested_qty:
                    try:
                        return (
                            self._get_stop_order_submitter()(
                                client_order_id=client_order_id,
                                symbol=symbol,
                                side=side,
                                qty=retry_qty,
                                stop_price=stop_price,
                                limit_price=limit_price,
                            ),
                            retry_qty,
                        )
                    except Exception as retry_exc:
                        error_message = self._format_stop_rejection_error(
                            log_label=log_label,
                            requested_qty=requested_qty,
                            available_qty=available_qty,
                            error=retry_exc,
                        )
                        self.repository.mark_paper_position_error(
                            conn,
                            plan_id=str(position_row["plan_id"]),
                            last_action_error=error_message,
                        )
                        print(
                            f"[execution-worker] {log_label} rejected plan_id={position_row['plan_id']} "
                            f"requested_qty={requested_qty} available_qty={available_qty} error={retry_exc}"
                        )
                        return None, retry_qty

            if self._is_insufficient_balance_error(exc):
                error_message = self._format_stop_rejection_error(
                    log_label=log_label,
                    requested_qty=requested_qty,
                    available_qty=available_qty,
                    error=exc,
                )
                self.repository.mark_paper_position_error(
                    conn,
                    plan_id=str(position_row["plan_id"]),
                    last_action_error=error_message,
                )
                print(
                    f"[execution-worker] {log_label} rejected plan_id={position_row['plan_id']} "
                    f"requested_qty={requested_qty} available_qty={available_qty} error={exc}"
                )
                return None, requested_qty

            raise

    def _finalize_stop_state(
        self,
        conn,
        *,
        position_row,
        stop_order_row,
        stop_qty: float,
        stop_price: float,
        stop_limit_price: float | None,
        log_label: str,
    ):
        updated_position = self.repository.update_paper_position_stop_state(
            conn,
            plan_id=str(position_row["plan_id"]),
            active_stop_paper_order_id=str(stop_order_row["paper_order_id"]),
            stop_qty=stop_qty,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
        )
        if updated_position is None:
            raise RuntimeError("paper position stop state update did not persist")
        updated_position = self.repository.update_paper_position_pending_action(
            conn,
            plan_id=str(position_row["plan_id"]),
            pending_action_status=PaperPositionActionStatus.NONE.value,
        )
        updated_position = self.repository.update_paper_position_action_error(
            conn,
            plan_id=str(position_row["plan_id"]),
            last_action_error=None,
        )
        print(
            f"[execution-worker] linked {log_label} plan_id={position_row['plan_id']} "
            f"paper_order_id={stop_order_row['paper_order_id']}"
        )
        return updated_position

    def _rearm_stop_after_tp1_if_needed(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        if str(position_row["position_status"]) != PaperPositionStatus.PARTIAL.value:
            raise RuntimeError("stop rearm action requires PARTIAL position")

        remaining_qty = self._coerce_float(position_row["remaining_qty"])
        if remaining_qty is None or remaining_qty <= 0:
            raise RuntimeError("stop rearm action requires remaining_qty > 0")

        trade_plan = self.repository.get_paper_trade_plan_by_plan_id(conn, plan_id)
        if trade_plan is None:
            raise RuntimeError("paper trade plan not found for stop rearm")

        stop_price = self._coerce_float(trade_plan["stop_px"])
        if stop_price is None or stop_price <= 0:
            raise RuntimeError("paper trade plan stop_px is missing or invalid")
        stop_limit_price = stop_price
        stop_submission_qty = self._normalize_exit_qty(
            symbol=str(position_row["symbol"]),
            qty=remaining_qty,
        )
        if stop_submission_qty <= 0:
            raise RuntimeError("stop rearm action requires normalized stop qty > 0")
        stop_client_order_id = self._build_rearm_stop_client_order_id(plan_id)
        parent_paper_order_id = self._coerce_optional_str(position_row["entry_order_id"])

        local_stop_order = self.repository.get_paper_order_by_client_order_id(conn, stop_client_order_id)
        if local_stop_order is not None:
            return self._finalize_stop_state(
                conn,
                position_row=position_row,
                stop_order_row=local_stop_order,
                stop_qty=self._coerce_float(local_stop_order["qty"]) or stop_submission_qty,
                stop_price=stop_price,
                stop_limit_price=stop_limit_price,
                log_label="rearmed stop",
            )

        remote_stop_order = self._get_stop_order_finder()(stop_client_order_id)
        if remote_stop_order is not None:
            local_stop_order = self.repository.insert_paper_order(
                conn,
                plan_id=plan_id,
                request_id=str(position_row["request_id"]),
                alert_id=str(position_row["alert_id"]),
                broker=remote_stop_order.broker,
                client_order_id=stop_client_order_id,
                symbol=str(position_row["symbol"]),
                side=str(position_row["side"]),
                qty=stop_submission_qty,
                notional=None,
                order_type="stop_limit",
                time_in_force="gtc",
                order_role=PaperOrderRole.STOP.value,
                parent_paper_order_id=parent_paper_order_id,
                limit_price=stop_limit_price,
                stop_price=stop_price,
                result=remote_stop_order,
            )
            return self._finalize_stop_state(
                conn,
                position_row=position_row,
                stop_order_row=local_stop_order,
                stop_qty=stop_submission_qty,
                stop_price=stop_price,
                stop_limit_price=stop_limit_price,
                log_label="rearmed stop",
            )

        stop_result, stop_submission_qty = self._submit_stop_order_with_balance_handling(
            conn,
            position_row=position_row,
            client_order_id=stop_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=stop_submission_qty,
            stop_price=stop_price,
            limit_price=stop_limit_price,
            log_label="rearmed stop",
        )
        if stop_result is None:
            return self.repository.get_paper_position_by_plan_id(conn, plan_id)
        local_stop_order = self.repository.insert_paper_order(
            conn,
            plan_id=plan_id,
            request_id=str(position_row["request_id"]),
            alert_id=str(position_row["alert_id"]),
            broker=stop_result.broker,
            client_order_id=stop_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=stop_submission_qty,
            notional=None,
            order_type="stop_limit",
            time_in_force="gtc",
            order_role=PaperOrderRole.STOP.value,
            parent_paper_order_id=parent_paper_order_id,
            limit_price=stop_limit_price,
            stop_price=stop_price,
            result=stop_result,
        )
        print(
            f"[execution-worker] rearmed stop plan_id={plan_id} "
            f"paper_order_id={local_stop_order['paper_order_id']} broker_order_id={local_stop_order['broker_order_id']}"
        )
        return self._finalize_stop_state(
            conn,
            position_row=position_row,
            stop_order_row=local_stop_order,
            stop_qty=stop_submission_qty,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
            log_label="rearmed stop",
        )

    def _cancel_stop_for_tp1_if_needed(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        active_stop_paper_order_id = self._coerce_optional_str(position_row["active_stop_paper_order_id"])

        if not active_stop_paper_order_id:
            return self._advance_to_submit_tp1(conn, position_row)

        stop_order_row = self.repository.get_paper_order_by_id(conn, active_stop_paper_order_id)
        if stop_order_row is None:
            return self._advance_to_submit_tp1(conn, position_row)

        stop_status = str(stop_order_row["order_status"])
        if stop_status in {"CANCELED", "REJECTED", "EXPIRED"}:
            return self._advance_to_submit_tp1(conn, position_row)
        if stop_status == "FILLED":
            raise RuntimeError("cannot advance TP1 submission after stop already filled")

        broker_order_id = self._coerce_optional_str(stop_order_row["broker_order_id"])
        if not broker_order_id:
            return self._advance_to_submit_tp1(conn, position_row)

        cancel_result = self._get_broker_order_canceler()(broker_order_id)
        if cancel_result is not None:
            self.reconciler.reconcile_paper_order(conn, stop_order_row, cancel_result)
            stop_order_row = self.repository.get_paper_order_by_id(conn, active_stop_paper_order_id)
        if stop_order_row is None:
            return self._advance_to_submit_tp1(conn, position_row)

        if str(stop_order_row["order_status"]) in {"CANCELED", "REJECTED", "EXPIRED"}:
            return self._advance_to_submit_tp1(conn, position_row)

        raise RuntimeError(
            f"stop cancel did not reach terminal non-filled state order_status={stop_order_row['order_status']}"
        )

    def _advance_to_submit_tp1(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        updated_position = self.repository.update_paper_position_stop_state(
            conn,
            plan_id=plan_id,
            active_stop_paper_order_id=None,
            stop_qty=self._coerce_float(position_row["stop_qty"]),
            stop_price=self._coerce_float(position_row["stop_price"]),
            stop_limit_price=self._coerce_float(position_row["stop_limit_price"]),
        )
        updated_position = self.repository.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=PaperPositionActionStatus.SUBMIT_TP1.value,
        )
        print(f"[execution-worker] advanced tp1 action plan_id={plan_id} next_action=SUBMIT_TP1")
        return self._submit_tp1_if_needed(conn, updated_position or position_row)

    def _submit_tp1_if_needed(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        trade_plan = self.repository.get_paper_trade_plan_by_plan_id(conn, plan_id)
        if trade_plan is None:
            raise RuntimeError("paper trade plan not found for tp1 submission")

        tp1_qty = self._coerce_float(position_row["tp1_target_qty"])
        if tp1_qty is None or tp1_qty <= 0:
            raise RuntimeError("tp1 submission requires tp1_target_qty > 0")

        tp1_price = self._coerce_float(trade_plan["t1_px"])
        if tp1_price is None or tp1_price <= 0:
            raise RuntimeError("paper trade plan t1_px is missing or invalid")
        tp1_client_order_id = self._build_tp1_client_order_id(plan_id)

        local_tp1_order = self.repository.get_paper_order_by_client_order_id(conn, tp1_client_order_id)
        if local_tp1_order is None:
            local_tp1_order = self.repository.get_paper_order_by_plan_id_and_role(
                conn,
                plan_id,
                PaperOrderRole.TP1.value,
            )

        if local_tp1_order is not None:
            return self._finalize_tp1_submission_state(
                conn,
                position_row=position_row,
                tp1_order_row=local_tp1_order,
            )

        remote_tp1_order = self._get_tp1_order_finder()(tp1_client_order_id)
        if remote_tp1_order is not None:
            local_tp1_order = self.repository.insert_paper_order(
                conn,
                plan_id=plan_id,
                request_id=str(position_row["request_id"]),
                alert_id=str(position_row["alert_id"]),
                broker=remote_tp1_order.broker,
                client_order_id=tp1_client_order_id,
                symbol=str(position_row["symbol"]),
                side=str(position_row["side"]),
                qty=tp1_qty,
                notional=None,
                order_type="limit",
                time_in_force="gtc",
                order_role=PaperOrderRole.TP1.value,
                parent_paper_order_id=str(position_row["entry_order_id"] or "") or None,
                limit_price=tp1_price,
                stop_price=None,
                result=remote_tp1_order,
            )
            return self._finalize_tp1_submission_state(
                conn,
                position_row=position_row,
                tp1_order_row=local_tp1_order,
            )

        tp1_result = self._get_tp1_order_submitter()(
            client_order_id=tp1_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=tp1_qty,
            limit_price=tp1_price,
        )
        local_tp1_order = self.repository.insert_paper_order(
            conn,
            plan_id=plan_id,
            request_id=str(position_row["request_id"]),
            alert_id=str(position_row["alert_id"]),
            broker=tp1_result.broker,
            client_order_id=tp1_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=tp1_qty,
            notional=None,
            order_type="limit",
            time_in_force="gtc",
            order_role=PaperOrderRole.TP1.value,
            parent_paper_order_id=str(position_row["entry_order_id"] or "") or None,
            limit_price=tp1_price,
            stop_price=None,
            result=tp1_result,
        )
        print(
            f"[execution-worker] submitted tp1 plan_id={plan_id} "
            f"paper_order_id={local_tp1_order['paper_order_id']} broker_order_id={local_tp1_order['broker_order_id']}"
        )
        return self._finalize_tp1_submission_state(
            conn,
            position_row=position_row,
            tp1_order_row=local_tp1_order,
        )

    def _finalize_tp1_submission_state(self, conn, *, position_row, tp1_order_row):
        plan_id = str(position_row["plan_id"])
        updated_position = self.repository.update_paper_position_tp1_state(
            conn,
            plan_id=plan_id,
            active_tp1_paper_order_id=str(tp1_order_row["paper_order_id"]),
        )
        updated_position = self.repository.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=PaperPositionActionStatus.WAIT_TP1_FILL.value,
        )
        updated_position = self.repository.update_paper_position_action_error(
            conn,
            plan_id=plan_id,
            last_action_error=None,
        )
        print(
            f"[execution-worker] waiting tp1 fill plan_id={plan_id} "
            f"paper_order_id={tp1_order_row['paper_order_id']}"
        )
        return updated_position

    def _cancel_stop_for_tp2_if_needed(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        active_stop_paper_order_id = self._coerce_optional_str(position_row["active_stop_paper_order_id"])

        if not active_stop_paper_order_id:
            return self._advance_to_submit_tp2(conn, position_row)

        stop_order_row = self.repository.get_paper_order_by_id(conn, active_stop_paper_order_id)
        if stop_order_row is None:
            return self._advance_to_submit_tp2(conn, position_row)

        stop_status = str(stop_order_row["order_status"])
        if stop_status in {"CANCELED", "REJECTED", "EXPIRED"}:
            return self._advance_to_submit_tp2(conn, position_row)
        if stop_status == "FILLED":
            return self._resolve_tp2_action_after_stop_filled(conn, position_row)

        broker_order_id = self._coerce_optional_str(stop_order_row["broker_order_id"])
        if not broker_order_id:
            return self._advance_to_submit_tp2(conn, position_row)

        cancel_result = self._get_broker_order_canceler()(broker_order_id)
        if cancel_result is not None:
            self.reconciler.reconcile_paper_order(conn, stop_order_row, cancel_result)
            stop_order_row = self.repository.get_paper_order_by_id(conn, active_stop_paper_order_id)
        if stop_order_row is None:
            return self._advance_to_submit_tp2(conn, position_row)

        if str(stop_order_row["order_status"]) in {"CANCELED", "REJECTED", "EXPIRED"}:
            return self._advance_to_submit_tp2(conn, position_row)

        raise RuntimeError(
            f"stop cancel did not reach terminal non-filled state order_status={stop_order_row['order_status']}"
        )

    def _resolve_tp2_action_after_stop_filled(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        updated_position = self.repository.update_paper_position_stop_state(
            conn,
            plan_id=plan_id,
            active_stop_paper_order_id=None,
            stop_qty=self._coerce_float(position_row["stop_qty"]),
            stop_price=self._coerce_float(position_row["stop_price"]),
            stop_limit_price=self._coerce_float(position_row["stop_limit_price"]),
        )
        updated_position = self.repository.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=PaperPositionActionStatus.NONE.value,
        )
        updated_position = self.repository.update_paper_position_action_error(
            conn,
            plan_id=plan_id,
            last_action_error=None,
        )
        print(
            f"[execution-worker] resolved tp2 action plan_id={plan_id} reason=stop_already_filled"
        )
        return updated_position

    def _advance_to_submit_tp2(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        updated_position = self.repository.update_paper_position_stop_state(
            conn,
            plan_id=plan_id,
            active_stop_paper_order_id=None,
            stop_qty=self._coerce_float(position_row["stop_qty"]),
            stop_price=self._coerce_float(position_row["stop_price"]),
            stop_limit_price=self._coerce_float(position_row["stop_limit_price"]),
        )
        updated_position = self.repository.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=PaperPositionActionStatus.SUBMIT_TP2.value,
        )
        print(f"[execution-worker] advanced tp2 action plan_id={plan_id} next_action=SUBMIT_TP2")
        return self._submit_tp2_if_needed(conn, updated_position or position_row)

    def _submit_tp2_if_needed(self, conn, position_row):
        plan_id = str(position_row["plan_id"])
        trade_plan = self.repository.get_paper_trade_plan_by_plan_id(conn, plan_id)
        if trade_plan is None:
            raise RuntimeError("paper trade plan not found for tp2 submission")

        tp2_qty = self._coerce_float(position_row["remaining_qty"])
        if tp2_qty is None or tp2_qty <= 0:
            raise RuntimeError("tp2 submission requires remaining_qty > 0")
        tp2_submission_qty = self._normalize_exit_qty(
            symbol=str(position_row["symbol"]),
            qty=tp2_qty,
        )
        if tp2_submission_qty <= 0:
            raise RuntimeError("tp2 submission requires normalized remaining_qty > 0")

        tp2_price = self._coerce_float(trade_plan["t2_px"])
        if tp2_price is None or tp2_price <= 0:
            raise RuntimeError("paper trade plan t2_px is missing or invalid")
        tp2_client_order_id = self._build_tp2_client_order_id(plan_id)

        local_tp2_order = self.repository.get_paper_order_by_client_order_id(conn, tp2_client_order_id)
        if local_tp2_order is not None:
            return self._finalize_tp2_submission_state(
                conn,
                position_row=position_row,
                tp2_order_row=local_tp2_order,
            )

        remote_tp2_order = self._get_tp1_order_finder()(tp2_client_order_id)
        if remote_tp2_order is not None:
            local_tp2_order = self.repository.insert_paper_order(
                conn,
                plan_id=plan_id,
                request_id=str(position_row["request_id"]),
                alert_id=str(position_row["alert_id"]),
                broker=remote_tp2_order.broker,
                client_order_id=tp2_client_order_id,
                symbol=str(position_row["symbol"]),
                side=str(position_row["side"]),
                qty=tp2_submission_qty,
                notional=None,
                order_type="limit",
                time_in_force="gtc",
                order_role=PaperOrderRole.TP2.value,
                parent_paper_order_id=str(position_row["entry_order_id"] or "") or None,
                limit_price=tp2_price,
                stop_price=None,
                result=remote_tp2_order,
            )
            return self._finalize_tp2_submission_state(
                conn,
                position_row=position_row,
                tp2_order_row=local_tp2_order,
            )

        tp2_result, tp2_submission_qty = self._submit_tp2_order_with_balance_handling(
            conn,
            position_row=position_row,
            client_order_id=tp2_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=tp2_submission_qty,
            limit_price=tp2_price,
        )
        if tp2_result is None:
            return self.repository.get_paper_position_by_plan_id(conn, plan_id)
        local_tp2_order = self.repository.insert_paper_order(
            conn,
            plan_id=plan_id,
            request_id=str(position_row["request_id"]),
            alert_id=str(position_row["alert_id"]),
            broker=tp2_result.broker,
            client_order_id=tp2_client_order_id,
            symbol=str(position_row["symbol"]),
            side=str(position_row["side"]),
            qty=tp2_submission_qty,
            notional=None,
            order_type="limit",
            time_in_force="gtc",
            order_role=PaperOrderRole.TP2.value,
            parent_paper_order_id=str(position_row["entry_order_id"] or "") or None,
            limit_price=tp2_price,
            stop_price=None,
            result=tp2_result,
        )
        print(
            f"[execution-worker] submitted tp2 plan_id={plan_id} "
            f"paper_order_id={local_tp2_order['paper_order_id']} broker_order_id={local_tp2_order['broker_order_id']}"
        )
        return self._finalize_tp2_submission_state(
            conn,
            position_row=position_row,
            tp2_order_row=local_tp2_order,
        )

    def _submit_tp2_order_with_balance_handling(
        self,
        conn,
        *,
        position_row,
        client_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
    ) -> tuple[object | None, float]:
        requested_qty = qty
        try:
            return (
                self._get_tp1_order_submitter()(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    qty=requested_qty,
                    limit_price=limit_price,
                ),
                requested_qty,
            )
        except Exception as exc:
            available_qty = self._extract_available_qty_from_insufficient_balance_error(exc)
            if available_qty is not None:
                retry_qty = self._normalize_exit_qty(symbol=symbol, qty=available_qty)
                if 0 < retry_qty < requested_qty:
                    try:
                        return (
                            self._get_tp1_order_submitter()(
                                client_order_id=client_order_id,
                                symbol=symbol,
                                side=side,
                                qty=retry_qty,
                                limit_price=limit_price,
                            ),
                            retry_qty,
                        )
                    except Exception as retry_exc:
                        error_message = self._format_tp2_rejection_error(
                            requested_qty=requested_qty,
                            available_qty=available_qty,
                            error=retry_exc,
                        )
                        self.repository.mark_paper_position_error(
                            conn,
                            plan_id=str(position_row["plan_id"]),
                            last_action_error=error_message,
                        )
                        print(
                            f"[execution-worker] tp2 rejected plan_id={position_row['plan_id']} "
                            f"requested_qty={requested_qty} available_qty={available_qty} error={retry_exc}"
                        )
                        return None, retry_qty

            if self._is_insufficient_balance_error(exc):
                error_message = self._format_tp2_rejection_error(
                    requested_qty=requested_qty,
                    available_qty=available_qty,
                    error=exc,
                )
                self.repository.mark_paper_position_error(
                    conn,
                    plan_id=str(position_row["plan_id"]),
                    last_action_error=error_message,
                )
                print(
                    f"[execution-worker] tp2 rejected plan_id={position_row['plan_id']} "
                    f"requested_qty={requested_qty} available_qty={available_qty} error={exc}"
                )
                return None, requested_qty

            raise

    def _finalize_tp2_submission_state(self, conn, *, position_row, tp2_order_row):
        plan_id = str(position_row["plan_id"])
        updated_position = self.repository.update_paper_position_tp2_state(
            conn,
            plan_id=plan_id,
            active_tp2_paper_order_id=str(tp2_order_row["paper_order_id"]),
        )
        updated_position = self.repository.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=PaperPositionActionStatus.WAIT_TP2_FILL.value,
        )
        updated_position = self.repository.update_paper_position_action_error(
            conn,
            plan_id=plan_id,
            last_action_error=None,
        )
        print(
            f"[execution-worker] waiting tp2 fill plan_id={plan_id} "
            f"paper_order_id={tp2_order_row['paper_order_id']}"
        )
        return updated_position

    def _wait_tp2_fill_if_needed(self, conn, position_row):
        active_tp2_paper_order_id = self._coerce_optional_str(position_row["active_tp2_paper_order_id"])
        if active_tp2_paper_order_id:
            return position_row
        updated_position = self._process_residual_position_if_needed(
            conn,
            position_row,
            source="stale_wait_tp2_fill",
        )
        return updated_position or position_row

    def _process_residual_position_if_needed(self, conn, position_row, *, source: str = "residual_scan"):
        if str(position_row["position_status"]) not in {
            PaperPositionStatus.OPEN.value,
            PaperPositionStatus.PARTIAL.value,
        }:
            return None

        pending_action_status = str(position_row["pending_action_status"])
        if pending_action_status not in {
            PaperPositionActionStatus.NONE.value,
            PaperPositionActionStatus.WAIT_TP2_FILL.value,
        }:
            return None

        active_stop_paper_order_id = self._coerce_optional_str(position_row["active_stop_paper_order_id"])
        active_tp2_paper_order_id = self._coerce_optional_str(position_row["active_tp2_paper_order_id"])
        if active_stop_paper_order_id or active_tp2_paper_order_id:
            return None

        remaining_qty = self._coerce_float(position_row["remaining_qty"])
        if remaining_qty is None or remaining_qty <= 0:
            if pending_action_status == PaperPositionActionStatus.WAIT_TP2_FILL.value:
                plan_id = str(position_row["plan_id"])
                updated_position = self.repository.update_paper_position_tp2_state(
                    conn,
                    plan_id=plan_id,
                    active_tp2_paper_order_id=None,
                )
                updated_position = self.repository.update_paper_position_pending_action(
                    conn,
                    plan_id=plan_id,
                    pending_action_status=PaperPositionActionStatus.NONE.value,
                )
                updated_position = self.repository.update_paper_position_action_error(
                    conn,
                    plan_id=plan_id,
                    last_action_error=None,
                )
                print(
                    f"[execution-worker] resolved residual plan_id={plan_id} "
                    f"action=clear_stale_wait reason=no_remaining_qty source={source}"
                )
                return updated_position
            return None

        plan_id = str(position_row["plan_id"])
        if self.repository.has_filled_exit_order(conn, plan_id):
            if pending_action_status == PaperPositionActionStatus.NONE.value:
                return None
            print(
                f"[execution-worker] continuing residual classification plan_id={plan_id} "
                f"reason=terminal_exit_recorded source={source}"
            )

        residual_notional = self._estimate_residual_notional(position_row)
        dust_threshold = self._residual_dust_notional_threshold()
        if residual_notional is not None and residual_notional <= dust_threshold:
            updated_position = self.repository.close_paper_position_as_dust(conn, plan_id=plan_id)
            print(
                f"[execution-worker] closed residual dust plan_id={plan_id} "
                f"remaining_qty={remaining_qty} residual_notional={residual_notional}"
            )
            return updated_position

        next_action = self._residual_reprotection_action(position_row)
        updated_position = self.repository.update_paper_position_tp2_state(
            conn,
            plan_id=plan_id,
            active_tp2_paper_order_id=None,
        )
        updated_position = self.repository.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=next_action,
        )
        updated_position = self.repository.update_paper_position_action_error(
            conn,
            plan_id=plan_id,
            last_action_error=None,
        )
        print(
            f"[execution-worker] scheduled residual protection plan_id={plan_id} "
            f"next_action={next_action} source={source}"
        )
        return updated_position

    def _estimate_residual_notional(self, position_row) -> float | None:
        remaining_qty = self._coerce_float(position_row["remaining_qty"])
        if remaining_qty is None or remaining_qty <= 0:
            return None

        fallback_prices = [
            self._coerce_float(position_row["entry_avg_fill_px"]),
            self._coerce_float(position_row["stop_price"]),
            self._coerce_float(position_row["stop_limit_price"]),
        ]
        positive_prices = [price for price in fallback_prices if price is not None and price > 0]
        if not positive_prices:
            return None
        return remaining_qty * max(positive_prices)

    @classmethod
    def _residual_dust_notional_threshold(cls) -> float:
        raw = os.getenv("EXECUTION_RESIDUAL_DUST_NOTIONAL", str(cls.DEFAULT_RESIDUAL_DUST_NOTIONAL)).strip()
        try:
            return max(float(raw), 0.0)
        except ValueError:
            return cls.DEFAULT_RESIDUAL_DUST_NOTIONAL

    @staticmethod
    def _residual_reprotection_action(position_row) -> str:
        if str(position_row["position_status"]) == PaperPositionStatus.OPEN.value:
            return PaperPositionActionStatus.PLACE_INITIAL_STOP.value
        return PaperPositionActionStatus.REARM_STOP_AFTER_TP1.value

    @staticmethod
    def _build_initial_stop_client_order_id(plan_id: str) -> str:
        return f"stop:{plan_id}"

    @staticmethod
    def _build_rearm_stop_client_order_id(plan_id: str) -> str:
        return f"stop:rearm:tp1:{plan_id}"

    @classmethod
    def _normalize_exit_qty(cls, *, symbol: str, qty: float) -> float:
        if config.MARKET_TYPE != "crypto":
            return qty
        quant = Decimal("1").scaleb(-cls.CRYPTO_EXIT_QTY_DECIMALS)
        normalized = Decimal(str(qty)).quantize(quant, rounding=ROUND_DOWN)
        return float(normalized)

    @classmethod
    def _is_insufficient_balance_error(cls, error: Exception) -> bool:
        return "insufficient balance" in str(error).lower()

    @classmethod
    def _extract_available_qty_from_insufficient_balance_error(cls, error: Exception) -> float | None:
        if not cls._is_insufficient_balance_error(error):
            return None
        match = cls._INSUFFICIENT_BALANCE_RE.search(str(error))
        if match is None:
            return None
        return float(match.group(2))

    @staticmethod
    def _format_stop_rejection_error(
        *,
        log_label: str,
        requested_qty: float,
        available_qty: float | None,
        error: Exception,
    ) -> str:
        if available_qty is None:
            return (
                f"{log_label} rejected: insufficient balance "
                f"requested_qty={requested_qty} error={error}"
            )
        return (
            f"{log_label} rejected: insufficient balance "
            f"requested_qty={requested_qty} available_qty={available_qty} error={error}"
        )

    @staticmethod
    def _format_tp2_rejection_error(
        *,
        requested_qty: float,
        available_qty: float | None,
        error: Exception,
    ) -> str:
        if available_qty is None:
            return (
                "tp2 rejected: insufficient balance "
                f"requested_qty={requested_qty} error={error}"
            )
        return (
            "tp2 rejected: insufficient balance "
            f"requested_qty={requested_qty} available_qty={available_qty} error={error}"
        )

    @staticmethod
    def _classify_broker_submit_failure(error: Exception) -> str:
        error_text = str(error).lower()
        if "potential wash trade detected" in error_text or "opposite side limit order exists" in error_text:
            return "WASH_TRADE_CONFLICT"
        if "insufficient balance" in error_text or "insufficient buying power" in error_text:
            return "INSUFFICIENT_BALANCE"
        if "read timed out" in error_text or "connection timed out" in error_text or "connect timeout" in error_text:
            return "NETWORK_TIMEOUT"
        if "alpaca" in error_text or "broker" in error_text or "status=" in error_text:
            return "BROKER_REJECT_OTHER"
        return "UNKNOWN"

    @classmethod
    def _extract_broker_reject_code(cls, error: Exception) -> str | None:
        match = cls._BROKER_REJECT_CODE_RE.search(str(error))
        if match is None:
            return None
        return match.group(1) or match.group(2)

    @staticmethod
    def _build_tp1_client_order_id(plan_id: str) -> str:
        return f"tp1:{plan_id}"

    @staticmethod
    def _build_tp2_client_order_id(plan_id: str) -> str:
        return f"tp2:{plan_id}"
