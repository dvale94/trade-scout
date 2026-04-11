-- Trade Scout execution health checks
--
-- Read-only operational checks for execution SQLite truth.
-- Intended for manual use with sqlite3:
--
--   sqlite3 analytics/execution.db < scripts/check_execution_health.sql
--
-- Notes:
-- - This file does not mutate data.
-- - The broken remaining_qty invariant check below derives STOP quantity from
--   paper_fills so it remains usable on older DB snapshots that do not have
--   paper_positions.stop_filled_qty.
-- - Dust checks use the default execution residual dust notional threshold of
--   1.0 unless you edit the params CTE below to match your environment.
-- - Dust notional is estimated as:
--   residual_qty * max(entry_avg_fill_px, stop_price, stop_limit_price).
-- - The Python wrapper prints a compact PASS/WARN/FAIL summary.

.headers on
.mode column

-- ---------------------------------------------------------------------------
-- A. Hard failures
-- ---------------------------------------------------------------------------

-- 1. Error positions
SELECT
    'error_positions' AS check_name,
    *
FROM paper_positions
WHERE position_status = 'ERROR';

-- 2a. Acceptable closed dust residuals.
-- Informational only: these are CLOSED positions with nonzero remaining_qty
-- whose estimated residual notional is at or below the dust threshold.
WITH params AS (
    SELECT 1.0 AS dust_notional_threshold
),
closed_residuals AS (
    SELECT
        p.*,
        MAX(
            COALESCE(p.entry_avg_fill_px, 0),
            COALESCE(p.stop_price, 0),
            COALESCE(p.stop_limit_price, 0)
        ) AS reference_price_estimate,
        ABS(COALESCE(p.remaining_qty, 0)) * MAX(
            COALESCE(p.entry_avg_fill_px, 0),
            COALESCE(p.stop_price, 0),
            COALESCE(p.stop_limit_price, 0)
        ) AS remaining_notional_estimate
    FROM paper_positions p
    WHERE p.position_status = 'CLOSED'
      AND ABS(COALESCE(p.remaining_qty, 0)) > 0.00000001
)
SELECT
    'acceptable_closed_dust_residuals' AS check_name,
    cr.*
FROM closed_residuals cr
CROSS JOIN params
WHERE cr.remaining_notional_estimate <= params.dust_notional_threshold;

-- 2b. Unexpected closed residuals above dust threshold.
WITH params AS (
    SELECT 1.0 AS dust_notional_threshold
),
closed_residuals AS (
    SELECT
        p.*,
        MAX(
            COALESCE(p.entry_avg_fill_px, 0),
            COALESCE(p.stop_price, 0),
            COALESCE(p.stop_limit_price, 0)
        ) AS reference_price_estimate,
        ABS(COALESCE(p.remaining_qty, 0)) * MAX(
            COALESCE(p.entry_avg_fill_px, 0),
            COALESCE(p.stop_price, 0),
            COALESCE(p.stop_limit_price, 0)
        ) AS remaining_notional_estimate
    FROM paper_positions p
    WHERE p.position_status = 'CLOSED'
      AND ABS(COALESCE(p.remaining_qty, 0)) > 0.00000001
)
SELECT
    'unexpected_closed_positions_with_leftover_qty' AS check_name,
    cr.*
FROM closed_residuals cr
CROSS JOIN params
WHERE cr.remaining_notional_estimate > params.dust_notional_threshold;

-- 3a. Acceptable dust remaining_qty invariant mismatches.
-- Uses STOP quantity derived from paper_fills so this remains notebook-safe and
-- compatible with DB snapshots where paper_positions.stop_filled_qty is absent.
-- Informational only when the expected residual notional is under threshold.
WITH params AS (
    SELECT 1.0 AS dust_notional_threshold
),
fill_stop AS (
    SELECT
        plan_id,
        SUM(CASE WHEN fill_role = 'STOP' THEN COALESCE(fill_qty, 0) ELSE 0 END) AS stop_fill_qty
    FROM paper_fills
    GROUP BY plan_id
),
qty_check AS (
    SELECT
        p.position_id,
        p.plan_id,
        p.request_id,
        p.alert_id,
        p.symbol,
        p.position_status,
        p.entry_avg_fill_px,
        p.stop_price,
        p.stop_limit_price,
        p.entry_filled_qty,
        p.tp1_filled_qty,
        p.tp2_filled_qty,
        COALESCE(fs.stop_fill_qty, 0) AS stop_filled_qty_from_fills,
        p.remaining_qty,
        MAX(
            COALESCE(p.entry_filled_qty, 0)
            - COALESCE(p.tp1_filled_qty, 0)
            - COALESCE(p.tp2_filled_qty, 0)
            - COALESCE(fs.stop_fill_qty, 0),
            0
        ) AS expected_remaining_qty_from_fills,
        MAX(
            COALESCE(p.entry_avg_fill_px, 0),
            COALESCE(p.stop_price, 0),
            COALESCE(p.stop_limit_price, 0)
        ) AS reference_price_estimate
    FROM paper_positions p
    LEFT JOIN fill_stop fs
        ON fs.plan_id = p.plan_id
),
qty_check_with_notional AS (
    SELECT
        *,
        ABS(expected_remaining_qty_from_fills) * reference_price_estimate AS expected_remaining_notional_estimate
    FROM qty_check
)
SELECT
    'acceptable_dust_invariant_mismatches' AS check_name,
    q.*
FROM qty_check_with_notional q
CROSS JOIN params
WHERE ABS(COALESCE(remaining_qty, 0) - expected_remaining_qty_from_fills) > 0.00000001
  AND q.expected_remaining_notional_estimate <= params.dust_notional_threshold;

-- 3b. Unexpected remaining_qty invariant failures above dust threshold.
WITH fill_stop AS (
    SELECT
        plan_id,
        SUM(CASE WHEN fill_role = 'STOP' THEN COALESCE(fill_qty, 0) ELSE 0 END) AS stop_fill_qty
    FROM paper_fills
    GROUP BY plan_id
),
qty_check AS (
    SELECT
        p.position_id,
        p.plan_id,
        p.request_id,
        p.alert_id,
        p.symbol,
        p.position_status,
        p.entry_avg_fill_px,
        p.stop_price,
        p.stop_limit_price,
        p.entry_filled_qty,
        p.tp1_filled_qty,
        p.tp2_filled_qty,
        COALESCE(fs.stop_fill_qty, 0) AS stop_filled_qty_from_fills,
        p.remaining_qty,
        MAX(
            COALESCE(p.entry_filled_qty, 0)
            - COALESCE(p.tp1_filled_qty, 0)
            - COALESCE(p.tp2_filled_qty, 0)
            - COALESCE(fs.stop_fill_qty, 0),
            0
        ) AS expected_remaining_qty_from_fills
        ,
        MAX(
            COALESCE(p.entry_avg_fill_px, 0),
            COALESCE(p.stop_price, 0),
            COALESCE(p.stop_limit_price, 0)
        ) AS reference_price_estimate
    FROM paper_positions p
    LEFT JOIN fill_stop fs
        ON fs.plan_id = p.plan_id
),
params AS (
    SELECT 1.0 AS dust_notional_threshold
),
qty_check_with_notional AS (
    SELECT
        *,
        ABS(expected_remaining_qty_from_fills) * reference_price_estimate AS expected_remaining_notional_estimate
    FROM qty_check
)
SELECT
    'unexpected_remaining_qty_invariant_failures' AS check_name,
    q.*
FROM qty_check_with_notional q
CROSS JOIN params
WHERE ABS(COALESCE(remaining_qty, 0) - expected_remaining_qty_from_fills) > 0.00000001
  AND q.expected_remaining_notional_estimate > params.dust_notional_threshold;

-- 4. Plans without positions
SELECT
    'plans_without_positions' AS check_name,
    p.*
FROM paper_trade_plans p
LEFT JOIN paper_positions pos
    ON pos.plan_id = p.plan_id
WHERE pos.plan_id IS NULL;

-- 5. Active stop pointer references no order
SELECT
    'active_stop_pointer_missing_order' AS check_name,
    pos.*
FROM paper_positions pos
LEFT JOIN paper_orders o
    ON o.paper_order_id = pos.active_stop_paper_order_id
WHERE pos.active_stop_paper_order_id IS NOT NULL
  AND o.paper_order_id IS NULL;

-- ---------------------------------------------------------------------------
-- B. Stuck workflow checks
-- ---------------------------------------------------------------------------

-- 6. Suspicious open / partial / closing positions missing active stop.
-- Missing an active stop can be normal during managed-exit transitions such as
-- canceling a stop before TP submission or rearming after TP1. This check warns
-- only when there is remaining live qty, no active stop, no active TP2, and the
-- pending action does not clearly explain the temporary protection gap.
SELECT
    'suspicious_open_positions_missing_active_stop' AS check_name,
    *
FROM paper_positions
WHERE position_status IN ('OPEN', 'PARTIAL', 'CLOSING')
  AND COALESCE(remaining_qty, 0) > 0.00000001
  AND active_stop_paper_order_id IS NULL
  AND active_tp2_paper_order_id IS NULL
  AND (
      COALESCE(pending_action_status, 'NONE') IN ('NONE', 'WAIT_TP2_FILL')
      OR COALESCE(pending_action_status, 'NONE') NOT IN (
          'PLACE_INITIAL_STOP',
          'CANCEL_STOP_FOR_TP1',
          'SUBMIT_TP1',
          'WAIT_TP1_FILL',
          'REARM_STOP_AFTER_TP1',
          'CANCEL_STOP_FOR_TP2',
          'SUBMIT_TP2'
      )
  );

-- 7. Stale positions with non-NONE pending_action_status.
-- There is no dedicated action-updated timestamp today. updated_ts_utc is the
-- best read-only proxy for how long the pending action has been present.
SELECT
    'stale_pending_action_positions' AS check_name,
    position_id,
    plan_id,
    request_id,
    alert_id,
    symbol,
    position_status,
    pending_action_status,
    active_stop_paper_order_id,
    active_tp1_paper_order_id,
    active_tp2_paper_order_id,
    remaining_qty,
    updated_ts_utc,
    CAST((julianday('now') - julianday(updated_ts_utc)) * 24 * 60 AS INTEGER) AS action_age_minutes,
    last_action_error
FROM paper_positions
WHERE COALESCE(pending_action_status, 'NONE') != 'NONE'
  AND updated_ts_utc IS NOT NULL
  AND (julianday('now') - julianday(updated_ts_utc)) * 24 * 60 > 15;

-- 8. Old requests still in nonterminal states.
-- Age is computed relative to SQLite's current UTC timestamp.
SELECT
    'old_nonterminal_requests' AS check_name,
    request_id,
    alert_id,
    status,
    requested_ts_utc,
    claimed_ts_utc,
    validated_ts_utc,
    submitted_ts_utc,
    CAST((julianday('now') - julianday(requested_ts_utc)) * 24 * 60 AS INTEGER) AS age_minutes,
    error_code,
    error_message
FROM execution_requests
WHERE status IN ('PENDING', 'CLAIMED', 'VALIDATED')
  AND requested_ts_utc IS NOT NULL
  AND (julianday('now') - julianday(requested_ts_utc)) * 24 * 60 > 30
ORDER BY requested_ts_utc;

-- ---------------------------------------------------------------------------
-- C. Lifecycle / join sanity
-- ---------------------------------------------------------------------------

-- 9a. Unexpected requests without plans.
-- Terminal pre-plan states can legitimately have no plan. Warn only on
-- SUBMITTED requests missing a plan or aged nonterminal requests missing a plan.
SELECT
    'unexpected_requests_without_plans' AS check_name,
    r.*,
    CAST((julianday('now') - julianday(r.requested_ts_utc)) * 24 * 60 AS INTEGER) AS age_minutes
FROM execution_requests r
LEFT JOIN paper_trade_plans p
    ON p.request_id = r.request_id
WHERE p.request_id IS NULL
  AND (
      r.status = 'SUBMITTED'
      OR (
          r.status IN ('PENDING', 'CLAIMED', 'VALIDATED')
          AND r.requested_ts_utc IS NOT NULL
          AND (julianday('now') - julianday(r.requested_ts_utc)) * 24 * 60 > 30
      )
  );

-- 9b. Expected terminal requests without plans.
-- Informational only: these are terminal request states that may end before plan
-- creation and should not automatically count as daily WARNs.
SELECT
    'expected_terminal_requests_without_plans' AS check_name,
    r.*
FROM execution_requests r
LEFT JOIN paper_trade_plans p
    ON p.request_id = r.request_id
WHERE p.request_id IS NULL
  AND r.status IN ('REJECTED', 'EXPIRED', 'FAILED');

-- 10. Position status summary
SELECT
    position_status,
    pending_action_status,
    COUNT(*) AS count
FROM paper_positions
GROUP BY position_status, pending_action_status
ORDER BY count DESC, position_status, pending_action_status;

-- 11. Request status summary
SELECT
    status,
    error_code,
    broker_reject_category,
    COUNT(*) AS count
FROM execution_requests
GROUP BY status, error_code, broker_reject_category
ORDER BY count DESC, status, error_code, broker_reject_category;

-- 12. Count summary
SELECT 'execution_requests' AS table_name, COUNT(*) AS row_count FROM execution_requests
UNION ALL
SELECT 'paper_trade_plans' AS table_name, COUNT(*) AS row_count FROM paper_trade_plans
UNION ALL
SELECT 'paper_positions' AS table_name, COUNT(*) AS row_count FROM paper_positions
UNION ALL
SELECT 'paper_orders' AS table_name, COUNT(*) AS row_count FROM paper_orders
UNION ALL
SELECT 'paper_fills' AS table_name, COUNT(*) AS row_count FROM paper_fills;
