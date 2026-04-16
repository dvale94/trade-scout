from __future__ import annotations

import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.Response = object
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    requests_stub.delete = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

watch_stub = types.ModuleType("trade_scout.watch")


class _FakeWatch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


watch_stub.Watch = _FakeWatch
sys.modules["trade_scout.watch"] = watch_stub

strategies_pkg_stub = types.ModuleType("trade_scout.strategies")
strategies_types_stub = types.ModuleType("trade_scout.strategies.types")


class _FakeStrategyMetadata:
    def __init__(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        setup_family: str,
        strategy_params_json: str,
        strategy_params_hash: str,
    ):
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version
        self.setup_family = setup_family
        self.strategy_params_json = strategy_params_json
        self.strategy_params_hash = strategy_params_hash


strategies_types_stub.StrategyMetadata = _FakeStrategyMetadata
strategies_types_stub.SetupCandidate = object
sys.modules["trade_scout.strategies"] = strategies_pkg_stub
sys.modules["trade_scout.strategies.types"] = strategies_types_stub

runtime_context_stub = types.ModuleType("trade_scout.runtime_context")
runtime_context_stub.EnabledStrategy = object
runtime_context_stub.RuntimeContext = object
sys.modules["trade_scout.runtime_context"] = runtime_context_stub

from trade_scout.alert_pipeline import AlertPipeline
from trade_scout.execution.queue import get_connection, init_db, utc_now_iso
from trade_scout.policy.models import PolicyActivationRow, PolicyDefinitionRow
from trade_scout.policy.repository import PolicyRepository, activate_policy, insert_policy_definition
from trade_scout.scoring.models import RuntimeShadowScoreOutput


class _FakeCooldown:
    def allowed(self, symbol, tier, now, strategy_id):
        return True

    def mark(self, symbol, tier, now, strategy_id):
        return None


class _FakeTelemetry:
    def record_alert_sent(self, tier):
        return None

    def record_cooldown_blocked(self, tier):
        return None

    def record_skipped_tight_stop(self):
        return None


class _FakeWatchManager:
    def __init__(self):
        self.added = []

    def add(self, watch):
        self.added.append(watch)


class AlertPipelinePolicyIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "execution.db")
        init_db(self.db_path)
        self.policy_repository = PolicyRepository(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_alert_triggers_policy_evaluation_and_persists_decision(self) -> None:
        self._activate_shadow_policy(filters={"allowed_symbols": ["AAPL"]})
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        conn = get_connection(self.db_path)
        try:
            decisions = conn.execute("SELECT * FROM policy_decisions").fetchall()
        finally:
            conn.close()

        self.assertEqual(1, len(decisions))
        self.assertEqual("alert-1", decisions[0]["alert_id"])
        self.assertEqual("ALLOW", decisions[0]["decision"])
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_duplicate_alert_does_not_duplicate_policy_decision(self) -> None:
        self._activate_shadow_policy(filters={"allowed_symbols": ["AAPL"]})
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1", "provisional-2", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())
            pipeline.process(**self._process_kwargs())

        conn = get_connection(self.db_path)
        try:
            row_count = conn.execute("SELECT COUNT(*) AS n FROM policy_decisions").fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(1, row_count)

    def test_no_active_policy_inserts_no_decision(self) -> None:
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        conn = get_connection(self.db_path)
        try:
            row_count = conn.execute("SELECT COUNT(*) AS n FROM policy_decisions").fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(0, row_count)
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_multiple_shadow_policy_variants_persist_one_decision_each(self) -> None:
        with patch(
            "trade_scout.alert_pipeline.config.SHADOW_POLICY_VARIANT_NAMES",
            [
                "shadow_score_gte_8_0",
                "shadow_score_gte_7_5",
                "shadow_block_score_lt_6_5",
                "shadow_score_gte_8_0_bear",
            ],
        ):
            pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs(anchor_regime="bear"))

        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT policy_name, decision
                FROM policy_decisions
                WHERE alert_id = ?
                ORDER BY policy_name
                """,
                ("alert-1",),
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(
            {
                "shadow_block_score_lt_6_5": "ALLOW",
                "shadow_score_gte_7_5": "ALLOW",
                "shadow_score_gte_8_0": "ALLOW",
                "shadow_score_gte_8_0_bear": "ALLOW",
            },
            {row["policy_name"]: row["decision"] for row in rows},
        )
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_shadow_policy_block_does_not_block_live_alert_flow(self) -> None:
        with patch(
            "trade_scout.alert_pipeline.config.SHADOW_POLICY_VARIANT_NAMES",
            ["shadow_score_gte_8_0", "shadow_block_score_lt_6_5"],
        ):
            pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs(score=6.2, anchor_regime="bear"))

        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT policy_name, decision FROM policy_decisions WHERE alert_id = ? ORDER BY policy_name",
                ("alert-1",),
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(
            {
                "shadow_block_score_lt_6_5": "BLOCK",
                "shadow_score_gte_8_0": "BLOCK",
            },
            {row["policy_name"]: row["decision"] for row in rows},
        )
        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))
        self.assertEqual("ALERT_CREATED", side_effects["log_intent"].call_args.args[0].alert_status)

    def test_one_shadow_policy_persist_failure_does_not_block_other_variants(self) -> None:
        with patch(
            "trade_scout.alert_pipeline.config.SHADOW_POLICY_VARIANT_NAMES",
            ["shadow_score_gte_8_0", "shadow_score_gte_7_5", "shadow_block_score_lt_6_5"],
        ):
            pipeline, context = self._make_pipeline()

        original_insert = self.policy_repository.insert_policy_decision

        def insert_side_effect(conn, row):
            if row.policy_name == "shadow_score_gte_7_5":
                raise sqlite3.OperationalError("variant persist boom")
            return original_insert(conn, row)

        with self._patch_alert_side_effects() as side_effects, patch.object(
            self.policy_repository,
            "insert_policy_decision",
            side_effect=insert_side_effect,
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs(anchor_regime="bear"))

        conn = get_connection(self.db_path)
        try:
            policy_names = [
                row["policy_name"]
                for row in conn.execute(
                    "SELECT policy_name FROM policy_decisions WHERE alert_id = ? ORDER BY policy_name",
                    ("alert-1",),
                ).fetchall()
            ]
        finally:
            conn.close()

        self.assertEqual(["shadow_block_score_lt_6_5", "shadow_score_gte_8_0"], policy_names)
        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))
        self.assertTrue(
            any(
                "[policy] error alert_id=alert-1 stage=persist policy_name=shadow_score_gte_7_5"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_shadow_policy_variants_differentiate_low_score_alert(self) -> None:
        with patch(
            "trade_scout.alert_pipeline.config.SHADOW_POLICY_VARIANT_NAMES",
            [
                "shadow_score_gte_8_0",
                "shadow_score_gte_7_5",
                "shadow_block_score_lt_6_5",
                "shadow_score_gte_8_0_bear",
            ],
        ):
            pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs(score=6.2, anchor_regime="bear"))

        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT policy_name, decision
                FROM policy_decisions
                WHERE alert_id = ?
                ORDER BY policy_name
                """,
                ("alert-1",),
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(
            {
                "shadow_block_score_lt_6_5": "BLOCK",
                "shadow_score_gte_7_5": "BLOCK",
                "shadow_score_gte_8_0": "BLOCK",
                "shadow_score_gte_8_0_bear": "BLOCK",
            },
            {row["policy_name"]: row["decision"] for row in rows},
        )

    def test_evaluator_failure_does_not_break_alert_flow(self) -> None:
        self._activate_shadow_policy(filters={"allowed_symbols": ["AAPL"]})
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.evaluate_policy",
            side_effect=RuntimeError("policy boom"),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_repository_failure_does_not_break_alert_flow(self) -> None:
        self._activate_shadow_policy(filters={"allowed_symbols": ["AAPL"]})
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch.object(
            self.policy_repository,
            "insert_policy_decision",
            side_effect=sqlite3.OperationalError("db boom"),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_policy_db_init_happens_during_pipeline_setup_not_per_alert(self) -> None:
        with patch.object(self.policy_repository, "init_db") as init_db_mock:
            pipeline, _ = self._make_pipeline()

        init_db_mock.assert_called_once_with()

        with self._patch_alert_side_effects(), patch.object(
            self.policy_repository,
            "init_db",
            side_effect=AssertionError("init_db should not be called during per-alert processing"),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

    def test_success_log_includes_alert_decision_policy_version_and_mode(self) -> None:
        self._activate_shadow_policy(filters={"allowed_symbols": ["AAPL"]})
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs())

        self.assertTrue(
            any(
                "[policy] evaluated alert_id=alert-1 decision=ALLOW "
                "policy_name=execution_policy policy_version=1.0.0 mode=SHADOW"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_shadow_score_outputs_are_added_to_alert_context_only(self) -> None:
        pipeline, context = self._make_pipeline(
            shadow_scorer=SimpleNamespace(
                score=lambda intent_row, context_row: RuntimeShadowScoreOutput(
                    score_model_version="shadow_score_v1_expected_r_only",
                    feature_set_version="score_features_v1",
                    score_recipe_version="score_recipe_expected_r_v1",
                    score_recipe_mode="expected_r_only",
                    score_bucket_version="shadow_score_buckets_v1",
                    shadow_score_status="SCORED",
                    pred_expected_r=0.42,
                    pred_win_prob=None,
                    pred_expected_mfe=None,
                    score_shadow=0.42,
                    score_shadow_bucket="MEDIUM_POSITIVE",
                )
            )
        )

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        context_rows = [call.args[0] for call in side_effects["log_context"].call_args_list if call.args]
        intent_rows = [call.args[0] for call in side_effects["log_intent"].call_args_list if call.args]

        self.assertEqual(1, len(context_rows))
        self.assertEqual(0.42, context_rows[0].score_shadow)
        self.assertEqual("expected_r_only", context_rows[0].score_recipe_mode)
        self.assertEqual("SCORED", context_rows[0].shadow_score_status)
        self.assertFalse(hasattr(intent_rows[0], "score_shadow"))
        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_experiment_tag_is_added_to_alert_context_when_configured(self) -> None:
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.journal_builders.config.EXPERIMENT_TAG",
            "vacation_baseline_v1",
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        context_row = side_effects["log_context"].call_args.args[0]
        intent_row = side_effects["log_intent"].call_args.args[0]
        self.assertEqual("vacation_baseline_v1", context_row.experiment_tag)
        self.assertFalse(hasattr(intent_row, "experiment_tag"))
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_experiment_tag_unset_remains_null_and_backward_compatible(self) -> None:
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.journal_builders.config.EXPERIMENT_TAG",
            None,
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        context_row = side_effects["log_context"].call_args.args[0]
        self.assertIsNone(context_row.experiment_tag)
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_shadow_score_failure_does_not_break_alert_flow(self) -> None:
        pipeline, context = self._make_pipeline(
            shadow_scorer=SimpleNamespace(score=MagicMock(side_effect=RuntimeError("score boom")))
        )

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs())

        context_rows = [call.args[0] for call in side_effects["log_context"].call_args_list if call.args]
        self.assertIsNone(context_rows[0].score_shadow)
        self.assertEqual("ERROR", context_rows[0].shadow_score_status)
        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))
        self.assertTrue(
            any(
                "[shadow-score] error alert_id=alert-1 mode=SHADOW stage=score error=score boom"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_shadow_score_config_failure_does_not_break_alert_flow(self) -> None:
        with patch(
            "trade_scout.alert_pipeline.active_shadow_score_config",
            side_effect=ValueError("bad scoring config"),
        ), patch("builtins.print") as print_mock:
            pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        context_rows = [call.args[0] for call in side_effects["log_context"].call_args_list if call.args]
        self.assertEqual("shadow_score_disabled", context_rows[0].score_model_version)
        self.assertEqual("DISABLED", context_rows[0].shadow_score_status)
        self.assertIsNone(context_rows[0].score_shadow)
        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))
        self.assertTrue(
            any(
                "[shadow-score] error alert_id=UNKNOWN mode=SHADOW stage=config error=bad scoring config"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_unsupported_shadow_recipe_mode_degrades_without_breaking_alert_flow(self) -> None:
        with patch.dict("os.environ", {"SHADOW_SCORE_RECIPE_MODE": "blended"}), patch("builtins.print") as print_mock:
            pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs())

        context_row = side_effects["log_context"].call_args.args[0]
        self.assertEqual("DISABLED", context_row.shadow_score_status)
        self.assertIsNone(context_row.score_shadow)
        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))
        self.assertTrue(
            any(
                "unsupported SHADOW_SCORE_RECIPE_MODE='blended'"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_different_shadow_scores_do_not_change_live_alert_behavior(self) -> None:
        live_results = []

        for idx, score_shadow in enumerate((0.1, 0.9), start=1):
            pipeline, context = self._make_pipeline(
                shadow_scorer=SimpleNamespace(
                    score=lambda intent_row, context_row, value=score_shadow: RuntimeShadowScoreOutput(
                        score_model_version="shadow_score_v1_expected_r_only",
                        feature_set_version="score_features_v1",
                        score_recipe_version="score_recipe_expected_r_v1",
                        score_recipe_mode="expected_r_only",
                        score_bucket_version="shadow_score_buckets_v1",
                        shadow_score_status="SCORED",
                        pred_expected_r=value,
                        pred_win_prob=None,
                        pred_expected_mfe=None,
                        score_shadow=value,
                        score_shadow_bucket="LOW_POSITIVE" if value < 0.25 else "HIGH_POSITIVE",
                    )
                )
            )

            with self._patch_alert_side_effects() as side_effects, patch(
                "trade_scout.alert_pipeline.uuid4",
                side_effect=[f"provisional-{idx}", f"alert-{idx}"],
            ):
                pipeline.process(**self._process_kwargs())

            live_results.append(
                {
                    "notified": side_effects["notify"].called,
                    "watch_count": len(context.watch_mgr.added),
                    "watch_score": context.watch_mgr.added[0].score,
                    "intent_status": side_effects["log_intent"].call_args.args[0].alert_status,
                    "shadow_score": side_effects["log_context"].call_args.args[0].score_shadow,
                }
            )

        self.assertEqual(0.1, live_results[0]["shadow_score"])
        self.assertEqual(0.9, live_results[1]["shadow_score"])
        for result in live_results:
            self.assertTrue(result["notified"])
            self.assertEqual(1, result["watch_count"])
            self.assertEqual(8.2, result["watch_score"])
            self.assertEqual("ALERT_CREATED", result["intent_status"])

    def test_no_active_policy_log_includes_reason_policy_name_and_mode(self) -> None:
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs())

        self.assertTrue(
            any(
                "[policy] skipped alert_id=alert-1 reason=no_active_policy "
                "policy_name=execution_policy mode=SHADOW"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_evaluator_failure_log_includes_stage_policy_name_and_mode(self) -> None:
        self._activate_shadow_policy(filters={"allowed_symbols": ["AAPL"]})
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.evaluate_policy",
            side_effect=RuntimeError("policy boom"),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs())

        self.assertTrue(
            any(
                "[policy] error alert_id=alert-1 stage=evaluate "
                "policy_name=execution_policy mode=SHADOW error=policy boom"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_repository_failure_log_includes_stage_policy_name_and_mode(self) -> None:
        self._activate_shadow_policy(filters={"allowed_symbols": ["AAPL"]})
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch.object(
            self.policy_repository,
            "insert_policy_decision",
            side_effect=sqlite3.OperationalError("db boom"),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs())

        self.assertTrue(
            any(
                "[policy] error alert_id=alert-1 stage=persist "
                "policy_name=execution_policy mode=SHADOW error=db boom"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def _activate_shadow_policy(self, *, filters: dict[str, object]) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(
                conn,
                PolicyDefinitionRow(
                    policy_id="policy-1",
                    policy_name="execution_policy",
                    policy_version="1.0.0",
                    policy_hash="hash-1",
                    status="SHADOW_ONLY",
                    config_json=self._config_json(filters),
                    description="shadow policy",
                    created_ts_utc=utc_now_iso(),
                ),
            )
            activate_policy(
                conn,
                PolicyActivationRow(
                    activation_id="activation-1",
                    policy_id="policy-1",
                    policy_name="execution_policy",
                    policy_version="1.0.0",
                    mode="SHADOW",
                    effective_from_ts_utc="2026-04-05T11:00:00+00:00",
                    effective_to_ts_utc=None,
                ),
            )
        finally:
            conn.close()

    def _make_pipeline(self, *, shadow_scorer=None):
        context = SimpleNamespace(
            profile=SimpleNamespace(timeframe_min=5, watch_timeout_min=240),
            config_hash="cfg-123",
            cooldown=_FakeCooldown(),
            telemetry=_FakeTelemetry(),
            watch_mgr=_FakeWatchManager(),
        )
        return AlertPipeline(
            context,
            policy_repository=self.policy_repository,
            shadow_scorer=shadow_scorer,
        ), context

    def _process_kwargs(self, *, score=8.2, anchor_regime="bull"):
        strategy = SimpleNamespace(
            build_trade_plan=lambda symbol, profile, setup_candidate: SimpleNamespace(
                timeframe_min=profile.timeframe_min,
                entry_low=100.0,
                entry_high=101.0,
                stop_px=95.0,
                t1_px=110.0,
                t2_px=115.0,
                stop_distance=6.0,
            )
        )
        metadata = SimpleNamespace(
            strategy_id="long_breakout_continuation",
            strategy_version="1.0.0",
            setup_family="breakout_continuation",
            strategy_params_json="{}",
            strategy_params_hash="params-123",
        )
        return {
            "enabled_strategy": SimpleNamespace(strategy=strategy, metadata=metadata),
            "symbol": "AAPL",
            "now": datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            "ts": datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            "setup_candidate": SimpleNamespace(
                score=score,
                reasons=["strong breakout"],
                tier="A",
            ),
            "computed_for_log": {
                "close": 100.5,
                "atr": 1.2,
                "ma20": 99.0,
                "ma50": 98.0,
                "ma200": 95.0,
                "breakout_level": 100.0,
                "breakout_distance_atr": 0.5,
                "ma20_distance_atr": 0.4,
                "ma50_distance_atr": 0.8,
                "ma200_distance_atr": 1.5,
                "ma20_slope_pct": 0.2,
                "ma50_slope_pct": 0.3,
                "range_pct": 1.0,
                "close_location_in_bar": 0.9,
                "anchor_symbol": "SPY",
                "anchor_regime": anchor_regime,
                "anchor_close": 500.0,
                "anchor_ma50": 495.0,
                "anchor_ma200": 480.0,
                "vol_z": 0.1,
                "extended_atr": 0.2,
            },
        }

    def _patch_alert_side_effects(self):
        journal_log_alert = patch("trade_scout.alert_pipeline.journal.log_alert")
        journal_log_intent = patch("trade_scout.alert_pipeline.journal.log_alert_intent")
        journal_log_context = patch("trade_scout.alert_pipeline.journal.log_alert_context")
        discord_notify = patch("trade_scout.alert_pipeline.notifier.discord_notify")
        return _PatchBundle(
            log_alert=journal_log_alert.start(),
            log_intent=journal_log_intent.start(),
            log_context=journal_log_context.start(),
            notify=discord_notify.start(),
            stoppers=[
                journal_log_alert.stop,
                journal_log_intent.stop,
                journal_log_context.stop,
                discord_notify.stop,
            ],
        )

    @staticmethod
    def _config_json(filters: dict[str, object]) -> str:
        import json

        return json.dumps({"filters": filters}, sort_keys=True)


class _PatchBundle(dict):
    def __init__(self, *, log_alert, log_intent, log_context, notify, stoppers):
        super().__init__(
            log_alert=log_alert,
            log_intent=log_intent,
            log_context=log_context,
            notify=notify,
        )
        self._stoppers = stoppers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for stop in reversed(self._stoppers):
            stop()
        return False


class AlertPipelineAutoExecutionIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "execution.db")
        init_db(self.db_path)
        self.policy_repository = PolicyRepository(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_off_mode_does_not_auto_create_execution_request(self) -> None:
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_MODE",
            "OFF",
        ), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_SYMBOLS",
            {"BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"},
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1"],
        ):
            pipeline.process(**self._process_kwargs(symbol="BTC/USD"))

        conn = get_connection(self.db_path)
        try:
            row_count = conn.execute("SELECT COUNT(*) AS n FROM execution_requests").fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(0, row_count)
        self.assertTrue(side_effects["log_alert"].called)
        self.assertEqual(1, len(context.watch_mgr.added))

    def test_filtered_mode_auto_creates_requests_for_allowed_symbols_only(self) -> None:
        allowed_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]

        for idx, symbol in enumerate(allowed_symbols, start=1):
            pipeline, _ = self._make_pipeline()
            with self._patch_alert_side_effects(), patch(
                "trade_scout.alert_pipeline.config.AUTO_EXECUTION_MODE",
                "FILTERED",
            ), patch(
                "trade_scout.alert_pipeline.config.AUTO_EXECUTION_SYMBOLS",
                set(allowed_symbols),
            ), patch(
                "trade_scout.alert_pipeline.uuid4",
                side_effect=[f"provisional-{idx}", f"alert-{idx}", f"request-{idx}"],
            ):
                pipeline.process(**self._process_kwargs(symbol=symbol))

        pipeline, _ = self._make_pipeline()
        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_MODE",
            "FILTERED",
        ), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_SYMBOLS",
            set(allowed_symbols),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-non", "alert-non"],
        ):
            pipeline.process(**self._process_kwargs(symbol="DOGE/USD"))

        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT alert_id FROM execution_requests ORDER BY alert_id ASC"
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(["alert-1", "alert-2", "alert-3", "alert-4"], [row["alert_id"] for row in rows])

    def test_all_mode_auto_creates_request_for_any_symbol(self) -> None:
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_MODE",
            "ALL",
        ), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_SYMBOLS",
            set(),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1", "request-1"],
        ):
            pipeline.process(**self._process_kwargs(symbol="DOGE/USD"))

        conn = get_connection(self.db_path)
        try:
            row = conn.execute("SELECT * FROM execution_requests WHERE alert_id = ?", ("alert-1",)).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        self.assertEqual("auto_execution", row["requested_by"])
        self.assertEqual("AUTO_EXECUTION_ALL", row["reason"])

    def test_duplicate_processing_does_not_duplicate_execution_request(self) -> None:
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_MODE",
            "FILTERED",
        ), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_SYMBOLS",
            {"BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"},
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1", "request-1", "provisional-2", "alert-1", "request-2"],
        ):
            pipeline.process(**self._process_kwargs(symbol="BTC/USD"))
            pipeline.process(**self._process_kwargs(symbol="BTC/USD"))

        conn = get_connection(self.db_path)
        try:
            row_count = conn.execute("SELECT COUNT(*) AS n FROM execution_requests").fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(1, row_count)

    def test_auto_execution_log_includes_alert_symbol_and_mode(self) -> None:
        pipeline, _ = self._make_pipeline()

        with self._patch_alert_side_effects(), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_MODE",
            "FILTERED",
        ), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_SYMBOLS",
            {"BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"},
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1", "request-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs(symbol="BTC/USD"))

        self.assertTrue(
            any(
                "[execution-auto] inserted alert_id=alert-1 symbol=BTC/USD mode=FILTERED"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def test_auto_execution_insert_failure_does_not_break_alert_flow(self) -> None:
        pipeline, context = self._make_pipeline()

        with self._patch_alert_side_effects() as side_effects, patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_MODE",
            "FILTERED",
        ), patch(
            "trade_scout.alert_pipeline.config.AUTO_EXECUTION_SYMBOLS",
            {"BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"},
        ), patch(
            "trade_scout.alert_pipeline.insert_request",
            side_effect=sqlite3.OperationalError("execution db boom"),
        ), patch(
            "trade_scout.alert_pipeline.uuid4",
            side_effect=["provisional-1", "alert-1", "request-1"],
        ), patch("builtins.print") as print_mock:
            pipeline.process(**self._process_kwargs(symbol="BTC/USD"))

        self.assertTrue(side_effects["notify"].called)
        self.assertEqual(1, len(context.watch_mgr.added))
        self.assertTrue(
            any(
                "[execution-auto] error alert_id=alert-1 symbol=BTC/USD mode=FILTERED "
                "stage=insert_request error=execution db boom"
                in str(call.args[0])
                for call in print_mock.call_args_list
                if call.args
            )
        )

    def _make_pipeline(self):
        context = SimpleNamespace(
            profile=SimpleNamespace(timeframe_min=5, watch_timeout_min=240),
            config_hash="cfg-123",
            cooldown=_FakeCooldown(),
            telemetry=_FakeTelemetry(),
            watch_mgr=_FakeWatchManager(),
        )
        return AlertPipeline(context, policy_repository=self.policy_repository), context

    def _process_kwargs(self, *, symbol: str):
        strategy = SimpleNamespace(
            build_trade_plan=lambda symbol, profile, setup_candidate: SimpleNamespace(
                timeframe_min=profile.timeframe_min,
                entry_low=100.0,
                entry_high=101.0,
                stop_px=95.0,
                t1_px=110.0,
                t2_px=115.0,
                stop_distance=6.0,
            )
        )
        metadata = SimpleNamespace(
            strategy_id="long_breakout_continuation",
            strategy_version="1.0.0",
            setup_family="breakout_continuation",
            strategy_params_json="{}",
            strategy_params_hash="params-123",
        )
        return {
            "enabled_strategy": SimpleNamespace(strategy=strategy, metadata=metadata),
            "symbol": symbol,
            "now": datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            "ts": datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            "setup_candidate": SimpleNamespace(
                score=8.2,
                reasons=["strong breakout"],
                tier="A",
            ),
            "computed_for_log": {
                "close": 100.5,
                "atr": 1.2,
                "ma20": 99.0,
                "ma50": 98.0,
                "ma200": 95.0,
                "breakout_level": 100.0,
                "breakout_distance_atr": 0.5,
                "ma20_distance_atr": 0.4,
                "ma50_distance_atr": 0.8,
                "ma200_distance_atr": 1.5,
                "ma20_slope_pct": 0.2,
                "ma50_slope_pct": 0.3,
                "range_pct": 1.0,
                "close_location_in_bar": 0.9,
                "anchor_symbol": "SPY",
                "anchor_regime": "bull",
                "anchor_close": 500.0,
                "anchor_ma50": 495.0,
                "anchor_ma200": 480.0,
                "vol_z": 0.1,
                "extended_atr": 0.2,
            },
        }

    def _patch_alert_side_effects(self):
        journal_log_alert = patch("trade_scout.alert_pipeline.journal.log_alert")
        journal_log_intent = patch("trade_scout.alert_pipeline.journal.log_alert_intent")
        journal_log_context = patch("trade_scout.alert_pipeline.journal.log_alert_context")
        discord_notify = patch("trade_scout.alert_pipeline.notifier.discord_notify")
        return _PatchBundle(
            log_alert=journal_log_alert.start(),
            log_intent=journal_log_intent.start(),
            log_context=journal_log_context.start(),
            notify=discord_notify.start(),
            stoppers=[
                journal_log_alert.stop,
                journal_log_intent.stop,
                journal_log_context.stop,
                discord_notify.stop,
            ],
        )
