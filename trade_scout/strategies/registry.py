from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .base import Strategy
from .types import StrategyMetadata
from .long_breakout_continuation import LongBreakoutContinuationStrategy

StrategyBuilder = Callable[[Mapping[str, object]], Strategy]


@dataclass
class StrategyRegistry:
    _builders: dict[str, StrategyBuilder] = field(default_factory=dict)

    def register(self, strategy_id: str, builder: StrategyBuilder) -> None:
        if strategy_id in self._builders:
            raise ValueError(f"Strategy already registered: {strategy_id}")
        self._builders[strategy_id] = builder

    def instantiate(self, strategy_id: str, params: Mapping[str, object] | None = None) -> Strategy:
        builder = self._builders.get(strategy_id)
        if builder is None:
            raise ValueError(f"Unknown strategy_id: {strategy_id}")
        return builder(params or {})

    def instantiate_enabled(
        self,
        enabled_strategy_ids: list[str],
        strategy_params: Mapping[str, Mapping[str, object]] | None = None,
    ) -> list[Strategy]:
        params_by_id = strategy_params or {}
        return [
            self.instantiate(strategy_id, params_by_id.get(strategy_id))
            for strategy_id in enabled_strategy_ids
        ]


def _build_long_breakout_continuation(params: Mapping[str, object]) -> Strategy:
    strategy_version = params.get("strategy_version", "v3_research_step2")
    if not isinstance(strategy_version, str) or not strategy_version.strip():
        raise ValueError("strategy_version must be a non-empty string")
    return LongBreakoutContinuationStrategy(strategy_version=strategy_version.strip())


def normalize_strategy_params(params: Mapping[str, object] | None = None) -> dict[str, object]:
    if params is None:
        return {}
    return {str(key): params[key] for key in sorted(params.keys())}


def strategy_params_json(params: Mapping[str, object] | None = None) -> str:
    return json.dumps(normalize_strategy_params(params), sort_keys=True, separators=(",", ":"))


def strategy_params_hash(params: Mapping[str, object] | None = None) -> str:
    payload = strategy_params_json(params)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_strategy_metadata(
    strategy: Strategy,
    params: Mapping[str, object] | None = None,
) -> StrategyMetadata:
    base = strategy.metadata_for_journaling()
    params_json = strategy_params_json(params)
    return StrategyMetadata(
        strategy_id=base.strategy_id,
        strategy_version=base.strategy_version,
        setup_family=base.setup_family,
        strategy_params_json=params_json,
        strategy_params_hash=strategy_params_hash(params),
        journal_metadata=base.journal_metadata,
    )


def make_strategy_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register("long_breakout_continuation", _build_long_breakout_continuation)
    return registry


def load_enabled_strategies(
    enabled_strategy_ids: list[str],
    strategy_params: Mapping[str, Mapping[str, object]] | None = None,
) -> list[Strategy]:
    return make_strategy_registry().instantiate_enabled(enabled_strategy_ids, strategy_params)
