from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowScoreBucketConfig:
    bucket_version: str
    low_positive_threshold: float
    medium_positive_threshold: float


DEFAULT_SHADOW_SCORE_BUCKET_CONFIG = ShadowScoreBucketConfig(
    bucket_version="shadow_score_buckets_v1",
    low_positive_threshold=0.25,
    medium_positive_threshold=0.75,
)


def shadow_score_bucket(
    score_shadow: float | None,
    bucket_config: ShadowScoreBucketConfig = DEFAULT_SHADOW_SCORE_BUCKET_CONFIG,
) -> str | None:
    if score_shadow is None:
        return None
    if score_shadow < 0:
        return "NEGATIVE"
    if score_shadow < bucket_config.low_positive_threshold:
        return "LOW_POSITIVE"
    if score_shadow < bucket_config.medium_positive_threshold:
        return "MEDIUM_POSITIVE"
    return "HIGH_POSITIVE"
