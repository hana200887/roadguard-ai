"""Phase 7 deterministic point-in-time feature registry and frame builder.

The module accepts only the validated Phase 6 export boundary. It revalidates
the copied frames and builds a target-free, non-learned feature frame; splitting
and preprocessing deliberately remain later-phase responsibilities.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import pandas as pd

from roadguard._db_types import RepositoryExport
from roadguard.contracts import DatasetSpec
from roadguard.data_quality import validate_cleaned_dataset
from roadguard.observations import NEVER_MAINTAINED_DAYS_CAP

FeatureKind = Literal["categorical", "datetime", "numeric"]


class FeatureInputError(ValueError):
    """Raised when a Phase 6 export cannot safely produce Phase 7 features."""


@dataclass(frozen=True)
class FeatureDefinition:
    """One frozen V1 feature and the raw source columns it requires."""

    name: str
    source_columns: tuple[str, ...]
    kind: FeatureKind
    classification_allowed: bool
    regression_allowed: bool


FEATURE_KEY_COLUMNS: Final[tuple[str, str]] = ("segment_id", "date")
FEATURE_REGISTRY: Final[tuple[FeatureDefinition, ...]] = (
    FeatureDefinition("province", ("province",), "categorical", True, True),
    FeatureDefinition("road_type", ("road_type",), "categorical", True, True),
    FeatureDefinition("construction_date", ("construction_date",), "datetime", True, True),
    FeatureDefinition("road_length_km", ("road_length_km",), "numeric", True, True),
    FeatureDefinition("traffic_volume", ("traffic_volume",), "numeric", True, True),
    FeatureDefinition("heavy_vehicle_ratio", ("heavy_vehicle_ratio",), "numeric", True, True),
    FeatureDefinition("road_age_days", ("road_age_days",), "numeric", True, True),
    FeatureDefinition("rainfall_mm", ("rainfall_mm",), "numeric", True, True),
    FeatureDefinition("temperature", ("temperature",), "numeric", True, True),
    FeatureDefinition("humidity", ("humidity",), "numeric", True, True),
    FeatureDefinition(
        "days_since_last_maintenance",
        ("days_since_last_maintenance",),
        "numeric",
        True,
        True,
    ),
    FeatureDefinition("previous_repairs", ("previous_repairs",), "numeric", True, True),
    FeatureDefinition("road_condition_score", ("road_condition_score",), "numeric", True, True),
    FeatureDefinition(
        "marking_condition_score", ("marking_condition_score",), "numeric", True, True
    ),
    FeatureDefinition(
        "guardrail_condition_score", ("guardrail_condition_score",), "numeric", True, True
    ),
    FeatureDefinition("sign_condition_score", ("sign_condition_score",), "numeric", True, True),
    FeatureDefinition("accident_count_30d", ("accident_count_30d",), "numeric", True, True),
    FeatureDefinition("accident_count_365d", ("accident_count_365d",), "numeric", True, True),
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(definition.name for definition in FEATURE_REGISTRY)
CLASSIFICATION_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.classification_allowed
)
REGRESSION_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.regression_allowed
)
FEATURE_FRAME_COLUMNS: Final[tuple[str, ...]] = FEATURE_KEY_COLUMNS + FEATURE_COLUMNS
_FEATURE_INT_COLUMNS: Final[tuple[str, ...]] = (
    "traffic_volume",
    "road_age_days",
    "days_since_last_maintenance",
    "previous_repairs",
    "road_condition_score",
    "marking_condition_score",
    "guardrail_condition_score",
    "sign_condition_score",
    "accident_count_30d",
    "accident_count_365d",
)
_FEATURE_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "road_length_km",
    "heavy_vehicle_ratio",
    "rainfall_mm",
    "temperature",
    "humidity",
)


def build_feature_frame(dataset: RepositoryExport, spec: DatasetSpec) -> pd.DataFrame:
    """Build the exact, canonical Phase 7 feature frame from a Phase 6 export.

    Targets and maintenance-event keys are used only to fresh-validate the
    source boundary. They never contribute a returned feature column or value.
    Caller-owned dataframes are deep-copied before validation and joining.
    """
    segments, observations, maintenance_events = _validated_source_copies(dataset, spec)
    _validate_maintenance_feature_provenance(observations, maintenance_events)
    try:
        joined = observations.merge(
            segments,
            how="inner",
            on="segment_id",
            sort=False,
            validate="many_to_one",
        )
    except pd.errors.MergeError as exc:
        raise FeatureInputError("feature source has an invalid segment-observation join") from exc
    if len(joined) != len(observations):
        raise FeatureInputError("feature source contains observations without a matching segment")

    frame = joined.loc[:, list(FEATURE_FRAME_COLUMNS)].copy()
    return (
        _normalize_feature_dtypes(frame)
        .sort_values(list(FEATURE_KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )


def _validated_source_copies(
    dataset: RepositoryExport,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if type(dataset) is not RepositoryExport:
        raise TypeError("dataset must be a RepositoryExport")
    if type(spec) is not DatasetSpec:
        raise TypeError("spec must be a DatasetSpec")

    segments = dataset.segments.copy(deep=True)
    observations = dataset.observations.copy(deep=True)
    targets = dataset.targets.copy(deep=True)
    maintenance_events = dataset.maintenance_events.copy(deep=True)
    report = validate_cleaned_dataset(segments, observations, targets, maintenance_events, spec)
    if report.error_count > 0:
        codes = ", ".join(sorted(report.counts_by_code))
        raise FeatureInputError(f"cleaned repository export validation failed with errors: {codes}")
    return segments, observations, maintenance_events


def _validate_maintenance_feature_provenance(
    observations: pd.DataFrame,
    maintenance_events: pd.DataFrame,
) -> None:
    event_dates_by_segment: dict[str, tuple[date, ...]] = {}
    for segment_id, group in maintenance_events.groupby("segment_id", sort=False):
        event_dates_by_segment[str(segment_id)] = tuple(
            sorted(pd.Timestamp(value).date() for value in group["maintenance_date"])
        )

    for _, row in observations.iterrows():
        segment_id = str(row["segment_id"])
        observation_date = pd.Timestamp(row["date"]).date()
        prior_dates = event_dates_by_segment.get(segment_id, ())
        prior_count = bisect_left(prior_dates, observation_date)
        expected_previous_repairs = prior_count
        if prior_count:
            expected_days_since = (observation_date - prior_dates[prior_count - 1]).days
        else:
            expected_days_since = min(int(row["road_age_days"]), NEVER_MAINTAINED_DAYS_CAP)

        if int(row["previous_repairs"]) != expected_previous_repairs:
            raise FeatureInputError(
                "previous_repairs does not match strictly-prior maintenance events "
                f"for segment {segment_id!r} at {observation_date}"
            )
        if int(row["days_since_last_maintenance"]) != expected_days_since:
            raise FeatureInputError(
                "days_since_last_maintenance does not match strictly-prior maintenance events "
                f"for segment {segment_id!r} at {observation_date}"
            )


def _normalize_feature_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ("segment_id", "province", "road_type"):
        normalized[column] = normalized[column].astype(object)
    for column in ("date", "construction_date"):
        normalized[column] = pd.to_datetime(normalized[column]).astype("datetime64[ns]")
    for column in _FEATURE_INT_COLUMNS:
        normalized[column] = normalized[column].astype("int64")
    for column in _FEATURE_FLOAT_COLUMNS:
        normalized[column] = normalized[column].astype("float64")
    return normalized


__all__ = [
    "CLASSIFICATION_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "FEATURE_FRAME_COLUMNS",
    "FEATURE_KEY_COLUMNS",
    "FEATURE_REGISTRY",
    "REGRESSION_FEATURE_COLUMNS",
    "FeatureDefinition",
    "FeatureInputError",
    "build_feature_frame",
]
