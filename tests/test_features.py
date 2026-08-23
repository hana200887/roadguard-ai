"""Phase 7 point-in-time feature-frame contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from roadguard import (
    DatasetSpec,
    RepositoryExport,
    clean_raw_dataset,
    derive_observation_targets,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
)
from roadguard.features import (
    FEATURE_COLUMNS,
    FEATURE_FRAME_COLUMNS,
    FEATURE_KEY_COLUMNS,
    FEATURE_REGISTRY,
    build_feature_frame,
)

SPEC = DatasetSpec(dataset_segments=2, dataset_months_per_segment=12, dataset_observations=24)
START = date(2022, 1, 1)


@pytest.fixture
def dataset() -> RepositoryExport:
    segments = generate_segments(SPEC, 42, observation_start=START)
    events = generate_maintenance_events(segments, SPEC, 42, start_date=START)
    timeline = generate_accident_timeline(segments, SPEC, 42, start_date=START)
    observations = generate_observations(segments, events, timeline, SPEC, 42, start_date=START)
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, SPEC)
    return RepositoryExport(
        segments=cleaned.segments,
        observations=cleaned.observations,
        targets=cleaned.targets,
        maintenance_events=cleaned.maintenance_events,
    )


def test_registry_has_exact_source_feature_order() -> None:
    assert FEATURE_KEY_COLUMNS == ("segment_id", "date")
    assert tuple(definition.name for definition in FEATURE_REGISTRY) == FEATURE_COLUMNS
    assert FEATURE_FRAME_COLUMNS == FEATURE_KEY_COLUMNS + FEATURE_COLUMNS
    assert FEATURE_COLUMNS == (
        "province",
        "road_type",
        "construction_date",
        "road_length_km",
        "traffic_volume",
        "heavy_vehicle_ratio",
        "road_age_days",
        "rainfall_mm",
        "temperature",
        "humidity",
        "days_since_last_maintenance",
        "previous_repairs",
        "road_condition_score",
        "marking_condition_score",
        "guardrail_condition_score",
        "sign_condition_score",
        "accident_count_30d",
        "accident_count_365d",
    )


def test_build_feature_frame_has_exact_target_free_schema(dataset: RepositoryExport) -> None:
    frame = build_feature_frame(dataset, SPEC)

    assert tuple(frame.columns) == FEATURE_FRAME_COLUMNS
    assert len(frame) == SPEC.dataset_observations
    assert not {
        "days_until_maintenance",
        "maintenance_within_30_days",
        "maintenance_date",
        "maintenance_cost",
        "thermoplastic_paint_kg",
        "traffic_base",
    }.intersection(frame.columns)


def test_build_feature_frame_is_canonical_and_does_not_mutate_inputs(
    dataset: RepositoryExport,
) -> None:
    before_segments = dataset.segments.copy(deep=True)
    before_observations = dataset.observations.copy(deep=True)
    shuffled = replace(
        dataset,
        segments=dataset.segments.sample(frac=1.0, random_state=3).reset_index(drop=True),
        observations=dataset.observations.sample(frac=1.0, random_state=7).reset_index(drop=True),
        targets=dataset.targets.sample(frac=1.0, random_state=11).reset_index(drop=True),
        maintenance_events=dataset.maintenance_events.sample(
            frac=1.0, random_state=13
        ).reset_index(drop=True),
    )

    expected = build_feature_frame(dataset, SPEC)
    actual = build_feature_frame(shuffled, SPEC)

    assert_frame_equal(actual, expected)
    assert_frame_equal(dataset.segments, before_segments)
    assert_frame_equal(dataset.observations, before_observations)


def test_valid_future_event_cannot_change_feature_values(dataset: RepositoryExport) -> None:
    extra = dataset.maintenance_events.iloc[[-1]].copy()
    extra["maintenance_date"] = (
        dataset.maintenance_events["maintenance_date"].max() + pd.DateOffset(years=1)
    )
    changed = replace(
        dataset,
        maintenance_events=pd.concat([dataset.maintenance_events, extra], ignore_index=True),
    )

    assert_frame_equal(build_feature_frame(changed, SPEC), build_feature_frame(dataset, SPEC))


def test_build_feature_frame_revalidates_forged_export(dataset: RepositoryExport) -> None:
    targets = dataset.targets.copy()
    targets.loc[0, "days_until_maintenance"] += 1
    forged = replace(dataset, targets=targets)

    with pytest.raises(ValueError, match="target_event_inconsistency"):
        build_feature_frame(forged, SPEC)


def test_build_feature_frame_rejects_non_export(dataset: RepositoryExport) -> None:
    with pytest.raises(TypeError, match="RepositoryExport"):
        build_feature_frame(dataset.observations, SPEC)
