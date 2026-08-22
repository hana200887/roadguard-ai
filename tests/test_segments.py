"""Tests for deterministic segment master generation."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from roadguard import DatasetSpec, SegmentMaster, generate_segments
from roadguard.segments import PROVINCES, ROAD_TYPES, SEGMENT_COLUMNS

SPEC = DatasetSpec(dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400)
ID_PATTERN = re.compile(r"^QL\d{2}-KM\d+-\d+$")


class TestSegmentIdentity:
    def test_ids_match_business_pattern(self) -> None:
        ids = generate_segments(SPEC, 42)["segment_id"].tolist()
        assert all(ID_PATTERN.fullmatch(sid) is not None for sid in ids)

    def test_ids_are_unique(self) -> None:
        ids = generate_segments(SPEC, 42)["segment_id"].tolist()
        assert len(ids) == SPEC.dataset_segments
        assert len(set(ids)) == len(ids)

    def test_ids_are_stable_across_seeds(self) -> None:
        ids_a = generate_segments(SPEC, 42)["segment_id"].tolist()
        ids_b = generate_segments(SPEC, 7)["segment_id"].tolist()
        assert ids_a == ids_b

    def test_marker_span_matches_road_length(self) -> None:
        frame = generate_segments(SPEC, 42)
        for segment_id, length in zip(frame["segment_id"], frame["road_length_km"], strict=True):
            match = re.fullmatch(r"^QL\d{2}-KM(\d+)-(\d+)$", segment_id)
            assert match is not None
            span = int(match.group(2)) - int(match.group(1))
            assert float(span) == length

    def test_road_length_stable_across_seeds(self) -> None:
        first = generate_segments(SPEC, 42)["road_length_km"].tolist()
        second = generate_segments(SPEC, 7)["road_length_km"].tolist()
        assert first == second


class TestDeterminism:
    def test_same_seed_gives_identical_segments(self) -> None:
        assert_frame_equal(generate_segments(SPEC, 42), generate_segments(SPEC, 42))

    def test_different_seeds_change_stochastic_values(self) -> None:
        first = generate_segments(SPEC, 42)
        second = generate_segments(SPEC, 43)
        assert not first.equals(second)
        assert not first["traffic_base"].equals(second["traffic_base"])
        assert not first["construction_date"].equals(second["construction_date"])


class TestSegmentValues:
    def test_columns_match_contract_order(self) -> None:
        frame = generate_segments(SPEC, 42)
        assert list(frame.columns) == list(SEGMENT_COLUMNS)

    def test_province_and_road_type_from_registries(self) -> None:
        frame = generate_segments(SPEC, 42)
        assert set(frame["province"]).issubset(set(PROVINCES))
        assert set(frame["road_type"]).issubset(set(ROAD_TYPES))

    def test_construction_date_before_observation_start(self) -> None:
        frame = generate_segments(SPEC, 42)
        assert (pd.to_datetime(frame["construction_date"]) < pd.Timestamp("2022-01-01")).all()

    def test_numeric_ranges(self) -> None:
        frame = generate_segments(SPEC, 42)
        assert frame["road_length_km"].between(4.0, 12.0).all()
        assert (frame["traffic_base"] >= 800).all()
        assert (frame["traffic_base"] <= 20_000).all()
        assert frame["heavy_vehicle_ratio_base"].between(0.05, 0.50).all()
        assert frame["weather_exposure"].between(0.6, 1.6).all()
        assert frame["deterioration_rate"].between(0.3, 1.8).all()
        assert frame["accident_propensity"].between(0.3, 2.5).all()
        assert frame["initial_condition"].between(60, 100).all()

    def test_small_spec_respected(self) -> None:
        small = DatasetSpec(
            dataset_segments=10, dataset_months_per_segment=12, dataset_observations=120
        )
        assert len(generate_segments(small, 1)) == 10

    def test_invalid_seed_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_segments(SPEC, 0)


class TestFromRecordValidation:
    def _valid_record(self) -> dict[str, object]:
        return {
            "segment_id": "QL01-KM10-20",
            "province": "NA",
            "road_type": "national",
            "construction_date": date(2010, 1, 1),
            "road_length_km": 8.5,
            "traffic_base": 5_000,
            "heavy_vehicle_ratio_base": 0.2,
            "weather_exposure": 1.0,
            "deterioration_rate": 0.8,
            "accident_propensity": 1.0,
            "initial_condition": 80,
        }

    def test_valid_record_roundtrip(self) -> None:
        master = SegmentMaster.from_record(self._valid_record())
        assert master.segment_id == "QL01-KM10-20"
        assert master.construction_date == date(2010, 1, 1)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("road_length_km", 0.0),
            ("traffic_base", -1),
            ("heavy_vehicle_ratio_base", 1.5),
            ("initial_condition", 101),
            ("deterioration_rate", 0.0),
            ("accident_propensity", -0.1),
            ("weather_exposure", 0.0),
        ],
    )
    def test_invalid_record_fields_rejected(self, field: str, value: object) -> None:
        record = self._valid_record()
        record[field] = value
        with pytest.raises(ValueError):
            SegmentMaster.from_record(record)

    def test_timestamp_construction_date_accepted(self) -> None:
        record = self._valid_record()
        record["construction_date"] = pd.Timestamp("2010-06-15")
        master = SegmentMaster.from_record(record)
        assert master.construction_date == date(2010, 6, 15)

    def test_string_construction_date_accepted(self) -> None:
        record = self._valid_record()
        record["construction_date"] = "2010-06-15"
        master = SegmentMaster.from_record(record)
        assert master.construction_date == date(2010, 6, 15)
