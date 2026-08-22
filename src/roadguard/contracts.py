"""Immutable V1 data and ML contracts.

The locked V1 production profile (:class:`V1Contract`) cannot be changed by
configuration: every value is enforced to match a module constant at
validation time. Tunable runtime settings live in
:mod:`roadguard.config`. Small dataset sizes needed by tests or generators
use the separate :class:`DatasetSpec`, which is not part of the production
configuration path.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Final

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

V1_DATASET_SEGMENTS: Final[int] = 300
V1_DATASET_MONTHS_PER_SEGMENT: Final[int] = 48
V1_DATASET_OBSERVATIONS: Final[int] = 14_400
V1_TRAIN_FRACTION: Final[float] = 0.70
V1_VALIDATION_FRACTION: Final[float] = 0.15
V1_TEST_FRACTION: Final[float] = 0.15
V1_MAINTENANCE_WINDOW_DAYS: Final[int] = 30
V1_TRAIN_DATE_COUNT: Final[int] = 34
V1_VALIDATION_DATE_COUNT: Final[int] = 7
V1_TEST_DATE_COUNT: Final[int] = 7
V1_RISK_BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 30), (31, 60), (61, 80), (81, 100))
V1_OBSERVATION_START: Final[date] = date(2022, 1, 1)


def _reject_boolean(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("boolean values are not allowed for numeric fields")
    return value


NonBooleanInt = Annotated[int, BeforeValidator(_reject_boolean)]
NonBooleanFloat = Annotated[float, BeforeValidator(_reject_boolean)]


class RiskBand(BaseModel):
    """An inclusive integer score range, e.g. ``RiskBand(lower=0, upper=30)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: NonBooleanInt
    upper: NonBooleanInt

    @model_validator(mode="after")
    def _validate_range(self) -> RiskBand:
        if not 0 <= self.lower <= 100:
            raise ValueError("Risk band bounds must be within 0 and 100")
        if not 0 <= self.upper <= 100:
            raise ValueError("Risk band bounds must be within 0 and 100")
        if self.lower > self.upper:
            raise ValueError("Risk band lower bound must not exceed its upper bound")
        return self


class RiskBands(BaseModel):
    """The four risk bands: LOW, MEDIUM, HIGH, CRITICAL.

    Bands must be contiguous and together cover 0 through 100 inclusive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: RiskBand = RiskBand(lower=0, upper=30)
    medium: RiskBand = RiskBand(lower=31, upper=60)
    high: RiskBand = RiskBand(lower=61, upper=80)
    critical: RiskBand = RiskBand(lower=81, upper=100)

    @model_validator(mode="after")
    def _validate_coverage(self) -> RiskBands:
        if self.low.lower != 0:
            raise ValueError("Risk bands must start at 0")
        if self.critical.upper != 100:
            raise ValueError("Risk bands must end at 100")
        pairs = (
            (self.low, self.medium),
            (self.medium, self.high),
            (self.high, self.critical),
        )
        for lower_band, upper_band in pairs:
            if lower_band.upper + 1 != upper_band.lower:
                raise ValueError("Risk bands must be contiguous without gaps or overlaps")
        return self


class V1Contract(BaseModel):
    """The locked V1 production data and ML contract.

    Every field is enforced to equal its module constant; constructing the
    model with any other value raises a validation error. Configuration
    (YAML or environment) cannot change these values because they are not
    fields of :class:`roadguard.config.RoadGuardConfig`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_segments: NonBooleanInt = V1_DATASET_SEGMENTS
    dataset_months_per_segment: NonBooleanInt = V1_DATASET_MONTHS_PER_SEGMENT
    dataset_observations: NonBooleanInt = V1_DATASET_OBSERVATIONS
    train_fraction: NonBooleanFloat = V1_TRAIN_FRACTION
    validation_fraction: NonBooleanFloat = V1_VALIDATION_FRACTION
    test_fraction: NonBooleanFloat = V1_TEST_FRACTION
    maintenance_window_days: NonBooleanInt = V1_MAINTENANCE_WINDOW_DAYS
    train_date_count: NonBooleanInt = V1_TRAIN_DATE_COUNT
    validation_date_count: NonBooleanInt = V1_VALIDATION_DATE_COUNT
    test_date_count: NonBooleanInt = V1_TEST_DATE_COUNT
    risk_bands: RiskBands = Field(default_factory=RiskBands)

    @model_validator(mode="after")
    def _validate_locked(self) -> V1Contract:
        if self.dataset_segments != V1_DATASET_SEGMENTS:
            raise ValueError(f"dataset_segments is locked to {V1_DATASET_SEGMENTS}")
        if self.dataset_months_per_segment != V1_DATASET_MONTHS_PER_SEGMENT:
            raise ValueError(
                f"dataset_months_per_segment is locked to {V1_DATASET_MONTHS_PER_SEGMENT}"
            )
        if self.dataset_observations != V1_DATASET_OBSERVATIONS:
            raise ValueError(f"dataset_observations is locked to {V1_DATASET_OBSERVATIONS}")
        if self.train_fraction != V1_TRAIN_FRACTION:
            raise ValueError(f"train_fraction is locked to {V1_TRAIN_FRACTION}")
        if self.validation_fraction != V1_VALIDATION_FRACTION:
            raise ValueError(f"validation_fraction is locked to {V1_VALIDATION_FRACTION}")
        if self.test_fraction != V1_TEST_FRACTION:
            raise ValueError(f"test_fraction is locked to {V1_TEST_FRACTION}")
        if self.maintenance_window_days != V1_MAINTENANCE_WINDOW_DAYS:
            raise ValueError(f"maintenance_window_days is locked to {V1_MAINTENANCE_WINDOW_DAYS}")
        if self.train_date_count != V1_TRAIN_DATE_COUNT:
            raise ValueError(f"train_date_count is locked to {V1_TRAIN_DATE_COUNT}")
        if self.validation_date_count != V1_VALIDATION_DATE_COUNT:
            raise ValueError(f"validation_date_count is locked to {V1_VALIDATION_DATE_COUNT}")
        if self.test_date_count != V1_TEST_DATE_COUNT:
            raise ValueError(f"test_date_count is locked to {V1_TEST_DATE_COUNT}")
        date_total = self.train_date_count + self.validation_date_count + self.test_date_count
        if date_total != self.dataset_months_per_segment:
            raise ValueError(
                f"date counts must sum to dataset_months_per_segment (got {date_total})"
            )
        bands = (
            (self.risk_bands.low.lower, self.risk_bands.low.upper),
            (self.risk_bands.medium.lower, self.risk_bands.medium.upper),
            (self.risk_bands.high.lower, self.risk_bands.high.upper),
            (self.risk_bands.critical.lower, self.risk_bands.critical.upper),
        )
        if bands != V1_RISK_BANDS:
            raise ValueError(
                "risk bands are locked to LOW 0-30, MEDIUM 31-60, HIGH 61-80, CRITICAL 81-100"
            )
        return self


class DatasetSpec(BaseModel):
    """A validated dataset specification for generators and unit tests.

    Intended for explicit small test/generator specifications. It is
    deliberately separate from :class:`V1Contract` and from
    :mod:`roadguard.config`, so a small test dataset can never be loaded as
    the production V1 profile.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_segments: NonBooleanInt
    dataset_months_per_segment: NonBooleanInt
    dataset_observations: NonBooleanInt

    @model_validator(mode="after")
    def _validate_consistency(self) -> DatasetSpec:
        if self.dataset_segments < 1:
            raise ValueError("dataset_segments must be a positive integer")
        if self.dataset_months_per_segment < 1:
            raise ValueError("dataset_months_per_segment must be a positive integer")
        if self.dataset_observations < 1:
            raise ValueError("dataset_observations must be a positive integer")
        expected = self.dataset_segments * self.dataset_months_per_segment
        if self.dataset_observations != expected:
            raise ValueError(
                "dataset_observations must equal dataset_segments * dataset_months_per_segment"
            )
        return self


__all__ = [
    "DatasetSpec",
    "RiskBand",
    "RiskBands",
    "V1Contract",
    "V1_DATASET_MONTHS_PER_SEGMENT",
    "V1_DATASET_OBSERVATIONS",
    "V1_DATASET_SEGMENTS",
    "V1_MAINTENANCE_WINDOW_DAYS",
    "V1_OBSERVATION_START",
    "V1_RISK_BANDS",
    "V1_TEST_DATE_COUNT",
    "V1_TEST_FRACTION",
    "V1_TRAIN_DATE_COUNT",
    "V1_TRAIN_FRACTION",
    "V1_VALIDATION_DATE_COUNT",
    "V1_VALIDATION_FRACTION",
]
