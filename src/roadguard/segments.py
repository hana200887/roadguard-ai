"""Deterministic segment master generation for the V1 synthetic network.

``generate_segments`` produces the static ``road_segments`` table plus the
latent state needed by later simulation (traffic base, heavy-vehicle base,
weather exposure, deterioration rate, accident propensity, initial
condition). Segment identifiers are registry-based business codes such as
``QL01-KM134-135`` and ``road_length_km`` equals the kilometre-marker span
``end - start``; both are deterministic and stable across seeds. Randomness
affects only the stochastic attributes, never the identifiers or physical
scale.

Determinism: every random draw comes from ``SeedSequence(seed)`` child
stream 0 (stream 1 is reserved for the maintenance-event engine).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final

import numpy as np
import pandas as pd

from roadguard.contracts import V1_OBSERVATION_START, DatasetSpec

SEGMENT_ID_PATTERN: Final[str] = r"^QL\d{2}-KM\d+-\d+$"
ROAD_CODES: Final[tuple[str, ...]] = ("QL01", "QL14", "QL19", "QL24", "QL27", "QL40")
PROVINCES: Final[tuple[str, ...]] = ("NA", "TH", "QB", "DN", "GL", "LA")
ROAD_TYPES: Final[tuple[str, ...]] = ("highway", "national", "provincial", "urban", "rural")

SEGMENT_COLUMNS: Final[tuple[str, ...]] = (
    "segment_id",
    "province",
    "road_type",
    "construction_date",
    "road_length_km",
    "traffic_base",
    "heavy_vehicle_ratio_base",
    "weather_exposure",
    "deterioration_rate",
    "accident_propensity",
    "initial_condition",
)

_MIN_AGE_YEARS: Final[int] = 5
_MAX_AGE_YEARS: Final[int] = 30


@dataclass(frozen=True)
class SegmentMaster:
    """Static road-segment attributes plus latent simulation state.

    The first five fields are the contract columns of ``road_segments``;
    ``road_length_km`` equals the kilometre-marker span of the business
    identifier (``end_km - start_km``). The remaining fields are latent
    state consumed by the maintenance-event engine and later observation
    simulation.
    """

    segment_id: str
    province: str
    road_type: str
    construction_date: date
    road_length_km: float
    traffic_base: int
    heavy_vehicle_ratio_base: float
    weather_exposure: float
    deterioration_rate: float
    accident_propensity: float
    initial_condition: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SegmentMaster:
        """Build a master from a validated DataFrame row dict."""
        construction_date = _as_date(record["construction_date"])
        road_length_km = float(record["road_length_km"])
        traffic_base = int(record["traffic_base"])
        heavy_vehicle_ratio_base = float(record["heavy_vehicle_ratio_base"])
        weather_exposure = float(record["weather_exposure"])
        deterioration_rate = float(record["deterioration_rate"])
        accident_propensity = float(record["accident_propensity"])
        initial_condition = int(record["initial_condition"])
        if road_length_km <= 0:
            raise ValueError("road_length_km must be positive")
        if traffic_base < 0:
            raise ValueError("traffic_base must be non-negative")
        if not 0.0 <= heavy_vehicle_ratio_base <= 1.0:
            raise ValueError("heavy_vehicle_ratio_base must be within 0 and 1")
        if not 1 <= initial_condition <= 100:
            raise ValueError("initial_condition must be within 1 and 100")
        if deterioration_rate <= 0:
            raise ValueError("deterioration_rate must be positive")
        if accident_propensity < 0:
            raise ValueError("accident_propensity must be non-negative")
        if weather_exposure <= 0:
            raise ValueError("weather_exposure must be positive")
        return cls(
            segment_id=str(record["segment_id"]),
            province=str(record["province"]),
            road_type=str(record["road_type"]),
            construction_date=construction_date,
            road_length_km=road_length_km,
            traffic_base=traffic_base,
            heavy_vehicle_ratio_base=heavy_vehicle_ratio_base,
            weather_exposure=weather_exposure,
            deterioration_rate=deterioration_rate,
            accident_propensity=accident_propensity,
            initial_condition=initial_condition,
        )


def generate_segments(
    spec: DatasetSpec,
    seed: int,
    *,
    observation_start: date | None = None,
) -> pd.DataFrame:
    """Generate the deterministic segment master for ``spec``.

    ``seed`` must be a positive integer. Identifiers come from a fixed
    registry and are identical for any seed; all stochastic attributes are
    drawn from ``SeedSequence(seed)`` child stream 0.
    """
    if seed < 1:
        raise ValueError("seed must be a positive integer")
    start = observation_start or V1_OBSERVATION_START
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
    records: list[dict[str, Any]] = []
    for index in range(spec.dataset_segments):
        road_code = ROAD_CODES[index % len(ROAD_CODES)]
        start_km = 10 + (index // len(ROAD_CODES)) * 17
        span_km = 4 + (index % 9)
        end_km = start_km + span_km
        road_length_km = float(span_km)
        construction_date = start - timedelta(
            days=int(rng.integers(_MIN_AGE_YEARS * 365, _MAX_AGE_YEARS * 365 + 1))
        )
        records.append(
            {
                "segment_id": f"{road_code}-KM{start_km}-{end_km}",
                "province": PROVINCES[index % len(PROVINCES)],
                "road_type": ROAD_TYPES[index % len(ROAD_TYPES)],
                "construction_date": construction_date,
                "road_length_km": road_length_km,
                "traffic_base": int(rng.integers(800, 20_000)),
                "heavy_vehicle_ratio_base": round(float(rng.uniform(0.05, 0.50)), 4),
                "weather_exposure": round(float(rng.uniform(0.6, 1.6)), 4),
                "deterioration_rate": round(float(rng.uniform(0.3, 1.8)), 4),
                "accident_propensity": round(float(rng.uniform(0.3, 2.5)), 4),
                "initial_condition": int(rng.integers(60, 101)),
            }
        )
    return pd.DataFrame(records, columns=list(SEGMENT_COLUMNS))


def _as_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


__all__ = [
    "PROVINCES",
    "ROAD_CODES",
    "ROAD_TYPES",
    "SEGMENT_COLUMNS",
    "SEGMENT_ID_PATTERN",
    "SegmentMaster",
    "generate_segments",
]
