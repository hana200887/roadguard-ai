"""Canonical PostgreSQL-reflected definitions for Phase 6 drift detection."""

from __future__ import annotations

import re
from typing import Final

_CAST_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"::(?:double\s+precision|integer|text)",
    re.IGNORECASE,
)

CHECK_DEFINITIONS: Final[dict[str, dict[str, str]]] = {
    "material_forecasts": {
        "ck_material_forecasts_forecast_quantity": (
            "forecast_quantity>=0andforecast_quantity<'Infinity'"
        ),
        "ck_material_forecasts_material": (
            "material=any(array['thermoplastic_paint_kg','reflective_sheet_m2',"
            "'guardrail_meter','traffic_sign_quantity'])"
        ),
    },
    "road_segments": {
        "ck_road_segments_province": "province=any(array['NA','TH','QB','DN','GL','LA'])",
        "ck_road_segments_road_length": ("road_length_km>0androad_length_km<'Infinity'"),
        "ck_road_segments_road_type": (
            "road_type=any(array['highway','national','provincial','urban','rural'])"
        ),
        "ck_road_segments_segment_id_format": ("segment_id~'^QL[0-9]{2}-KM[0-9]+-[0-9]+$'"),
    },
    "maintenance_events": {},
    "road_observations": {
        "ck_road_observations_accident_counts": (
            "accident_count_30d>=0andaccident_count_365d>=accident_count_30d"
        ),
        "ck_road_observations_guardrail_score": (
            "guardrail_condition_score>=1andguardrail_condition_score<=100"
        ),
        "ck_road_observations_heavy_vehicle_ratio": (
            "heavy_vehicle_ratio>=0andheavy_vehicle_ratio<=1"
        ),
        "ck_road_observations_humidity": "humidity>=0andhumidity<=100",
        "ck_road_observations_maintenance_age": (
            "days_since_last_maintenance>=0anddays_since_last_maintenance<=road_age_days"
        ),
        "ck_road_observations_marking_score": (
            "marking_condition_score>=1andmarking_condition_score<=100"
        ),
        "ck_road_observations_previous_repairs": "previous_repairs>=0",
        "ck_road_observations_rainfall_mm": ("rainfall_mm>=0andrainfall_mm<'Infinity'"),
        "ck_road_observations_road_age_days": "road_age_days>=0",
        "ck_road_observations_road_score": ("road_condition_score>=1androad_condition_score<=100"),
        "ck_road_observations_sign_score": ("sign_condition_score>=1andsign_condition_score<=100"),
        "ck_road_observations_temperature": "temperature>='-50'andtemperature<=60",
        "ck_road_observations_traffic_volume": "traffic_volume>=0",
    },
    "maintenance_history": {
        "ck_maintenance_history_guardrail_quantity": (
            "guardrail_meter>=0andguardrail_meter<'Infinity'"
        ),
        "ck_maintenance_history_maintenance_cost": "maintenance_cost>0",
        "ck_maintenance_history_paint_quantity": (
            "thermoplastic_paint_kg>=0andthermoplastic_paint_kg<'Infinity'"
        ),
        "ck_maintenance_history_sheet_quantity": (
            "reflective_sheet_m2>=0andreflective_sheet_m2<'Infinity'"
        ),
        "ck_maintenance_history_sign_quantity": "traffic_sign_quantity>=0",
    },
    "observation_targets": {
        "ck_observation_targets_days_until_maintenance": "days_until_maintenance>=0",
        "ck_observation_targets_maintenance_within_30_days": (
            "maintenance_within_30_days=any(array[0,1])"
        ),
        "ck_observation_targets_target_consistency": (
            "maintenance_within_30_days=casewhendays_until_maintenance<=30then1else0end"
        ),
    },
    "predictions": {
        "ck_predictions_maintenance_probability": (
            "maintenance_probability>=0andmaintenance_probability<=1"
        ),
        "ck_predictions_risk_band": (
            "risk_band='LOW'andrisk_score>=0andrisk_score<=30or"
            "risk_band='MEDIUM'andrisk_score>=31andrisk_score<=60or"
            "risk_band='HIGH'andrisk_score>=61andrisk_score<=80or"
            "risk_band='CRITICAL'andrisk_score>=81andrisk_score<=100"
        ),
        "ck_predictions_risk_score": "risk_score>=0andrisk_score<=100",
    },
}

IndexDefinition = tuple[str, bool, tuple[str, ...], tuple[str, ...], str | None]
INDEX_DEFINITIONS: Final[dict[str, set[IndexDefinition]]] = {
    "material_forecasts": {
        ("ix_material_forecasts_period", False, ("period",), (), None),
    },
    "road_segments": set(),
    "maintenance_events": {
        ("ix_maintenance_events_date", False, ("maintenance_date",), (), None),
        (
            "uq_maintenance_events_segment_month",
            True,
            (
                "segment_id",
                "extract(yearfrommaintenance_date)",
                "extract(monthfrommaintenance_date)",
            ),
            (),
            None,
        ),
    },
    "road_observations": {
        ("ix_road_observations_date", False, ("date",), (), None),
    },
    "maintenance_history": {
        ("ix_maintenance_history_date", False, ("maintenance_date",), (), None),
    },
    "observation_targets": set(),
    "predictions": {("ix_predictions_date", False, ("date",), (), None)},
}


def normalize_definition(value: str) -> str:
    """Normalize stable PostgreSQL reflection formatting without changing semantics."""
    parts = re.split(r"('(?:''|[^'])*')", value)
    normalized: list[str] = []
    for position, part in enumerate(parts):
        if position % 2:
            normalized.append(part)
            continue
        without_casts = _CAST_PATTERN.sub("", part)
        normalized.append("".join(without_casts.replace('"', "").lower().split()))
    return "".join(normalized)


def normalize_optional_definition(value: object) -> str | None:
    """Normalize an optional inspector expression."""
    if value is None:
        return None
    return normalize_definition(str(value))


__all__ = [
    "CHECK_DEFINITIONS",
    "INDEX_DEFINITIONS",
    "normalize_definition",
    "normalize_optional_definition",
]
