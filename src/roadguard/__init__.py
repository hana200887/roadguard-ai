"""RoadGuard AI: predictive maintenance and risk intelligence for road infrastructure."""

from __future__ import annotations

__version__ = "0.1.0"

from roadguard.config import ENV_PREFIX, ConfigError, RoadGuardConfig, load_config
from roadguard.contracts import (
    DatasetSpec,
    RiskBand,
    RiskBands,
    V1Contract,
)
from roadguard.events import (
    GenerationError,
    decay_condition,
    generate_accident_timeline,
    generate_maintenance_events,
    month_transition,
    monthly_hazard,
    observation_dates,
)
from roadguard.observations import generate_observations
from roadguard.risk import risk_score_from_probability
from roadguard.segments import SegmentMaster, generate_segments
from roadguard.targets import (
    TARGET_COLUMNS,
    days_until_maintenance,
    derive_observation_targets,
    maintenance_within_30_days,
)

__all__ = [
    "ConfigError",
    "ENV_PREFIX",
    "DatasetSpec",
    "GenerationError",
    "RiskBand",
    "RiskBands",
    "RoadGuardConfig",
    "SegmentMaster",
    "TARGET_COLUMNS",
    "V1Contract",
    "__version__",
    "days_until_maintenance",
    "decay_condition",
    "derive_observation_targets",
    "generate_accident_timeline",
    "generate_maintenance_events",
    "generate_observations",
    "generate_segments",
    "load_config",
    "maintenance_within_30_days",
    "month_transition",
    "monthly_hazard",
    "observation_dates",
    "risk_score_from_probability",
]
