"""Central typed configuration for RoadGuard AI.

Configuration is resolved with the following precedence (later wins):

1. Built-in defaults defined on :class:`RoadGuardConfig`.
2. An optional YAML mapping supplied as an explicit path or via the
   ``ROADGUARD_CONFIG_PATH`` environment variable.
3. Environment variables named ``ROADGUARD_<FIELD>``.

Only runtime settings (environment, seed, data directories and an optional
credential-bearing PostgreSQL URL) are tunable.
The V1 data and ML contract is locked by :class:`roadguard.contracts.V1Contract`
and cannot be changed through YAML or environment variables: keys or
variables that attempt it are rejected. Unknown ``ROADGUARD_*`` environment
variables fail fast with :class:`ConfigError`; ``ROADGUARD_CONFIG_PATH`` is
the reserved control variable for locating the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from roadguard.contracts import NonBooleanInt, V1Contract

ENV_PREFIX: Final[str] = "ROADGUARD_"
CONFIG_PATH_ENV: Final[str] = f"{ENV_PREFIX}CONFIG_PATH"


class ConfigError(Exception):
    """Raised when configuration cannot be read or parsed."""


class RoadGuardConfig(BaseModel):
    """Validated runtime configuration.

    The locked V1 contract is exposed through the ``contract`` property and
    is not configurable here; only runtime settings can be tuned. Database
    credentials are held in :class:`pydantic.SecretStr` and masked in reprs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    env: Literal["development", "test", "production"] = "development"
    seed: NonBooleanInt = 42
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    database_url: SecretStr | None = None

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        if value < 1:
            raise ValueError("seed must be a positive integer")
        return value

    @property
    def contract(self) -> V1Contract:
        """Return the locked V1 data and ML contract."""
        return V1Contract()


def load_config(config_path: Path | str | None = None) -> RoadGuardConfig:
    """Load and validate configuration.

    Precedence (later wins): built-in defaults, YAML file, environment
    variables. The YAML file is taken from ``config_path`` when given,
    otherwise from the ``ROADGUARD_CONFIG_PATH`` environment variable.
    Unsupported ``ROADGUARD_*`` environment variables raise
    :class:`ConfigError`.
    """
    values: dict[str, Any] = _load_file_values(config_path)
    values.update(_load_env_values())
    return RoadGuardConfig.model_validate(values)


def _load_file_values(config_path: Path | str | None) -> dict[str, Any]:
    resolved = config_path if config_path is not None else os.getenv(CONFIG_PATH_ENV)
    if resolved is None:
        return {}
    path = Path(resolved)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Failed to read configuration file {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Configuration file {path} must contain a mapping, got {type(raw).__name__}"
        )
    return {str(key): value for key, value in raw.items()}


def _load_env_values() -> dict[str, Any]:
    values: dict[str, Any] = {}
    unknown: list[str] = []
    for env_name, value in os.environ.items():
        if not env_name.startswith(ENV_PREFIX):
            continue
        if env_name == CONFIG_PATH_ENV:
            continue
        field_name = env_name[len(ENV_PREFIX) :].lower()
        if field_name in RoadGuardConfig.model_fields:
            values[field_name] = value
        else:
            unknown.append(env_name)
    if unknown:
        raise ConfigError(
            "Unsupported ROADGUARD_* environment variable(s): " + ", ".join(sorted(unknown))
        )
    return values


__all__ = [
    "CONFIG_PATH_ENV",
    "ConfigError",
    "ENV_PREFIX",
    "RoadGuardConfig",
    "load_config",
]
