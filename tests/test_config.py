"""Tests for central typed configuration loading and validation."""

from __future__ import annotations

import traceback
from pathlib import Path

import pytest
from pydantic import ValidationError

from roadguard import (
    ConfigError,
    DatasetSpec,
    RiskBand,
    RiskBands,
    RoadGuardConfig,
    V1Contract,
    load_config,
)


class TestDefaults:
    def test_runtime_defaults(self) -> None:
        config = load_config()
        assert config.env == "development"
        assert config.seed == 42
        assert config.data_dir == Path("data")
        assert config.artifacts_dir == Path("artifacts")

    def test_v1_contract_is_locked_and_exposed(self) -> None:
        contract = load_config().contract
        assert contract.dataset_segments == 300
        assert contract.dataset_months_per_segment == 48
        assert contract.dataset_observations == 14_400
        assert contract.train_fraction == 0.70
        assert contract.validation_fraction == 0.15
        assert contract.test_fraction == 0.15
        assert contract.maintenance_window_days == 30
        assert contract.train_date_count == 34
        assert contract.validation_date_count == 7
        assert contract.test_date_count == 7

    def test_default_risk_bands_match_locked_contract(self) -> None:
        bands = load_config().contract.risk_bands
        assert (bands.low.lower, bands.low.upper) == (0, 30)
        assert (bands.medium.lower, bands.medium.upper) == (31, 60)
        assert (bands.high.lower, bands.high.upper) == (61, 80)
        assert (bands.critical.lower, bands.critical.upper) == (81, 100)


class TestV1Contract:
    def test_locked_defaults(self) -> None:
        contract = V1Contract()
        assert (contract.dataset_segments, contract.dataset_months_per_segment) == (300, 48)
        assert contract.dataset_observations == 14_400
        assert (contract.train_fraction, contract.validation_fraction, contract.test_fraction) == (
            0.70,
            0.15,
            0.15,
        )
        assert contract.maintenance_window_days == 30
        assert (
            contract.train_date_count,
            contract.validation_date_count,
            contract.test_date_count,
        ) == (
            34,
            7,
            7,
        )

    @pytest.mark.parametrize(
        "override",
        [
            {"dataset_segments": 500},
            {"dataset_months_per_segment": 12},
            {"dataset_observations": 24_000},
            {"train_fraction": 0.8},
            {"validation_fraction": 0.2},
            {"test_fraction": 0.2},
            {"maintenance_window_days": 45},
            {"train_date_count": 33},
            {"validation_date_count": 8},
            {"test_date_count": 8},
        ],
    )
    def test_locked_values_cannot_be_changed(self, override: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            V1Contract(**override)

    def test_date_counts_sum_to_month_count(self) -> None:
        contract = V1Contract()
        total = (
            contract.train_date_count + contract.validation_date_count + contract.test_date_count
        )
        assert total == 48
        assert total == contract.dataset_months_per_segment

    def test_alternate_contiguous_band_layout_rejected(self) -> None:
        alternate = RiskBands(
            low=RiskBand(lower=0, upper=25),
            medium=RiskBand(lower=26, upper=50),
            high=RiskBand(lower=51, upper=75),
            critical=RiskBand(lower=76, upper=100),
        )
        with pytest.raises(ValidationError):
            V1Contract(risk_bands=alternate)

    def test_contract_is_immutable(self) -> None:
        contract = V1Contract()
        with pytest.raises(ValidationError):
            contract.dataset_segments = 1  # type: ignore[misc]


class TestDatasetSpec:
    def test_small_explicit_spec_valid(self) -> None:
        spec = DatasetSpec(
            dataset_segments=10, dataset_months_per_segment=12, dataset_observations=120
        )
        assert spec.dataset_segments == 10
        assert spec.dataset_observations == 120

    @pytest.mark.parametrize(
        "field", ["dataset_segments", "dataset_months_per_segment", "dataset_observations"]
    )
    def test_spec_counts_must_be_positive(self, field: str) -> None:
        kwargs = {
            "dataset_segments": 10,
            "dataset_months_per_segment": 12,
            "dataset_observations": 120,
        }
        kwargs[field] = 0
        with pytest.raises(ValidationError):
            DatasetSpec(**kwargs)

    def test_spec_shape_must_be_consistent(self) -> None:
        with pytest.raises(ValidationError):
            DatasetSpec(
                dataset_segments=10, dataset_months_per_segment=12, dataset_observations=119
            )

    def test_spec_is_immutable(self) -> None:
        spec = DatasetSpec(
            dataset_segments=10, dataset_months_per_segment=12, dataset_observations=120
        )
        with pytest.raises(ValidationError):
            spec.dataset_segments = 1  # type: ignore[misc]


class TestEnvironmentOverrides:
    def test_environment_overrides_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROADGUARD_SEED", "7")
        monkeypatch.setenv("ROADGUARD_ENV", "test")
        config = load_config()
        assert config.seed == 7
        assert config.env == "test"

    def test_numeric_string_env_value_is_coerced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROADGUARD_SEED", "42")
        assert load_config().seed == 42

    def test_path_env_value_is_coerced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROADGUARD_DATA_DIR", "C:\\tmp\\rg-data")
        assert load_config().data_dir == Path("C:\\tmp\\rg-data")

    def test_invalid_environment_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROADGUARD_SEED", "not-a-number")
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            load_config()


class TestUnknownEnvironmentVariables:
    @pytest.mark.parametrize("name", ["ROADGUARD_BOGUS", "ROADGUARD_SED", "ROADGUARD_DATA_DRI"])
    def test_unknown_environment_variable_rejected(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        monkeypatch.setenv(name, "value")
        with pytest.raises(ConfigError, match="Unsupported ROADGUARD_"):
            load_config()

    def test_multiple_unknown_variables_are_all_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ROADGUARD_BOGUS", "1")
        monkeypatch.setenv("ROADGUARD_SED", "2")
        with pytest.raises(ConfigError) as excinfo:
            load_config()
        message = str(excinfo.value)
        assert "ROADGUARD_BOGUS" in message
        assert "ROADGUARD_SED" in message

    def test_unknown_environment_traceback_locals_mask_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "ROADGUARD_DATABASE_URL", "postgresql+psycopg://user:LOCAL_ENV_SECRET@host/db"
        )
        monkeypatch.setenv("ROADGUARD_BOGUS", "value")
        with pytest.raises(ConfigError) as excinfo:
            load_config()
        rendered = "".join(
            traceback.TracebackException.from_exception(excinfo.value, capture_locals=True).format()
        )
        assert "LOCAL_ENV_SECRET" not in rendered

    def test_locked_v1_fields_cannot_be_set_via_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ROADGUARD_DATASET_SEGMENTS", "300")
        monkeypatch.setenv("ROADGUARD_TRAIN_FRACTION", "0.70")
        with pytest.raises(ConfigError):
            load_config()

    def test_config_path_control_variable_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("seed: 55\n", encoding="utf-8")
        monkeypatch.setenv("ROADGUARD_CONFIG_PATH", str(path))
        assert load_config().seed == 55

    def test_unknown_variable_rejected_even_with_valid_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("seed: 55\n", encoding="utf-8")
        monkeypatch.setenv("ROADGUARD_CONFIG_PATH", str(path))
        monkeypatch.setenv("ROADGUARD_SED", "2")
        with pytest.raises(ConfigError):
            load_config()


class TestFileLoading:
    def test_yaml_file_loaded_from_argument(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "seed: 123\nenv: production\ndata_dir: C:\\tmp\\rg-data\n",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.seed == 123
        assert config.env == "production"
        assert config.data_dir == Path("C:\\tmp\\rg-data")

    def test_environment_wins_over_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("seed: 123\n", encoding="utf-8")
        monkeypatch.setenv("ROADGUARD_SEED", "9")
        assert load_config(path).seed == 9

    def test_empty_yaml_file_yields_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config(path) == RoadGuardConfig()

    def test_missing_config_file_raises_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            load_config(tmp_path / "missing.yaml")

    def test_malformed_yaml_raises_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("seed: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_non_mapping_yaml_raises_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_unknown_yaml_key_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("bogus: true\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            load_config(path)

    def test_invalid_database_url_load_error_masks_credential(self, tmp_path: Path) -> None:
        sentinel = "PHASE6_DATABASE_SECRET"
        path = tmp_path / "config.yaml"
        path.write_text(
            f"database_url:\n  - postgresql+psycopg://user:{sentinel}@host/db\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as excinfo:
            load_config(path)
        assert sentinel not in str(excinfo.value)
        assert sentinel not in repr(excinfo.value)
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__context__ is None

    def test_malformed_yaml_error_masks_credential(self, tmp_path: Path) -> None:
        sentinel = "PHASE6_YAML_SECRET"
        path = tmp_path / "config.yaml"
        path.write_text(
            f'database_url: "postgresql+psycopg://user:{sentinel}@host/db\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as excinfo:
            load_config(path)
        assert sentinel not in str(excinfo.value)
        assert sentinel not in repr(excinfo.value)
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__context__ is None

    def test_validation_error_traceback_locals_mask_credential(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "database_url:\n  - postgresql+psycopg://user:LOCAL_CAPTURE_SECRET@host/db\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as excinfo:
            load_config(path)
        rendered = "".join(
            traceback.TracebackException.from_exception(excinfo.value, capture_locals=True).format()
        )
        assert "LOCAL_CAPTURE_SECRET" not in rendered

    def test_non_mapping_yaml_traceback_locals_mask_credential(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "- postgresql+psycopg://user:LOCAL_RAW_SECRET@host/db\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as excinfo:
            load_config(path)
        rendered = "".join(
            traceback.TracebackException.from_exception(excinfo.value, capture_locals=True).format()
        )
        assert "LOCAL_RAW_SECRET" not in rendered

    @pytest.mark.parametrize(
        "yaml_text",
        [
            "dataset_segments: 500\n",
            "maintenance_window_days: 45\n",
            "train_fraction: 0.8\n",
        ],
    )
    def test_locked_v1_fields_cannot_be_set_via_yaml(self, tmp_path: Path, yaml_text: str) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            load_config(path)

    def test_yaml_boolean_seed_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("seed: true\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            load_config(path)

    def test_yaml_numeric_string_seed_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text('seed: "42"\n', encoding="utf-8")
        assert load_config(path).seed == 42


class TestValidation:
    @pytest.mark.parametrize("seed", [0, -1, -42])
    def test_non_positive_seed_rejected(self, seed: int) -> None:
        with pytest.raises(ValidationError):
            RoadGuardConfig(seed=seed)

    def test_non_positive_seed_rejected_on_load_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROADGUARD_SEED", "-3")
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            load_config()

    def test_invalid_env_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RoadGuardConfig(env="staging")

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RoadGuardConfig.model_validate({"seed": 42, "bogus": True})

    def test_contract_key_rejected_as_direct_field(self) -> None:
        with pytest.raises(ValidationError):
            RoadGuardConfig.model_validate({"seed": 42, "contract": {"dataset_segments": 500}})


class TestBooleanRejection:
    def test_boolean_seed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RoadGuardConfig(seed=True)

    def test_boolean_dataset_counts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            V1Contract(dataset_segments=True)

    def test_boolean_window_rejected(self) -> None:
        with pytest.raises(ValidationError):
            V1Contract(maintenance_window_days=True)

    def test_boolean_fraction_rejected(self) -> None:
        with pytest.raises(ValidationError):
            V1Contract(train_fraction=True)

    def test_boolean_date_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            V1Contract(train_date_count=True)

    def test_boolean_band_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RiskBand(lower=False, upper=True)

    def test_boolean_spec_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetSpec(
                dataset_segments=True,
                dataset_months_per_segment=12,
                dataset_observations=120,
            )


class TestRiskBands:
    @pytest.mark.parametrize(
        ("lower", "upper"),
        [(-1, 30), (0, 101), (50, 10)],
    )
    def test_band_bounds_rejected(self, lower: int, upper: int) -> None:
        with pytest.raises(ValidationError):
            RiskBand(lower=lower, upper=upper)

    def test_band_numeric_strings_accepted(self) -> None:
        band = RiskBand(lower="10", upper="30")
        assert band.lower == 10
        assert band.upper == 30

    def test_gap_between_bands_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RiskBands(low=RiskBand(lower=0, upper=25))

    def test_bands_must_cover_0_to_100(self) -> None:
        with pytest.raises(ValidationError):
            RiskBands(critical=RiskBand(lower=81, upper=99))

    def test_low_band_must_start_at_zero(self) -> None:
        with pytest.raises(ValidationError):
            RiskBands(low=RiskBand(lower=1, upper=30))


class TestImmutability:
    def test_config_is_immutable(self) -> None:
        config = RoadGuardConfig()
        with pytest.raises(ValidationError):
            config.seed = 9  # type: ignore[misc]

    def test_band_is_immutable(self) -> None:
        band = RiskBand(lower=0, upper=30)
        with pytest.raises(ValidationError):
            band.upper = 40  # type: ignore[misc]
