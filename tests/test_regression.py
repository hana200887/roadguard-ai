"""Phase 12 advanced-regression contract tests.

The authority is docs/contracts.md section 19.  The generated and crafted
exports reuse already-reviewed Phase 11 test helpers so this module can focus
on the regression-specific public API, metrics, selection, temporal boundary,
configuration isolation, and immutable output contract.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import SecretStr
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

import roadguard
from roadguard import RepositoryExport, RoadGuardConfig
from roadguard.features import FEATURE_KEY_COLUMNS, build_feature_frame
from roadguard.preprocessing import fit_preprocessor, split_chronologically, transform
from roadguard.regression import (
    ADVANCED_REGRESSOR_CONTRACT_VERSION,
    ADVANCED_REGRESSOR_RNG_NAMESPACE,
    CANDIDATE_REGRESSOR_NAMES,
    AdvancedRegressionError,
    AdvancedRegressionEvaluation,
    CandidateRegressionValidationMetrics,
    TestRegressionMetrics,
    _select_candidate_index,
    evaluate_advanced_regressor,
)
from test_classification import (
    CRAFTED_SPEC,
    V1_SPEC,
    _canonical_fit,
    _canonical_split,
    _export_c,
    _generated_export,
)

PUBLIC_NAMES = (
    "evaluate_advanced_regressor",
    "AdvancedRegressionError",
    "CandidateRegressionValidationMetrics",
    "TestRegressionMetrics",
    "AdvancedRegressionEvaluation",
    "ADVANCED_REGRESSOR_CONTRACT_VERSION",
    "ADVANCED_REGRESSOR_RNG_NAMESPACE",
    "CANDIDATE_REGRESSOR_NAMES",
)
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "segment_id",
        "date",
        "days_until_maintenance",
        "maintenance_within_30_days",
        "maintenance_date",
        "maintenance_history",
        "maintenance_cost",
        "thermoplastic_paint_kg",
        "reflective_sheet_m2",
        "guardrail_meter",
        "traffic_sign_quantity",
        "traffic_base",
        "heavy_vehicle_ratio_base",
        "weather_exposure",
        "deterioration_rate",
        "accident_propensity",
        "initial_condition",
    }
)


@pytest.fixture(scope="module")
def dataset_c() -> RepositoryExport:
    return _export_c()


@pytest.fixture(scope="module")
def split_c(dataset_c: RepositoryExport) -> Any:
    return _canonical_split(dataset_c, CRAFTED_SPEC)


@pytest.fixture(scope="module")
def fit_c(split_c: Any) -> Any:
    return _canonical_fit(split_c, CRAFTED_SPEC)


@pytest.fixture(scope="module")
def result_c(dataset_c: RepositoryExport, split_c: Any, fit_c: Any) -> Any:
    return evaluate_advanced_regressor(
        dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
    )


def _reference(dataset: RepositoryExport, seed: int = 42) -> dict[str, Any]:
    frame = build_feature_frame(dataset, CRAFTED_SPEC)
    split = split_chronologically(frame, CRAFTED_SPEC)
    fit = fit_preprocessor(split, CRAFTED_SPEC)
    x: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        transformed = transform(getattr(split, name), fit)
        x[name] = transformed.features.to_numpy(dtype="float64")
        joined = transformed.keys.merge(
            dataset.targets[["segment_id", "date", "days_until_maintenance"]],
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )
        y[name] = joined["days_until_maintenance"].to_numpy(dtype="int64")
    derived = int(
        np.random.SeedSequence([seed, ADVANCED_REGRESSOR_RNG_NAMESPACE, 1]).generate_state(
            1, dtype=np.uint32
        )[0]
    )
    models = (
        Ridge(alpha=1.0, fit_intercept=True, solver="svd", tol=1e-8, positive=False),
        HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=derived,
        ),
    )
    records: list[tuple[float, float]] = []
    for model in models:
        model.fit(x["train"], y["train"])
        prediction = model.predict(x["validation"])
        records.append(
            (
                float(mean_absolute_error(y["validation"], prediction)),
                float(root_mean_squared_error(y["validation"], prediction)),
            )
        )
    selected = min(range(2), key=lambda index: (*records[index], index))
    test_prediction = models[selected].predict(x["test"])
    return {
        "records": records,
        "selected": selected,
        "test": (
            float(mean_absolute_error(y["test"], test_prediction)),
            float(root_mean_squared_error(y["test"], test_prediction)),
            float(r2_score(y["test"], test_prediction, force_finite=False)),
        ),
    }


class TestPublicSurface:
    def test_exact_module_surface_and_signature(self) -> None:
        import roadguard.regression as regression

        assert tuple(regression.__all__) == PUBLIC_NAMES
        assert tuple(inspect.signature(evaluate_advanced_regressor).parameters) == (
            "dataset",
            "split",
            "fit",
            "spec",
            "config",
        )
        assert not hasattr(regression, "evaluate_test")

    def test_constants_are_exact(self) -> None:
        assert ADVANCED_REGRESSOR_CONTRACT_VERSION == "roadguard.phase12.v1"
        assert ADVANCED_REGRESSOR_RNG_NAMESPACE == 0x5247312
        assert CANDIDATE_REGRESSOR_NAMES == ("ridge_l2", "hist_gradient_boosting")
        for name in PUBLIC_NAMES:
            assert getattr(roadguard, name) is not None


class TestFrozenSchema:
    def test_exact_fields_and_frozen_instances(self) -> None:
        assert tuple(
            field.name for field in dataclasses.fields(CandidateRegressionValidationMetrics)
        ) == (
            "regressor_name",
            "validation_mae",
            "validation_rmse",
        )
        assert tuple(field.name for field in dataclasses.fields(TestRegressionMetrics)) == (
            "mae",
            "rmse",
            "r2",
        )
        assert tuple(field.name for field in dataclasses.fields(AdvancedRegressionEvaluation)) == (
            "contract_version",
            "selected_regressor_name",
            "feature_columns",
            "train_rows",
            "validation_rows",
            "test_rows",
            "candidates",
            "test",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            CandidateRegressionValidationMetrics("x", 1.0, 1.0).validation_mae = 2.0

    def test_result_contains_only_frozen_scalar_provenance(
        self, result_c: AdvancedRegressionEvaluation
    ) -> None:
        assert result_c.contract_version == ADVANCED_REGRESSOR_CONTRACT_VERSION
        assert (
            tuple(record.regressor_name for record in result_c.candidates)
            == CANDIDATE_REGRESSOR_NAMES
        )
        assert result_c.selected_regressor_name in CANDIDATE_REGRESSOR_NAMES
        assert not any(
            forbidden in {field.name for field in dataclasses.fields(result_c)}
            for forbidden in ("model", "seed", "config", "prediction", "coefficient", "path")
        )


class TestReferenceAndSelection:
    def test_matches_independent_sklearn_reference(
        self, dataset_c: RepositoryExport, result_c: AdvancedRegressionEvaluation
    ) -> None:
        reference = _reference(dataset_c)
        for record, expected in zip(result_c.candidates, reference["records"], strict=True):
            assert record.validation_mae == pytest.approx(expected[0])
            assert record.validation_rmse == pytest.approx(expected[1])
        assert result_c.selected_regressor_name == CANDIDATE_REGRESSOR_NAMES[reference["selected"]]
        assert result_c.test.mae == pytest.approx(reference["test"][0])
        assert result_c.test.rmse == pytest.approx(reference["test"][1])
        assert result_c.test.r2 == pytest.approx(reference["test"][2])

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ((1.0, 9.0), (2.0, 1.0), 0),
            ((1.0, 2.0), (1.0, 3.0), 0),
            ((1.0, 2.0), (1.0, 2.0), 0),
            ((2.0, 1.0), (1.0, 9.0), 1),
        ],
    )
    def test_candidate_ranking(
        self, left: tuple[float, float], right: tuple[float, float], expected: int
    ) -> None:
        records = (
            CandidateRegressionValidationMetrics("left", *left),
            CandidateRegressionValidationMetrics("right", *right),
        )
        assert _select_candidate_index(records) == expected


class TestTemporalAndFeatureBoundary:
    def test_test_feature_mutation_changes_only_test_metrics(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        baseline = evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        observations = dataset_c.observations.copy(deep=True)
        mask = observations["date"].isin(pd.to_datetime(split_c.test_dates))
        observations.loc[mask, "rainfall_mm"] *= 2.0
        mutated = replace(dataset_c, observations=observations)
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)
        changed = evaluate_advanced_regressor(
            mutated, mutated_split, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        assert changed.candidates == baseline.candidates
        assert changed.selected_regressor_name == baseline.selected_regressor_name
        assert changed.feature_columns == baseline.feature_columns
        assert (changed.train_rows, changed.validation_rows, changed.test_rows) == (
            baseline.train_rows,
            baseline.validation_rows,
            baseline.test_rows,
        )

    def test_validation_feature_mutation_does_not_change_train_fit(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        observations = dataset_c.observations.copy(deep=True)
        mask = observations["date"].isin(pd.to_datetime(split_c.validation_dates))
        observations.loc[mask, "temperature"] += 1.0
        mutated = replace(dataset_c, observations=observations)
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)
        assert fit_preprocessor(mutated_split, CRAFTED_SPEC) == fit_c
        assert_frame_equal(
            transform(mutated_split.train, fit_c).features,
            transform(split_c.train, fit_c).features,
        )

    def test_feature_columns_match_fit_and_exclude_forbidden(
        self, result_c: AdvancedRegressionEvaluation, fit_c: Any
    ) -> None:
        assert result_c.feature_columns == fit_c.transformed_feature_columns
        assert FORBIDDEN_FEATURE_NAMES.isdisjoint(result_c.feature_columns)

    def test_repeated_and_shuffled_inputs_are_deterministic(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        baseline = evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        assert (
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
            == baseline
        )
        shuffled = replace(
            dataset_c,
            observations=dataset_c.observations.sample(frac=1.0, random_state=7).reset_index(
                drop=True
            ),
            targets=dataset_c.targets.sample(frac=1.0, random_state=11).reset_index(drop=True),
            maintenance_events=dataset_c.maintenance_events.sample(
                frac=1.0, random_state=13
            ).reset_index(drop=True),
        )
        shuffled_split = _canonical_split(shuffled, CRAFTED_SPEC)
        assert (
            evaluate_advanced_regressor(
                shuffled, shuffled_split, fit_c, CRAFTED_SPEC, RoadGuardConfig()
            )
            == baseline
        )


class TestInputAndConfigBoundary:
    def test_wrong_top_level_types_fail_before_field_access(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        valid = (dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        for index, label in enumerate(("dataset", "split", "fit", "spec", "config")):
            arguments = list(valid)
            arguments[index] = SimpleNamespace()
            with pytest.raises(TypeError, match=label):
                evaluate_advanced_regressor(*arguments)

    def test_invalid_and_missing_config_seed_rejected(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        class IntSubclass(int):
            pass

        for seed in (False, 0, -1, "42", np.int64(42), IntSubclass(42), None):
            config = RoadGuardConfig.model_construct(seed=seed)
            with pytest.raises(AdvancedRegressionError, match="seed"):
                evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, config)
        missing = RoadGuardConfig.model_construct()
        object.__setattr__(missing, "__dict__", {})
        with pytest.raises(AdvancedRegressionError, match="seed"):
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, missing)

    def test_only_seed_is_read_from_config(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        seed_only = RoadGuardConfig.model_construct(seed=42)
        object.__setattr__(seed_only, "__dict__", {"seed": 42})
        baseline = evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, seed_only)
        altered = RoadGuardConfig(
            seed=42,
            env="production",
            data_dir=Path("ignored"),
            artifacts_dir=Path("ignored-too"),
            database_url=SecretStr("ignored-database-url"),
        )
        assert (
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, altered)
            == baseline
        )

    def test_caller_owned_frames_unchanged(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        before = tuple(
            frame.copy(deep=True)
            for frame in (
                dataset_c.segments,
                dataset_c.observations,
                dataset_c.targets,
                dataset_c.maintenance_events,
            )
        )
        evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        for actual, expected in zip(
            (
                dataset_c.segments,
                dataset_c.observations,
                dataset_c.targets,
                dataset_c.maintenance_events,
            ),
            before,
            strict=True,
        ):
            assert_frame_equal(actual, expected)

    def test_forged_split_fit_and_target_rejected(
        self, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        with pytest.raises(AdvancedRegressionError):
            evaluate_advanced_regressor(
                dataset_c,
                replace(split_c, validation=split_c.test),
                fit_c,
                CRAFTED_SPEC,
                RoadGuardConfig(),
            )
        with pytest.raises(AdvancedRegressionError):
            evaluate_advanced_regressor(
                dataset_c,
                split_c,
                replace(fit_c, means=tuple(value + 1.0 for value in fit_c.means)),
                CRAFTED_SPEC,
                RoadGuardConfig(),
            )
        forged_targets = dataset_c.targets.copy(deep=True)
        forged_targets.loc[0, "days_until_maintenance"] = 999
        with pytest.raises(AdvancedRegressionError):
            evaluate_advanced_regressor(
                replace(dataset_c, targets=forged_targets),
                split_c,
                fit_c,
                CRAFTED_SPEC,
                RoadGuardConfig(),
            )


class TestMetricSemantics:
    def test_negative_finite_r2_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch, dataset_c: RepositoryExport, split_c: Any, fit_c: Any
    ) -> None:
        import roadguard.regression as regression

        monkeypatch.setattr(regression, "r2_score", lambda *args, **kwargs: -3.5)
        result = evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        assert result.test.r2 == -3.5


class TestV1Profile:
    def test_v1_complete_evaluation_is_reproducible(self) -> None:
        dataset = _generated_export(V1_SPEC)
        split = _canonical_split(dataset, V1_SPEC)
        fit = _canonical_fit(split, V1_SPEC)
        result = evaluate_advanced_regressor(dataset, split, fit, V1_SPEC, RoadGuardConfig())
        assert (result.train_rows, result.validation_rows, result.test_rows) == (
            10_200,
            2_100,
            2_100,
        )
        assert (
            tuple(record.regressor_name for record in result.candidates)
            == CANDIDATE_REGRESSOR_NAMES
        )
        for record in result.candidates:
            assert math.isfinite(record.validation_mae) and record.validation_mae >= 0.0
            assert math.isfinite(record.validation_rmse) and record.validation_rmse >= 0.0
        assert math.isfinite(result.test.mae) and result.test.mae >= 0.0
        assert math.isfinite(result.test.rmse) and result.test.rmse >= 0.0
        assert math.isfinite(result.test.r2)
        assert (
            evaluate_advanced_regressor(dataset, split, fit, V1_SPEC, RoadGuardConfig()) == result
        )
