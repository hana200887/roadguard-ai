"""Adversarial Phase 12 estimator, temporal, prediction and metric tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from test_classification import CRAFTED_SPEC, _canonical_fit, _canonical_split, _export_c

import roadguard.regression as regression
from roadguard import RepositoryExport, RoadGuardConfig
from roadguard.classification import AdvancedClassificationError
from roadguard.features import FEATURE_KEY_COLUMNS
from roadguard.preprocessing import transform
from roadguard.regression import (
    ADVANCED_REGRESSOR_RNG_NAMESPACE,
    AdvancedRegressionError,
    evaluate_advanced_regressor,
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


def _install_prediction_stub(
    monkeypatch: pytest.MonkeyPatch, prediction: Any, *, target: str = "ridge"
) -> None:
    class Stub:
        def fit(self, x: Any, y: Any) -> Stub:
            return self

        def predict(self, x: Any) -> Any:
            return prediction

    if target == "ridge":
        monkeypatch.setattr(regression, "Ridge", lambda **kwargs: Stub())
    else:
        monkeypatch.setattr(regression, "HistGradientBoostingRegressor", lambda **kwargs: Stub())


class TestConstructorsAndSeed:
    def test_exact_constructor_parameters_and_one_hgb_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        captured: dict[str, dict[str, object]] = {}
        real_seed_sequence = np.random.SeedSequence
        seed_calls: list[object] = []

        def ridge_spy(**kwargs: object) -> Any:
            captured["ridge"] = kwargs
            return Ridge(**kwargs)

        def hgb_spy(**kwargs: object) -> Any:
            captured["hgb"] = kwargs
            return HistGradientBoostingRegressor(**kwargs)

        def seed_spy(entropy: object) -> Any:
            seed_calls.append(entropy)
            return real_seed_sequence(entropy)

        monkeypatch.setattr(regression, "Ridge", ridge_spy)
        monkeypatch.setattr(regression, "HistGradientBoostingRegressor", hgb_spy)
        monkeypatch.setattr(regression.np.random, "SeedSequence", seed_spy)
        evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
        )
        assert captured["ridge"] == {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "svd",
            "tol": 1e-8,
            "positive": False,
        }
        expected_seed = int(
            real_seed_sequence([42, ADVANCED_REGRESSOR_RNG_NAMESPACE, 1]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        assert captured["hgb"] == {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": expected_seed,
        }
        assert seed_calls == [[42, ADVANCED_REGRESSOR_RNG_NAMESPACE, 1]]

    def test_seed_derivation_failure_is_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        monkeypatch.setattr(
            regression.np.random,
            "SeedSequence",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("secret seed")),
        )
        with pytest.raises(AdvancedRegressionError, match="seed derivation") as excinfo:
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "secret" not in str(excinfo.value)
        assert excinfo.value.__cause__ is None


class TestFitPredictAndTransformBoundary:
    def test_each_candidate_fits_train_predicts_validation_and_only_winner_gets_test(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        calls: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []

        def factory(kind: str, constructor: Any) -> Any:
            def build(**kwargs: object) -> Any:
                delegate = constructor(**kwargs)

                class Recording:
                    def fit(self, x: Any, y: Any) -> Recording:
                        calls.append((kind, "fit", np.asarray(x).copy(), np.asarray(y).copy()))
                        delegate.fit(x, y)
                        return self

                    def predict(self, x: Any) -> Any:
                        calls.append((kind, "predict", np.asarray(x).copy(), None))
                        return delegate.predict(x)

                return Recording()

            return build

        monkeypatch.setattr(regression, "Ridge", factory("ridge", Ridge))
        monkeypatch.setattr(
            regression,
            "HistGradientBoostingRegressor",
            factory("hgb", HistGradientBoostingRegressor),
        )
        result = evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        fit_calls = [call for call in calls if call[1] == "fit"]
        predict_calls = [call for call in calls if call[1] == "predict"]
        validation_x = transform(split_c.validation, fit_c).features.to_numpy(dtype="float64")
        test_x = transform(split_c.test, fit_c).features.to_numpy(dtype="float64")
        assert len(fit_calls) == 2
        train_transformed = transform(split_c.train, fit_c)
        expected_x_train = train_transformed.features.to_numpy(dtype="float64")
        expected_y_train = train_transformed.keys.merge(
            dataset_c.targets[[*FEATURE_KEY_COLUMNS, "days_until_maintenance"]],
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )["days_until_maintenance"].to_numpy(dtype="int64")
        assert all(np.array_equal(call[2], expected_x_train) for call in fit_calls)
        assert all(np.array_equal(call[3], expected_y_train) for call in fit_calls)
        assert len(predict_calls) == 3
        validation_calls = [call for call in predict_calls if np.array_equal(call[2], validation_x)]
        test_calls = [call for call in predict_calls if np.array_equal(call[2], test_x)]
        assert len(validation_calls) == 2
        assert len(test_calls) == 1
        expected_winner = "ridge" if result.selected_regressor_name == "ridge_l2" else "hgb"
        assert test_calls[0][0] == expected_winner
        assert [(call[0], call[1]) for call in calls] == [
            ("ridge", "fit"),
            ("ridge", "predict"),
            ("hgb", "fit"),
            ("hgb", "predict"),
            (expected_winner, "predict"),
        ]

    def test_changed_seed_changes_only_hgb_constructor_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        ridge_calls: list[dict[str, object]] = []
        hgb_calls: list[dict[str, object]] = []

        def ridge_spy(**kwargs: object) -> Any:
            ridge_calls.append(kwargs)
            return Ridge(**kwargs)

        def hgb_spy(**kwargs: object) -> Any:
            hgb_calls.append(kwargs)
            return HistGradientBoostingRegressor(**kwargs)

        monkeypatch.setattr(regression, "Ridge", ridge_spy)
        monkeypatch.setattr(regression, "HistGradientBoostingRegressor", hgb_spy)
        for seed in (42, 43):
            evaluate_advanced_regressor(
                dataset_c,
                split_c,
                fit_c,
                CRAFTED_SPEC,
                RoadGuardConfig(seed=seed),
            )
        assert ridge_calls[0] == ridge_calls[1]
        assert {key: value for key, value in hgb_calls[0].items() if key != "random_state"} == {
            key: value for key, value in hgb_calls[1].items() if key != "random_state"
        }
        assert hgb_calls[0]["random_state"] != hgb_calls[1]["random_state"]

    def test_test_transform_occurs_once_after_two_validation_predictions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        real_transform = regression.transform
        events: list[str] = []

        def transform_spy(frame: Any, fit: Any) -> Any:
            if frame is split_c.test or frame.reset_index(drop=True).equals(
                split_c.test.reset_index(drop=True)
            ):
                events.append("test-transform")
            return real_transform(frame, fit)

        monkeypatch.setattr(regression, "transform", transform_spy)
        evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert events == ["test-transform"]


class TestPredictionValidation:
    @pytest.mark.parametrize(
        "prediction",
        [
            [1.0] * 7,
            np.zeros((7, 1)),
            np.zeros(6),
            np.array(["x"] * 7, dtype=object),
            np.array(["1.25"] * 7, dtype=object),
            np.array([np.nan] + [1.0] * 6),
            np.array([np.inf] + [1.0] * 6),
        ],
    )
    def test_invalid_prediction_output_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
        prediction: Any,
    ) -> None:
        _install_prediction_stub(monkeypatch, prediction)
        with pytest.raises(AdvancedRegressionError, match="prediction"):
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_negative_predictions_reach_metrics_without_clipping(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        negative = np.full(len(split_c.validation), -7.25, dtype="float64")
        seen: list[np.ndarray] = []
        _install_prediction_stub(monkeypatch, negative)
        real_metric = regression.mean_absolute_error

        def metric_spy(y_true: Any, y_pred: Any) -> Any:
            seen.append(np.asarray(y_pred).copy())
            return real_metric(y_true, y_pred)

        monkeypatch.setattr(regression, "mean_absolute_error", metric_spy)
        evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert any(np.all(values == -7.25) for values in seen)

    def test_estimator_value_error_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        class Boom:
            def fit(self, x: Any, y: Any) -> None:
                raise ValueError("secret estimator detail")

        monkeypatch.setattr(regression, "Ridge", lambda **kwargs: Boom())
        with pytest.raises(AdvancedRegressionError, match="fitted") as excinfo:
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "secret" not in str(excinfo.value)

    def test_unexpected_exception_is_not_relabelled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        class Boom:
            def fit(self, x: Any, y: Any) -> None:
                raise RuntimeError("programming failure")

        monkeypatch.setattr(regression, "Ridge", lambda **kwargs: Boom())
        with pytest.raises(RuntimeError, match="programming failure"):
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())


class TestMetricValidation:
    def test_r2_uses_force_finite_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        calls: list[dict[str, object]] = []

        def r2_spy(y_true: Any, y_pred: Any, **kwargs: object) -> float:
            calls.append(kwargs)
            return -0.25

        monkeypatch.setattr(regression, "r2_score", r2_spy)
        result = evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        assert result.test.r2 == -0.25
        assert calls == [{"force_finite": False}]

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("mean_absolute_error", -1.0),
            ("root_mean_squared_error", -1.0),
            ("mean_absolute_error", "1.0"),
            ("root_mean_squared_error", "1.0"),
            ("mean_absolute_error", float("nan")),
            ("root_mean_squared_error", float("inf")),
        ],
    )
    def test_invalid_loss_metric_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
        name: str,
        value: float,
    ) -> None:
        monkeypatch.setattr(regression, name, lambda *args, **kwargs: value)
        with pytest.raises(AdvancedRegressionError):
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_nonfinite_r2_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        monkeypatch.setattr(regression, "r2_score", lambda *args, **kwargs: float("nan"))
        with pytest.raises(AdvancedRegressionError, match="R-squared"):
            evaluate_advanced_regressor(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())


def test_phase12_does_not_reuse_classification_error() -> None:
    assert not issubclass(AdvancedRegressionError, AdvancedClassificationError)


class TestPoisonedNestedObjects:
    def test_poisoned_export_copy_method_cannot_leak_or_change_result(
        self,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        baseline = evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        poisoned = dataset_c.observations.copy(deep=True)
        object.__setattr__(
            poisoned,
            "copy",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("SENSITIVE_INPUT_MARKER")),
        )
        forged = replace(dataset_c, observations=poisoned)
        assert (
            evaluate_advanced_regressor(
                forged,
                split_c,
                fit_c,
                CRAFTED_SPEC,
                RoadGuardConfig(),
            )
            == baseline
        )

    def test_poisoned_split_reset_index_method_cannot_leak_or_change_result(
        self,
        dataset_c: RepositoryExport,
        split_c: Any,
        fit_c: Any,
    ) -> None:
        baseline = evaluate_advanced_regressor(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        poisoned = split_c.train.copy(deep=True)
        object.__setattr__(
            poisoned,
            "reset_index",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("SENSITIVE_INPUT_MARKER")),
        )
        forged = replace(split_c, train=poisoned)
        assert (
            evaluate_advanced_regressor(
                dataset_c,
                forged,
                fit_c,
                CRAFTED_SPEC,
                RoadGuardConfig(),
            )
            == baseline
        )
