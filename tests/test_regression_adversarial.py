"""Adversarial Phase 12 estimator, temporal, prediction and metric tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

import roadguard.regression as regression
from roadguard import RepositoryExport, RoadGuardConfig
from roadguard.classification import AdvancedClassificationError
from roadguard.regression import (
    ADVANCED_REGRESSOR_RNG_NAMESPACE,
    AdvancedRegressionError,
    evaluate_advanced_regressor,
)
from test_classification import CRAFTED_SPEC, _canonical_fit, _canonical_split, _export_c


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
        assert len(fit_calls) == 2
        assert all(call[2].shape[0] == len(split_c.train) for call in fit_calls)
        assert len(predict_calls) == 3
        validation_calls = [
            call for call in predict_calls if call[2].shape[0] == len(split_c.validation)
        ]
        test_calls = [call for call in predict_calls if call[2].shape[0] == len(split_c.test)]
        assert len(validation_calls) == 2
        assert len(test_calls) == 1
        expected_winner = "ridge" if result.selected_regressor_name == "ridge_l2" else "hgb"
        assert test_calls[0][0] == expected_winner

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
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("mean_absolute_error", -1.0),
            ("root_mean_squared_error", -1.0),
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
