"""Phase 11 adversarial and temporal-boundary contract tests.

These tests spy on the Phase 11 module boundary (constructor globals,
``transform``, and estimator methods) to prove the locked temporal contract:
exact constructors and derived seeds, exactly one fit per candidate on the
exact transformed training matrix, exactly one validation ``predict_proba``
per candidate, exactly one test transform, exactly one selected test
``predict_proba``, no loser test call, and no ``predict`` call anywhere.
Stubs and monkeypatches also exercise adversarial boundaries that real
estimators never produce (invalid probability/metric/confusion output and
estimator arithmetic failures).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

import roadguard.classification as classification_module
from roadguard import (
    DatasetSpec,
    RepositoryExport,
    RoadGuardConfig,
    clean_raw_dataset,
    derive_observation_targets,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
)
from roadguard.classification import (
    ADVANCED_CLASSIFIER_RNG_NAMESPACE,
    CANDIDATE_CLASSIFIER_NAMES,
    AdvancedClassificationError,
    evaluate_advanced_classifier,
)
from roadguard.features import FEATURE_KEY_COLUMNS, FeatureInputError, build_feature_frame
from roadguard.preprocessing import (
    ChronologicalSplit,
    PreprocessorFit,
    fit_preprocessor,
    split_chronologically,
    transform,
)

START = date(2022, 1, 1)
CRAFTED_SPEC = DatasetSpec(
    dataset_segments=1, dataset_months_per_segment=48, dataset_observations=48
)


def _month_first(month_index: int) -> date:
    year = 2022 + (month_index - 1) // 12
    month = (month_index - 1) % 12 + 1
    return date(year, month, 1)


def _month_15(month_index: int) -> date:
    return _month_first(month_index).replace(day=15)


def _crafted_export() -> RepositoryExport:
    """Two-class-everywhere crafted export (months 36-44 and 46-60 plus two
    early events), identical in shape to the ordinary Phase 11 fixture."""
    events = (
        [date(2022, 3, 20), date(2022, 10, 5)]
        + [_month_15(month_index) for month_index in range(36, 45)]
        + [_month_15(month_index) for month_index in range(46, 61)]
    )
    construction = pd.Timestamp("2000-01-01")
    segments = pd.DataFrame(
        {
            "segment_id": pd.Series(["QL01-KM0-1"], dtype=object),
            "province": pd.Series(["NA"], dtype=object),
            "road_type": pd.Series(["national"], dtype=object),
            "construction_date": pd.to_datetime([construction]).astype("datetime64[ns]"),
            "road_length_km": pd.Series([1.0], dtype="float64"),
        }
    )
    event_frame = pd.DataFrame(
        {
            "segment_id": pd.Series(["QL01-KM0-1"] * len(events), dtype=object),
            "maintenance_date": pd.to_datetime(pd.Series(events)).astype("datetime64[ns]"),
        }
    )
    rows: list[dict[str, object]] = []
    for month_index in range(1, 49):
        t = _month_first(month_index)
        prior = [event for event in events if event < t]
        rows.append(
            {
                "segment_id": "QL01-KM0-1",
                "date": t,
                "traffic_volume": 1000,
                "heavy_vehicle_ratio": 0.3,
                "road_age_days": (t - construction.date()).days,
                "rainfall_mm": 100.0,
                "temperature": 27.0,
                "humidity": 60.0,
                "days_since_last_maintenance": (
                    (t - prior[-1]).days if prior else min((t - construction.date()).days, 3650)
                ),
                "previous_repairs": len(prior),
                "road_condition_score": 50,
                "marking_condition_score": 50,
                "guardrail_condition_score": 50,
                "sign_condition_score": 50,
                "accident_count_30d": 0,
                "accident_count_365d": 0,
            }
        )
    observations = pd.DataFrame(rows)
    observations["segment_id"] = observations["segment_id"].astype(object)
    observations["date"] = pd.to_datetime(observations["date"]).astype("datetime64[ns]")
    for column in (
        "traffic_volume",
        "road_age_days",
        "days_since_last_maintenance",
        "previous_repairs",
        "road_condition_score",
        "marking_condition_score",
        "guardrail_condition_score",
        "sign_condition_score",
        "accident_count_30d",
        "accident_count_365d",
    ):
        observations[column] = observations[column].astype("int64")
    for column in ("heavy_vehicle_ratio", "rainfall_mm", "temperature", "humidity"):
        observations[column] = observations[column].astype("float64")
    targets = derive_observation_targets(observations, event_frame)
    return RepositoryExport(
        segments=segments,
        observations=observations,
        targets=targets,
        maintenance_events=event_frame,
    )


def _generated_export(spec: DatasetSpec, seed: int = 42) -> RepositoryExport:
    segments = generate_segments(spec, seed, observation_start=START)
    events = generate_maintenance_events(segments, spec, seed, start_date=START)
    timeline = generate_accident_timeline(segments, spec, seed, start_date=START)
    observations = generate_observations(segments, events, timeline, spec, seed, start_date=START)
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, spec)
    return RepositoryExport(
        segments=cleaned.segments,
        observations=cleaned.observations,
        targets=cleaned.targets,
        maintenance_events=cleaned.maintenance_events,
    )


def _canonical_split(dataset: RepositoryExport, spec: DatasetSpec) -> ChronologicalSplit:
    return split_chronologically(build_feature_frame(dataset, spec), spec)


def _canonical_fit(split: ChronologicalSplit, spec: DatasetSpec) -> PreprocessorFit:
    return fit_preprocessor(split, spec)


def _expected_matrices(
    dataset: RepositoryExport, spec: DatasetSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical = _canonical_split(dataset, spec)
    refit = _canonical_fit(canonical, spec)
    matrices = {}
    labels = {}
    for name in ("train", "validation", "test"):
        transformed = transform(getattr(canonical, name), refit)
        matrices[name] = transformed.features.to_numpy(dtype="float64")
        joined = transformed.keys.merge(
            dataset.targets[["segment_id", "date", "maintenance_within_30_days"]],
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )
        labels[name] = joined["maintenance_within_30_days"].to_numpy(dtype="int64")
    return (
        matrices["train"],
        matrices["validation"],
        matrices["test"],
        labels["train"],
        labels["validation"],
        labels["test"],
    )


def _recording_wrapper(delegate: Any, calls: list[tuple[Any, str, Any, Any]]) -> Any:
    class _RecordingEstimator:
        def fit(self, x: Any, y: Any) -> Any:
            calls.append((self, "fit", np.asarray(x), np.asarray(y)))
            delegate.fit(x, y)
            return self

        def predict_proba(self, x: Any) -> Any:
            calls.append((self, "predict_proba", np.asarray(x), None))
            return delegate.predict_proba(x)

        def predict(self, x: Any) -> Any:
            calls.append((self, "predict", np.asarray(x), None))
            return delegate.predict(x)

        def __getattr__(self, name: str) -> Any:
            return getattr(delegate, name)

    return _RecordingEstimator()


@pytest.fixture(scope="module")
def dataset_c() -> RepositoryExport:
    return _crafted_export()


@pytest.fixture(scope="module")
def split_c(dataset_c: RepositoryExport) -> ChronologicalSplit:
    return _canonical_split(dataset_c, CRAFTED_SPEC)


@pytest.fixture(scope="module")
def fit_c(split_c: ChronologicalSplit) -> PreprocessorFit:
    return _canonical_fit(split_c, CRAFTED_SPEC)


class TestExactConstructorsAndSeeds:
    def test_exact_constructor_parameters_and_derived_seeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        captured: dict[str, dict[str, object]] = {}

        def logistic_spy(**kwargs: object) -> Any:
            captured["logistic"] = kwargs
            return LogisticRegression(**kwargs)

        def hgb_spy(**kwargs: object) -> Any:
            captured["hgb"] = kwargs
            return HistGradientBoostingClassifier(**kwargs)

        monkeypatch.setattr(classification_module, "LogisticRegression", logistic_spy)
        monkeypatch.setattr(classification_module, "HistGradientBoostingClassifier", hgb_spy)

        result = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
        )
        expected_logistic_seed = int(
            np.random.SeedSequence([42, ADVANCED_CLASSIFIER_RNG_NAMESPACE, 0]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        expected_hgb_seed = int(
            np.random.SeedSequence([42, ADVANCED_CLASSIFIER_RNG_NAMESPACE, 1]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        assert captured["logistic"] == {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-8,
            "fit_intercept": True,
            "class_weight": None,
            "random_state": expected_logistic_seed,
        }
        assert captured["hgb"] == {
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": expected_hgb_seed,
        }
        assert type(captured["logistic"]["random_state"]) is int
        assert type(captured["hgb"]["random_state"]) is int
        assert captured["logistic"]["random_state"] != captured["hgb"]["random_state"]
        assert result.selected_classifier_name in CANDIDATE_CLASSIFIER_NAMES

    def test_derived_seeds_follow_config_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        captured: dict[str, dict[str, object]] = {}

        def logistic_spy(**kwargs: object) -> Any:
            captured["logistic"] = kwargs
            return LogisticRegression(**kwargs)

        def hgb_spy(**kwargs: object) -> Any:
            captured["hgb"] = kwargs
            return HistGradientBoostingClassifier(**kwargs)

        monkeypatch.setattr(classification_module, "LogisticRegression", logistic_spy)
        monkeypatch.setattr(classification_module, "HistGradientBoostingClassifier", hgb_spy)
        evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=7)
        )
        for index, key in ((0, "logistic"), (1, "hgb")):
            expected = int(
                np.random.SeedSequence(
                    [7, ADVANCED_CLASSIFIER_RNG_NAMESPACE, index]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            assert captured[key]["random_state"] == expected

    def test_seed_derivation_failure_is_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        class _BoomSeedSequence:
            def generate_state(self, *args: Any, **kwargs: Any) -> np.ndarray:
                raise ValueError("raw seed detail")

        monkeypatch.setattr(
            classification_module.np.random,
            "SeedSequence",
            lambda *args, **kwargs: _BoomSeedSequence(),
        )
        with pytest.raises(AdvancedClassificationError, match="seed derivation") as excinfo:
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "raw" not in str(excinfo.value)
        assert excinfo.value.__cause__ is None


class TestFitAndPredictBoundaries:
    def test_validation_mutation_cannot_change_candidate_fit_inputs_or_seeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        fit_calls: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            "logistic": [],
            "hgb": [],
        }
        constructor_calls: dict[str, list[dict[str, object]]] = {
            "logistic": [],
            "hgb": [],
        }

        def make_spy(kind: str) -> Any:
            def factory(**kwargs: object) -> Any:
                constructor_calls[kind].append(dict(kwargs))
                delegate = (
                    LogisticRegression(**kwargs)
                    if kind == "logistic"
                    else HistGradientBoostingClassifier(**kwargs)
                )

                class _FitSpy:
                    def fit(self, x: Any, y: Any) -> _FitSpy:
                        fit_calls[kind].append((np.asarray(x).copy(), np.asarray(y).copy()))
                        delegate.fit(x, y)
                        return self

                    def predict_proba(self, x: Any) -> Any:
                        return delegate.predict_proba(x)

                    def __getattr__(self, name: str) -> Any:
                        return getattr(delegate, name)

                return _FitSpy()

            return factory

        monkeypatch.setattr(classification_module, "LogisticRegression", make_spy("logistic"))
        monkeypatch.setattr(
            classification_module, "HistGradientBoostingClassifier", make_spy("hgb")
        )

        observations = dataset_c.observations.copy(deep=True)
        mask = observations["date"].isin(pd.to_datetime(split_c.validation_dates))
        observations.loc[mask, "temperature"] = observations.loc[mask, "temperature"] + 1.0
        mutated = replace(dataset_c, observations=observations)
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)

        evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
        )
        evaluate_advanced_classifier(
            mutated, mutated_split, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
        )

        original_matrices = _expected_matrices(dataset_c, CRAFTED_SPEC)
        mutated_matrices = _expected_matrices(mutated, CRAFTED_SPEC)
        assert not np.array_equal(original_matrices[1], mutated_matrices[1])
        for kind in ("logistic", "hgb"):
            assert len(constructor_calls[kind]) == 2
            assert constructor_calls[kind][0] == constructor_calls[kind][1]
            assert len(fit_calls[kind]) == 2
            assert np.array_equal(fit_calls[kind][0][0], fit_calls[kind][1][0])
            assert np.array_equal(fit_calls[kind][0][1], fit_calls[kind][1][1])

    def test_exact_fit_and_predict_proba_boundaries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        calls: list[tuple[Any, str, Any, Any]] = []
        wrappers: dict[str, Any] = {}

        def make_spy(kind: str) -> Any:
            def factory(**kwargs: object) -> Any:
                delegate = (
                    LogisticRegression(**kwargs)
                    if kind == "logistic"
                    else HistGradientBoostingClassifier(**kwargs)
                )
                wrapper = _recording_wrapper(delegate, calls)
                wrappers[kind] = wrapper
                return wrapper

            return factory

        monkeypatch.setattr(classification_module, "LogisticRegression", make_spy("logistic"))
        monkeypatch.setattr(
            classification_module, "HistGradientBoostingClassifier", make_spy("hgb")
        )
        result = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
        )
        x_train, x_validation, x_test, y_train, _, _ = _expected_matrices(dataset_c, CRAFTED_SPEC)
        fit_calls = [call for call in calls if call[1] == "fit"]
        proba_calls = [call for call in calls if call[1] == "predict_proba"]
        predict_calls = [call for call in calls if call[1] == "predict"]
        assert len(wrappers) == 2
        assert len(fit_calls) == 2
        for call in fit_calls:
            assert np.array_equal(call[2], x_train)
            assert np.array_equal(call[3], y_train)
        assert predict_calls == []
        winner_kind = (
            "logistic"
            if result.selected_classifier_name == CANDIDATE_CLASSIFIER_NAMES[0]
            else "hgb"
        )
        loser_kind = "hgb" if winner_kind == "logistic" else "logistic"
        winner_calls = [call for call in proba_calls if call[0] is wrappers[winner_kind]]
        loser_calls = [call for call in proba_calls if call[0] is wrappers[loser_kind]]
        assert len(winner_calls) == 2
        assert np.array_equal(winner_calls[0][2], x_validation)
        assert np.array_equal(winner_calls[1][2], x_test)
        assert len(loser_calls) == 1
        assert np.array_equal(loser_calls[0][2], x_validation)

    def test_validation_predict_proba_happens_once_per_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        calls: list[tuple[Any, str, Any, Any]] = []
        wrappers: dict[str, Any] = {}

        def make_spy(kind: str) -> Any:
            def factory(**kwargs: object) -> Any:
                delegate = (
                    LogisticRegression(**kwargs)
                    if kind == "logistic"
                    else HistGradientBoostingClassifier(**kwargs)
                )
                wrapper = _recording_wrapper(delegate, calls)
                wrappers[kind] = wrapper
                return wrapper

            return factory

        monkeypatch.setattr(classification_module, "LogisticRegression", make_spy("logistic"))
        monkeypatch.setattr(
            classification_module, "HistGradientBoostingClassifier", make_spy("hgb")
        )
        evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
        )
        _, x_validation, _, _, _, _ = _expected_matrices(dataset_c, CRAFTED_SPEC)
        for kind in ("logistic", "hgb"):
            fit_count = sum(1 for call in calls if call[0] is wrappers[kind] and call[1] == "fit")
            validation_count = sum(
                1
                for call in calls
                if call[0] is wrappers[kind]
                and call[1] == "predict_proba"
                and np.array_equal(call[2], x_validation)
            )
            assert fit_count == 1
            assert validation_count == 1


class TestTransformBoundary:
    def test_exactly_one_transform_per_partition_in_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        calls: list[tuple[pd.DataFrame, PreprocessorFit]] = []
        real_transform = classification_module.transform

        def spy(frame: pd.DataFrame, fitted: PreprocessorFit) -> Any:
            calls.append((frame, fitted))
            return real_transform(frame, fitted)

        monkeypatch.setattr(classification_module, "transform", spy)
        evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert len(calls) == 3
        canonical = _canonical_split(dataset_c, CRAFTED_SPEC)
        assert_frame_equal(calls[0][0], canonical.train)
        assert_frame_equal(calls[1][0], canonical.validation)
        assert_frame_equal(calls[2][0], canonical.test)
        for _, fitted in calls:
            assert fitted == fit_c


class TestExportSnapshotBoundary:
    def test_feature_validation_receives_only_a_deep_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        dataset = RepositoryExport(
            segments=dataset_c.segments.copy(deep=True),
            observations=dataset_c.observations.copy(deep=True),
            targets=dataset_c.targets.copy(deep=True),
            maintenance_events=dataset_c.maintenance_events.copy(deep=True),
        )
        before_segments = dataset.segments.copy(deep=True)
        received: list[RepositoryExport] = []

        def mutating_failure(export: RepositoryExport, spec: DatasetSpec) -> pd.DataFrame:
            received.append(export)
            export.segments.loc[0, "segment_id"] = "downstream-mutation"
            raise FeatureInputError("raw downstream detail")

        monkeypatch.setattr(classification_module, "build_feature_frame", mutating_failure)
        with pytest.raises(AdvancedClassificationError) as excinfo:
            evaluate_advanced_classifier(dataset, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

        assert len(received) == 1
        assert received[0] is not dataset
        assert received[0].segments is not dataset.segments
        assert received[0].observations is not dataset.observations
        assert received[0].targets is not dataset.targets
        assert received[0].maintenance_events is not dataset.maintenance_events
        assert_frame_equal(dataset.segments, before_segments)
        assert "downstream" not in str(excinfo.value)


class TestEstimatorFailures:
    def test_fit_value_error_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        class _BoomValue:
            def fit(self, x: Any, y: Any) -> None:
                raise ValueError("secret detail")

        monkeypatch.setattr(classification_module, "LogisticRegression", lambda **kw: _BoomValue())
        monkeypatch.setattr(
            classification_module,
            "HistGradientBoostingClassifier",
            lambda **kw: HistGradientBoostingClassifier(**kw),
        )
        with pytest.raises(AdvancedClassificationError, match="fitted") as excinfo:
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "secret" not in str(excinfo.value)

    def test_fit_arithmetic_failure_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        class _BoomArithmetic:
            def fit(self, x: Any, y: Any) -> None:
                raise OverflowError("overflow detail")

        monkeypatch.setattr(
            classification_module, "LogisticRegression", lambda **kw: _BoomArithmetic()
        )
        monkeypatch.setattr(
            classification_module,
            "HistGradientBoostingClassifier",
            lambda **kw: HistGradientBoostingClassifier(**kw),
        )
        with pytest.raises(AdvancedClassificationError, match="fitted") as excinfo:
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "overflow" not in str(excinfo.value)


class TestProbabilityOutputValidation:
    @staticmethod
    def _install_predict_proba_stub(
        monkeypatch: pytest.MonkeyPatch,
        probabilities: Any,
        classes: Any = (0, 1),
        target: str = "logistic",
    ) -> None:
        class _Stub:
            classes_ = np.asarray(classes)

            def fit(self, x: Any, y: Any) -> _Stub:
                return self

            def predict_proba(self, x: Any) -> Any:
                return probabilities

        def stub_factory(**kwargs: object) -> Any:
            return _Stub()

        if target == "logistic":
            monkeypatch.setattr(classification_module, "LogisticRegression", stub_factory)
            monkeypatch.setattr(
                classification_module,
                "HistGradientBoostingClassifier",
                lambda **kw: HistGradientBoostingClassifier(**kw),
            )
        else:
            monkeypatch.setattr(
                classification_module, "LogisticRegression", lambda **kw: LogisticRegression(**kw)
            )
            monkeypatch.setattr(
                classification_module, "HistGradientBoostingClassifier", stub_factory
            )

    def test_malformed_probability_shape_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        self._install_predict_proba_stub(monkeypatch, np.zeros((7, 1)))
        with pytest.raises(AdvancedClassificationError, match="malformed"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_non_ndarray_probabilities_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        self._install_predict_proba_stub(monkeypatch, [[0.5, 0.5]] * 7)
        with pytest.raises(AdvancedClassificationError, match="malformed"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_non_numeric_probabilities_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        probabilities = np.full((7, 2), "x", dtype=object)
        self._install_predict_proba_stub(monkeypatch, probabilities)
        with pytest.raises(AdvancedClassificationError, match="probabilit"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_non_finite_probabilities_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        probabilities = np.full((7, 2), 0.5, dtype="float64")
        probabilities[0, 1] = np.nan
        probabilities[1, 1] = np.inf
        self._install_predict_proba_stub(monkeypatch, probabilities)
        with pytest.raises(AdvancedClassificationError, match="non-finite"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_out_of_range_probabilities_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        probabilities = np.full((7, 2), 0.5, dtype="float64")
        probabilities[0, 1] = 1.5
        self._install_predict_proba_stub(monkeypatch, probabilities)
        with pytest.raises(AdvancedClassificationError, match="range"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        probabilities = np.full((7, 2), 0.5, dtype="float64")
        probabilities[0, 1] = -0.2
        self._install_predict_proba_stub(monkeypatch, probabilities)
        with pytest.raises(AdvancedClassificationError, match="range"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_unexpected_class_order_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        self._install_predict_proba_stub(
            monkeypatch, np.full((7, 2), 0.5, dtype="float64"), classes=(1, 0)
        )
        with pytest.raises(AdvancedClassificationError, match="class"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_predict_proba_value_error_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        class _Boom:
            classes_ = np.asarray([0, 1])

            def fit(self, x: Any, y: Any) -> _Boom:
                return self

            def predict_proba(self, x: Any) -> Any:
                raise ValueError("probability detail")

        monkeypatch.setattr(classification_module, "LogisticRegression", lambda **kw: _Boom())
        monkeypatch.setattr(
            classification_module,
            "HistGradientBoostingClassifier",
            lambda **kw: HistGradientBoostingClassifier(**kw),
        )
        with pytest.raises(AdvancedClassificationError, match="probabilit") as excinfo:
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "detail" not in str(excinfo.value)

    def test_class_state_value_error_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        class _Boom:
            @property
            def classes_(self) -> np.ndarray:
                raise ValueError("raw class state")

            def fit(self, x: Any, y: Any) -> _Boom:
                return self

            def predict_proba(self, x: Any) -> np.ndarray:
                return np.full((len(x), 2), 0.5, dtype="float64")

        monkeypatch.setattr(classification_module, "LogisticRegression", lambda **kw: _Boom())
        with pytest.raises(AdvancedClassificationError, match="class") as excinfo:
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "raw" not in str(excinfo.value)


class TestMetricOutputValidation:
    @pytest.mark.parametrize("metric_name", ["f1_score", "recall_score"])
    def test_out_of_range_threshold_metric_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
        metric_name: str,
    ) -> None:
        real_metric = getattr(classification_module, metric_name)
        calls = 0

        def one_invalid_value(*args: Any, **kwargs: Any) -> float:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 2.0
            return float(real_metric(*args, **kwargs))

        monkeypatch.setattr(classification_module, metric_name, one_invalid_value)
        with pytest.raises(AdvancedClassificationError, match="range"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_invalid_validation_metric_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        def boom(y_true: Any, y_score: Any) -> float:
            raise ValueError("metric detail")

        monkeypatch.setattr(classification_module, "average_precision_score", boom)
        with pytest.raises(AdvancedClassificationError, match="PR-AUC") as excinfo:
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "detail" not in str(excinfo.value)

    def test_out_of_range_validation_metric_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        monkeypatch.setattr(
            classification_module, "average_precision_score", lambda y_true, y_score: 2.0
        )
        with pytest.raises(AdvancedClassificationError, match="range"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_non_finite_test_metric_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        monkeypatch.setattr(
            classification_module, "roc_auc_score", lambda y_true, y_score: float("nan")
        )
        with pytest.raises(AdvancedClassificationError, match="finite"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_out_of_range_test_metric_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        monkeypatch.setattr(classification_module, "accuracy_score", lambda y_true, y_pred: 1.5)
        with pytest.raises(AdvancedClassificationError, match="range"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_test_metric_value_error_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        def boom(y_true: Any, y_pred: Any, **kwargs: object) -> float:
            raise ValueError("accuracy detail")

        monkeypatch.setattr(classification_module, "accuracy_score", boom)
        with pytest.raises(AdvancedClassificationError, match="accuracy") as excinfo:
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert "detail" not in str(excinfo.value)


class TestConfusionMatrixValidation:
    def test_malformed_shape_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        monkeypatch.setattr(
            classification_module,
            "confusion_matrix",
            lambda y_true, y_pred, labels: np.zeros((1, 3)),
        )
        with pytest.raises(AdvancedClassificationError, match="malformed"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_non_integer_counts_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        monkeypatch.setattr(
            classification_module,
            "confusion_matrix",
            lambda y_true, y_pred, labels: np.array([[0.5, 1.0], [2.0, 3.0]]),
        )
        with pytest.raises(AdvancedClassificationError, match="malformed"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_negative_counts_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        monkeypatch.setattr(
            classification_module,
            "confusion_matrix",
            lambda y_true, y_pred, labels: np.array([[-1, 0], [0, 8]]),
        )
        with pytest.raises(AdvancedClassificationError, match="malformed"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_wrong_total_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        monkeypatch.setattr(
            classification_module,
            "confusion_matrix",
            lambda y_true, y_pred, labels: np.array([[1, 1], [1, 1]]),
        )
        with pytest.raises(AdvancedClassificationError, match="malformed"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())


class TestGeneratedProfileBoundaries:
    def test_generated_export_boundaries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spec = DatasetSpec(
            dataset_segments=5, dataset_months_per_segment=48, dataset_observations=240
        )
        dataset = _generated_export(spec)
        split = _canonical_split(dataset, spec)
        fit = _canonical_fit(split, spec)
        calls: list[tuple[pd.DataFrame, PreprocessorFit]] = []
        real_transform = classification_module.transform

        def spy(frame: pd.DataFrame, fitted: PreprocessorFit) -> Any:
            calls.append((frame, fitted))
            return real_transform(frame, fitted)

        monkeypatch.setattr(classification_module, "transform", spy)
        result = evaluate_advanced_classifier(dataset, split, fit, spec, RoadGuardConfig())
        assert len(calls) == 3
        assert result.train_rows == 5 * 34
        assert result.validation_rows == 5 * 7
        assert result.test_rows == 5 * 7
        assert result.selected_classifier_name in CANDIDATE_CLASSIFIER_NAMES
        assert (
            tuple(record.classifier_name for record in result.candidates)
            == CANDIDATE_CLASSIFIER_NAMES
        )
