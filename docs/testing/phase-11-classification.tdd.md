# Phase 11 TDD Evidence — Advanced classification

## Source and scope

The journeys and acceptance criteria are derived from
[`docs/phase-plan.md`](../phase-plan.md) and the Phase 11 addendum in
[`docs/contracts.md`](../contracts.md) section 18. Phase 11 fits exactly two
locked feature-dependent candidates — `logistic_l2`
(`LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
tol=1e-8, fit_intercept=True, class_weight=None)`) and
`hist_gradient_boosting`
(`HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100,
max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False)`) — selects
exactly one by validation PR-AUC/F1/recall with fixed-order tie-break, and
evaluates only the frozen selected candidate on the test partition exactly
once in one private stage. Each candidate's `random_state` derives from
`config.seed` through the locked namespace `0x5247311`. Advanced regression,
calibration, search, cross-validation, artifact persistence, risk mapping,
forecasting, optimization, inference, explainability, and all Phase 12+
behavior are out of scope.

Phase 11 adds no dependency: it uses the `scikit-learn>=1.5,<2` environment
locked by Phase 10. `pyproject.toml` and `uv.lock` are unchanged.

## User journeys

- As a modelling workflow, I need two fixed feature-dependent candidates with
  reproducible derived seeds, selected purely by validation evidence.
- As a reviewer, I need exact train-only fitting, exactly one validation
  `predict_proba` per candidate, exactly one frozen selected-only test stage,
  and no global RNG or unseeded stochastic operation.
- As an operator, I need forged exports, splits, fits, config seeds, invalid
  estimator/metric output, and degenerate partitions to fail contextually with
  sanitized messages.

## RED evidence

| Stage | Command | Genuine result |
| --- | --- | --- |
| No implementation | `uv run pytest tests/test_classification.py tests/test_classification_adversarial.py tests/test_package.py -q` | RED: `ModuleNotFoundError: No module named 'roadguard.classification'` — collection error for both new contract test modules; all Phase 11 tests blocked. The Phase 11 public module/API is absent, not a broken fixture or missing dependency. |
| After first implementation | same command | RED: 3 failed, 75 passed — pydantic `model_construct()` applies field defaults (the forged missing-seed scenario needed an emptied `__dict__`), and the out-of-range metric message did not match the "range" context regex. |
| DeepSeek handoff before independent review | same command | GREEN: 78 passed in 58.31s (76 classification + adversarial contract tests + 2 package-surface). |
| Independent Codex review probes | focused adversarial probes and regression tests | RED: the feature builder received the caller export instead of its snapshot; forged exact-type nested state leaked raw `AttributeError`; invalid estimator class state leaked raw `ValueError`; and one-shot threshold F1/recall values above 1.0 could influence selection without failure. |
| After review repairs and evidence additions | same command | GREEN: 89 passed in 63.76s (87 classification + adversarial contract tests + 2 package-surface). |

## GREEN evidence

| Command | Result |
| --- | --- |
| `uv lock --check` | GREEN: resolved 38 packages with no lockfile change. |
| `uv sync --frozen` | GREEN: locked environment checked; 38 packages, no lockfile drift. |
| `uv run pytest tests/test_classification.py tests/test_classification_adversarial.py tests/test_package.py -q` | GREEN: **89 passed in 63.76s**. |
| Full suite with real disposable PostgreSQL (`postgres:16-alpine` container, `TEST_DATABASE_URL` injected through the environment, never stored) | GREEN: `uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80` → **962 passed in 685.42s**; total coverage **93.81%** (branch coverage included); `roadguard.classification` **89%** (remaining lines are defensive sanitize/except branches, consistent with earlier phases). |
| `uv run ruff check .` | All checks passed. |
| `uv run ruff format --check .` | 54 files already formatted. |
| `uv run mypy src` | Success: no issues found in 27 source files. |
| `git diff --check` | clean (exit 0). |

The locked `penalty="l2"` constructor emits a `FutureWarning` under the pinned
scikit-learn 1.9 environment; the parameter is contract-frozen by section 18
and is intentionally not changed.

## Independent review and repairs

The Codex review found no P1 issue and four P2 issues. RED-first regression
tests were added before each repair where the public failure could be exercised
directly. The evaluator now passes one deep-copied export snapshot to both
feature validation and target joining; sanitizes missing exact-type
export/split/fit/spec fields and expected estimator class-state failures;
validates threshold-search F1 and recall inside `[0, 1]`; and sanitizes seed
derivation failures. Additional spies prove that changing validation features
changes the validation matrix but cannot change either candidate's constructor
seed or train `X`/`y`, and that an exact `RoadGuardConfig` containing only
`seed` succeeds without reading any irrelevant field. Two independent
read-only re-reviews finished with zero P1/P2 blockers.

## Guarantees covered

| # | What is guaranteed | Test |
| --- | --- | --- |
| 1 | Exact module `__all__` (8 symbols), package-root exports, constants (`roadguard.phase11.v1`, `0x5247311`, locked name pair), exact five-argument signature, no public test-evaluation function. | `TestPublicSurface` |
| 2 | Frozen dataclasses with the exact field names and order from section 18; only built-in scalars and tuples stored. | `TestFrozenSchema` |
| 3 | Both exact constructors receive every locked parameter and the exact formula-derived `random_state` per candidate index; derived seeds follow `config.seed`; expected seed-derivation failures are sanitized with suppressed chaining. | `TestExactConstructorsAndSeeds` |
| 4 | Known-vector metric comparisons against an independent direct-scikit-learn reference for crafted and generated exports; selected name matches exactly one candidate record. | `TestKnownVectors` |
| 5 | Threshold contract: candidates `{0.0, 1.0}` plus distinct validation probabilities; ranking F1, then recall, then higher threshold (crafted-vector tests for all three rules). | `TestThresholdRanking` |
| 6 | Candidate selection contract: higher validation PR-AUC, then F1, then recall, then earlier locked position (crafted-record tests for all four rules). | `TestCandidateSelectionRanking` |
| 7 | Exactly one fit per candidate receiving the exact transformed training matrix and labels; exactly one validation `predict_proba` per candidate; winner receives exactly one test `predict_proba`; loser never receives a test matrix; `predict` never called. | `TestFitAndPredictBoundaries` |
| 8 | Exactly three transforms (train, validation, test) with the canonical frames and the reproduced fit. | `TestTransformBoundary` |
| 9 | Test feature/target mutations cannot change candidate records, selected name, threshold, feature schema, or row counts; validation mutations change the validation matrix but cannot change either candidate's constructor seed or train `X`/`y`; shuffled inputs and repeated calls reproduce equal results. | `TestTemporalLeakage`, `TestFitAndPredictBoundaries` |
| 10 | Caller-owned export frames, split partitions, fit, and config remain unchanged on success and failure; downstream validation receives only a deep snapshot even when it mutates before failure. | `TestCallerImmutability`, `TestExportSnapshotBoundary` |
| 11 | Wrong top-level types, lookalikes, and subclasses raise `TypeError` before field access (all five arguments). | `TestInputValidation` |
| 12 | Forged invalid config seeds (bool, zero, negative, string, numpy integer, int subclass, missing field) raise the fixed configuration-seed error; an exact config containing only `seed` succeeds; irrelevant config fields, environment variables, and `load_config` are never read; changed seed preserves provenance fields. | `TestConfigValidation` |
| 13 | Forged split membership/dates, forged `PreprocessorFit`, mismatched spec, missing exact-type export/split/fit/spec fields, and missing/duplicate/forged targets fail contextually with sanitized fixed messages containing no raw values. | `TestForgedInputs`, `TestTargetAlignment`, `TestSanitizedErrors` |
| 14 | Keys, both target columns, future event keys, costs, materials, and latent fields never become model features; `feature_columns` equals `PreprocessorFit.transformed_feature_columns`. | `TestForbiddenFields` |
| 15 | Single-class train/validation/test partitions fail contextually (the test-class check only inside the frozen private test stage). | `TestDegeneratePartitions` |
| 16 | Malformed, non-ndarray, non-numeric, non-finite, and out-of-range probabilities; unexpected or failing estimator class state; estimator fit `ValueError`/arithmetic failures — all sanitized. | `TestProbabilityOutputValidation`, `TestEstimatorFailures` |
| 17 | Invalid, out-of-range, and non-finite metric outputs, including threshold-search F1/recall; malformed, non-integer, negative, and wrong-total confusion matrices — all rejected. | `TestMetricOutputValidation`, `TestConfusionMatrixValidation` |
| 18 | Complete V1 evaluation: exact schema and 10,200/2,100/2,100 row counts, fixed candidate order, selected name inside the locked set, finite/ranged metrics, and full reproducibility — with no locked winner or superiority claim. | `TestV1Profile` |

## External integration

The complete suite requires real PostgreSQL and ran against the disposable
`postgres:16-alpine` container on localhost (non-default port, database and
user both suffixed `_test`; the URL is injected through `TEST_DATABASE_URL`
and never stored). Phase 11 itself performs no database or filesystem I/O;
the existing Phase 6 PostgreSQL suite remains the cross-phase integration
gate.

No dependency was added: the lockfile was not regenerated (`uv lock --check`
passes against the unchanged `pyproject.toml`).
