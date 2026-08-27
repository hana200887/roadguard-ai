# Phase 10 TDD Evidence — Baseline supervised evaluation

## Source and scope

The journeys and acceptance criteria are derived from
[`docs/phase-plan.md`](../phase-plan.md) and the Phase 10 addendum in
[`docs/contracts.md`](../contracts.md) section 17. Phase 10 establishes one
deterministic learned baseline per supervised target:
`DummyClassifier(strategy="prior")` (public name `dummy_prior`) and
`DummyRegressor(strategy="median")` (public name `dummy_median`). Both are
deterministic and use no RNG, so Phase 10 adds no seed argument or RNG
namespace. The phase fresh-validates the export, rebuilds the Phase 7 feature
frame and Phase 8 split, refits the train-only preprocessor, selects the
classifier threshold on validation only, evaluates the frozen test partition
exactly once in one private stage, and returns immutable metrics. Advanced
models, tuning, calibration, artifacts, risk mapping, forecasting,
optimization, inference, and all Phase 11+ behavior are out of scope.

## Dependency change (locked by section 17)

`scikit-learn>=1.5,<2` was added to `pyproject.toml`; `uv.lock` resolves
`scikit-learn 1.9.0` plus transitive `joblib`, `scipy`, `threadpoolctl`, and
`narwhals`. This is the only new runtime dependency and is the locked
dependency of the Phase 10 contract.

## User journeys

- As a modelling workflow, I need deterministic baseline metrics for both
  supervised targets without touching the frozen test partition during
  selection.
- As a reviewer, I need fit-on-train-only, validation-only threshold
  selection, a single frozen test stage, and no RNG or seed surface.
- As an operator, I need forged exports, splits, fits, degenerate partitions,
  and invalid estimator output to fail contextually with sanitized messages.

## RED evidence

| Stage | Command | Genuine result |
| --- | --- | --- |
| No implementation | `uv run pytest tests/test_baselines.py -q` | RED: `ModuleNotFoundError: No module named 'roadguard.baselines'` — collection error, all contract tests blocked. The Phase 10 public module/API is absent, not a broken fixture or missing dependency. |
| After first implementation | `uv run pytest tests/test_baselines.py tests/test_package.py -q` | RED: 15 failed, 35 passed, 6 errors. The target-frame join indexed target columns with a tuple (pandas MultiIndex lookup), the constant-test-target and single-class-test fixtures were unreachable behind earlier degenerate-partition checks, and a validation-event mutation broke the strictly-prior feature provenance check. |
| After boundary fixes | `uv run pytest tests/test_baselines.py tests/test_package.py -q` | RED: 4 failed, 52 passed — crafted exports still placed events inside the test window and the mutated event was not the last strictly-prior event for later test observations. |
| DeepSeek handoff before independent review | `uv run pytest tests/test_baselines.py tests/test_package.py -q` | GREEN at that point: 56 passed in 41.98s (54 baseline contract tests + 2 package-surface). This is retained as historical pre-review evidence, not the final state. |
| Codex review — error boundaries | Focused forged-input, estimator arithmetic, and scalar-conversion tests | RED: 9 failed, 54 deselected. Exact dataclass instances with invalid nested fields leaked raw `AttributeError`/`ValueError`; expected estimator arithmetic and non-numeric prediction values were not sanitized. |
| Codex review — fitted state | Focused non-finite fitted-state tests | RED: 2 failed, 64 deselected. Non-finite classifier prior and regressor median state were accepted instead of rejected. |
| Final after fixes and independent re-review | `uv run pytest tests/test_baselines.py tests/test_package.py -q` | GREEN: **68 passed in 47.00s** (66 baseline contract tests + 2 package-surface). Both reviewers reported 0 remaining P1/P2 in code or tests. |

## GREEN evidence

| Command | Result |
| --- | --- |
| `uv sync --frozen` | GREEN: locked environment checked; 38 packages, no lockfile drift. |
| `uv run pytest tests/test_baselines.py tests/test_package.py -q` | GREEN: **68 passed in 47.00s**. |
| Full suite with real disposable PostgreSQL (`postgres:16-alpine` container, `TEST_DATABASE_URL` injected through the environment, never stored) | GREEN: `uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80` → **875 passed in 604.77s**; total coverage **94.20%** (branch coverage included; `roadguard.baselines` **88%** — remaining lines are defensive validation/sanitization branches). |
| `uv run ruff check .` | All checks passed. |
| `uv run ruff format --check .` | 50 files already formatted. |
| `uv run mypy src` | Success: no issues found in 26 source files. |
| `git diff --check` | clean (exit 0). |

## Guarantees covered

| # | What is guaranteed | Test |
| --- | --- | --- |
| 1 | Exact module `__all__` (8 symbols), package-root exports, constants `roadguard.phase10.v1` / `dummy_prior` / `dummy_median`. | `TestPublicSurface` |
| 2 | No seed or RNG API; signature is exactly `(dataset, split, fit, spec)`; no public frozen-test evaluation function or cross-call state. | `TestPublicSurface`, `test_repeated_calls_identical_no_cross_call_state` |
| 3 | Frozen dataclasses with the exact field names and order from section 17; only built-in scalars and tuples stored. | `TestFrozenSchema` |
| 4 | Locked estimator behavior: classifier equals the training prior, regressor equals the training median; hard labels always derived from `predict_proba` plus the frozen threshold. | `TestEstimatorContracts`, `TestKnownVectors` |
| 5 | Known-vector metrics: hand-computed threshold `2/34`, PR-AUC `6/7`, F1 `12/13`, confusion `((0, 1), (0, 6))`, ROC-AUC `0.5`; full field-for-field match against an independent direct-scikit-learn reference for crafted and generated exports. | `TestKnownVectors` |
| 6 | Threshold contract: candidates are `0.0`, `1.0` and every distinct finite validation probability; ranking is higher validation F1, then higher recall, then higher threshold (F1-primary, recall tie-break, threshold residual tie-break exercised on crafted vectors). | `TestThresholdRanking` |
| 7 | Train-only fit and validation-only threshold selection; a spy trace proves one fit per estimator on the exact canonical 34-row train `x/y`, validation prediction before one private test stage, and one test prediction per estimator. Validation/test mutations demonstrably change classification labels without crossing the frozen boundaries. | `TestEstimatorContracts`, `TestTemporalLeakage` |
| 8 | Canonical determinism: shuffled upstream frames followed by canonical rebuilding produce equal results. | `test_shuffled_upstream_inputs_produce_equal_results` |
| 9 | Caller-owned export frames, split partitions, and fitted state remain unchanged on success and failure. | `TestCallerImmutability` |
| 10 | Wrong top-level types, lookalikes, and subclasses raise `TypeError` before field access. | `TestInputValidation` |
| 11 | Forged split membership/dates, invalid nested export/split frame types, forged `PreprocessorFit` values or field types, mismatched spec, and missing/duplicate/forged targets fail contextually. | `TestForgedInputs`, `TestTargetAlignment` |
| 12 | Expected lower-phase failures are translated to `BaselineEvaluationError` with fixed sanitized messages that contain no raw values, row data, paths, or error codes. | `TestSanitizedErrors` |
| 13 | Keys, targets, future event keys, costs, materials, and latent fields never become model features; `feature_columns` equals `PreprocessorFit.transformed_feature_columns`. | `TestForbiddenFields` |
| 14 | Estimator arithmetic failures, non-finite fitted prior/median state, non-numeric prediction scalars, invalid/non-finite or out-of-range probabilities, malformed shapes, unexpected class order, and non-finite regression predictions are rejected with sanitized errors. | `TestEstimatorOutputValidation` |
| 15 | Single-class train/validation/test partitions and constant test regression targets fail contextually (train/validation checks before selection; both test checks only inside the frozen test stage, regression-constant check ordered first so it stays reachable). | `TestDegeneratePartitions` |
| 16 | Complete V1 evaluation: exact row counts 10,200/2,100/2,100, locked schema, finite/ranged metrics, threshold equals the training prior, and full reproducibility — with no hard-coded performance superiority requirement. | `TestV1Profile` |

## External integration

The complete suite requires real PostgreSQL and ran against the disposable
`postgres:16-alpine` container on localhost (non-default port, database and
user both suffixed `_test`; the URL is injected through `TEST_DATABASE_URL`
and never stored). Phase 10 itself performs no database or filesystem I/O;
the existing Phase 6 PostgreSQL suite remains the cross-phase integration
gate.

The lockfile was intentionally regenerated once: the addition of
`scikit-learn>=1.5,<2` is the locked Phase 10 dependency change.
