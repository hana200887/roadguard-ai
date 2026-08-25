# Phase 8 TDD Evidence — Chronological splitting and train-only preprocessing

## Source and scope

The journeys and acceptance criteria are derived from
[`docs/phase-plan.md`](../phase-plan.md) and the Phase 8 addendum in
[`docs/contracts.md`](../contracts.md). Phase 8 is limited to the locked
34/7/7 unique-date split of the exact Phase 7 feature frame and deterministic
preprocessing fitted exclusively on the training partition. Target joining,
lag/rolling features, EDA, models, tuning, artifacts, database writes, and API
work are out of scope.

## User journeys

- As a modelling workflow, I need disjoint, contiguous 34/7/7 partitions
  split by unique observation dates so no future date leaks into training.
- As a reviewer, I need encoders/scalers fitted on training statistics only,
  with validation/test restricted to transform, so selection and evaluation
  cannot influence fitted state.
- As an operator, I need forged or invalid feature frames to fail
  contextually before any split or transform.

## RED evidence

| Stage | Command | Genuine result |
| --- | --- | --- |
| No implementation | `uv run pytest tests/test_preprocessing.py -q` | RED: `ModuleNotFoundError: No module named 'roadguard.preprocessing'` — collection error, all 41 contract tests blocked. |
| After module created | `uv run pytest tests/test_preprocessing.py -q` | RED: 8 failed, 24 passed, 9 errors. Errors: `AttributeError: module 'roadguard' has no attribute 'split_chronologically'` (public exports missing); failures: `fit_preprocessor` required the full-frame row count instead of the 34-date train shape, the duplicate-column-label fixture did not produce a duplicate label, and the completeness test compared an incidental concat index type. |
| After boundary fixes | `uv run pytest tests/test_preprocessing.py -q` | RED: 1 failed, 40 passed — index-type-only difference in the completeness assertion. |
| Independent adversarial review | `uv run pytest tests/test_preprocessing.py -q` after adding reviewer regression tests | RED: 9 failed, 29 passed, 9 errors. Confirmed arbitrary future 34-date fitting, mismatched `DatasetSpec` month counts, a gapped monthly calendar, and non-canonical datetime units were accepted before hardening. |

## GREEN evidence

| Command | Result |
| --- | --- |
| `uv sync --frozen` | GREEN: locked environment synchronized; 33 packages checked, no lockfile change. |
| `uv run pytest tests/test_preprocessing.py tests/test_package.py -q` | GREEN: 56 passed in 36.78s. |
| `uv run pytest tests/test_features.py tests/test_package.py -q` | GREEN: Phase 7 regression tests unchanged. |
| Full suite with real disposable PostgreSQL (`postgres:16-alpine`; URL injected through `TEST_DATABASE_URL`, never stored) | GREEN: `uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80` → 672 passed in 319.09s; total combined coverage 94.65% (95.71% statements, 92.07% branches); `roadguard.preprocessing` 85.89% combined (89.19% statements, 79.71% branches). |
| `uv run ruff check .` | All checks passed. |
| `uv run ruff format --check .` | 43 files already formatted. |
| `uv run mypy src` | Success: no issues found in 23 source files. |
| `git diff --check` | clean (exit 0). |

## Guarantees covered

| # | What is guaranteed | Test |
| --- | --- | --- |
| 1 | Exact 34/7/7 unique-date boundaries and V1 row counts (10,200/2,100/2,100). | `test_exact_34_7_7_date_boundaries_and_rows`, `test_v1_split_exact_counts` |
| 2 | Partitions are disjoint, contiguous, and reproduce the complete input. | `test_partitions_disjoint_contiguous_and_complete` |
| 3 | Partitions are canonically sorted by key; shuffled input is invariant. | `test_partitions_canonically_sorted`, `test_input_row_order_invariance` |
| 4 | Caller input immutability across split, fit, and transform. | `test_caller_frame_not_mutated`, `test_fit_and_transform_do_not_mutate_inputs` |
| 5 | Exact Phase 7 schema enforcement (missing/extra/reordered/duplicate labels). | `TestInputBoundary` |
| 6 | Duplicate natural keys, malformed/tz-aware/non-midnight dates, nulls, non-finite values, invalid dtypes rejected. | `TestInputBoundary` |
| 7 | Incomplete/mismatched V1 observation grids and non-contiguous monthly calendars are rejected. | `test_incomplete_grid_missing_row_rejected`, `test_incomplete_grid_missing_date_rejected`, `test_spec_mismatch_rejected`, `test_spec_month_count_must_match_locked_48_date_split`, `test_monthly_calendar_gap_rejected` |
| 8 | Keys never become model features; keys returned separately. | `test_transformed_output_excludes_keys`, `test_forbidden_fields_never_become_features` |
| 9 | Fitting requires a complete provenance-checked split; arbitrary future-contaminated 34-date frames are rejected. | `test_fit_requires_train_partition` |
| 10 | One-hot encoding uses sorted training categories only; unseen categories encode as all-zero without schema change. | `test_one_hot_encoder_uses_train_categories_only`, `test_unseen_category_encoded_as_all_zeros` |
| 11 | Construction date becomes a deterministic epoch-day representation scaled with training statistics. | `test_construction_date_day_representation_and_train_scaling` |
| 12 | All scaling uses training statistics only. | `test_scaling_uses_train_statistics_only` |
| 13 | Changing validation/test cannot change fitted state or transformed train output. | `test_changing_validation_test_cannot_change_fitted_state`, `test_changing_validation_test_cannot_change_transformed_train` |
| 14 | Zero-variance training columns, including very large constants, transform to exact finite zero. | `test_zero_variance_training_column_remains_finite`, `test_large_zero_variance_training_column_transforms_to_exact_zero` |
| 15 | Deterministic feature names and float64 dtypes; repeated execution identical. | `test_deterministic_feature_names_and_dtypes`, `test_repeated_execution_identical` |
| 16 | V1 fit/transform shapes and finiteness. | `test_v1_fit_transform_shapes` |
| 17 | Non-string object values and wrong spec types fail contextually. | `test_non_string_object_value_rejected`, `test_split_rejects_non_spec`, `test_fit_rejects_non_spec` |
| 18 | Fitted state is canonical, immutable, bounded to registries, and finite; forged state fails closed. | `test_forged_fitted_state_is_rejected`, `test_finite_extreme_that_overflows_transform_is_rejected` |
| 19 | Segment identifiers and static master attributes preserve Phase 7 provenance during split and transform. | `test_malformed_segment_id_rejected`, `test_static_segment_attribute_drift_rejected`, `test_transform_rejects_static_attribute_drift` |

## External integration

The complete suite requires real PostgreSQL. It ran against a disposable
`postgres:16-alpine` container on localhost (non-default port, database and
user both suffixed `_test`):

```text
TEST_DATABASE_URL=<injected disposable PostgreSQL test URL; not stored>
uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80
672 passed in 319.09s; all 28 PostgreSQL integration tests passed
```

No dependency was added: preprocessing is implemented deterministically with
pandas/numpy, so `uv.lock` was not regenerated and no scikit-learn dependency
was introduced. The full-suite gate therefore used the locked environment
unchanged.
