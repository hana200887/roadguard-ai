# Phase 9 TDD Evidence — Train-only EDA and deterministic data card

## Source and scope

The journeys and acceptance criteria are derived from
[`docs/phase-plan.md`](../phase-plan.md) and the Phase 9 addendum in
[`docs/contracts.md`](../contracts.md) section 16. Phase 9 is limited to
read-only descriptive evidence: a fresh-validated Phase 6 `RepositoryExport`,
the canonical Phase 8 split, Decimal-exact training statistics, a SHA-256
training fingerprint, and deterministic in-memory Markdown. Preprocessing
fit/transform, models, thresholds, artifacts, plots, notebooks, filesystem,
database and network writes are out of scope. No dependency was added, so
`uv.lock` is unchanged.

## User journeys

- As a modelling workflow, I need descriptive evidence calculated from the
  canonical 34-date training partition only, with validation/test restricted
  to row/date inventory metadata.
- As a reviewer, I need exact Decimal arithmetic (no pandas/NumPy reductions),
  a canonical training fingerprint, and byte-identical Markdown across
  repeated and shuffled-source calls.
- As an operator, I need forged exports, splits, reports and injected dynamic
  text rejected contextually.

## RED evidence

| Stage | Command | Genuine result |
| --- | --- | --- |
| No implementation | `uv run pytest tests/test_eda.py tests/test_package.py -q` | RED: `ModuleNotFoundError: No module named 'roadguard.eda'` — collection error, all contract tests blocked. |
| After first implementation | `uv run pytest tests/test_eda.py -q` | RED: 12 failed, 40 passed, 38 errors. Errors: missing `__init__` exports (`AttributeError: module 'roadguard' has no attribute 'split_chronologically'`-style). Failures: the training join merged keys only (missing feature columns), the placeholder known-vector constants, and the validation/test-target mutation broke fresh validation (`target_event_inconsistency`). |
| After join fix and canonical constants | `uv run pytest tests/test_eda.py -q` | RED: 4 failed, 86 passed — the independent fingerprint reconstruction did not filter to training rows, and the target-mutation test still used an invalid mutation. |

## GREEN evidence

| Command | Result |
| --- | --- |
| `uv run pytest tests/test_eda.py tests/test_package.py -q` | GREEN: 132 passed in 61.42s (130 EDA contract tests + 2 package-surface). |
| Full suite with real disposable PostgreSQL (`postgres:16-alpine` container, `TEST_DATABASE_URL=postgresql+psycopg://roadguard_test:roadguard@127.0.0.1:54329/roadguard_test`) | `uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80` → **802 passed in 799.68s**; total coverage **94.76%** (statements 95.8%, branches 92.4%); `roadguard.eda` **95% statements / 92.9% branches** (remaining lines are defensive trap/guard branches). |
| `uv run ruff check .` | All checks passed. |
| `uv run ruff format --check .` | Already formatted. |
| `uv run mypy src` | Success: no issues found in 24 source files. |
| `git diff --check` | clean (exit 0). |

## P2 remediation after independent review

An independent review found two renderer-validation gaps: a forged finite
numeric mean outside its declared minimum/maximum bounds could render, and an
integer `positive_rate` could reach formatting rather than fail as `EDAError`.

| Stage | Command | Genuine result |
| --- | --- | --- |
| RED | `uv run pytest tests/test_eda.py -q -k "numeric_mean_outside_summary_bounds or non_float_positive_rate"` | 2 failed: the forged mean rendered; the integer rate rendered without `EDAError`. |
| GREEN | Same command after the renderer guards | 2 passed, 130 deselected. |
| Targeted | `uv run pytest tests/test_eda.py tests/test_package.py -q` | 134 passed in 29.17s. |
| Full gate | `uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80` against disposable PostgreSQL | 804 passed in 393.58s; 95.81% line coverage and 92.23% branch coverage. |

The renderer now rejects a numeric mean outside `[minimum, maximum]` and
requires `ClassificationBalance.positive_rate` to be an exact finite `float`
before formatting. These guards apply equally to feature and regression
numeric summaries and preserve the Phase 9 read-only scope.

## Guarantees covered

| # | What is guaranteed | Test class |
| --- | --- | --- |
| 1 | Exact `roadguard.eda.__all__`, package-root exports, all Phase 1–8 exports preserved. | `TestPublicSurface` |
| 2 | Frozen dataclasses with exact field names/order and built-in scalar/tuple-only storage. | `TestPublicSurface` |
| 3 | Wrong types, lookalikes, mismatched spec, forged/mutated exports, schema/dtype/key/date failures, missing target rows, supplied-split mismatches. | `TestInputValidation` |
| 4 | Caller frames unchanged on success and failure. | `TestInputValidation` |
| 5 | Validation/test feature and target mutations cannot change summaries, correlations, fingerprint or Markdown; shuffled upstream input produces equal reports and byte-identical Markdown. | `TestTemporalLeakage` |
| 6 | Exact registry order for numeric/categorical/datetime features; population std (ddof=0); linear quartile vectors; strict IQR boundaries; classification balance; 30 correlations, feature-major, no duplicates. | `TestDescriptiveCalculations` |
| 7 | Constant columns produce exact zero std and `zero_variance=True`; non-constant computed-zero variance fails closed; constant sequences yield `pearson_r=None`; known Pearson vectors. | `TestDescriptiveCalculations` |
| 8 | Huge finite constant and non-constant float64 values; unrepresentable Decimal→float raises `EDAError`; rendering near both float64 limits. | `TestAdversarialNumeric` |
| 9 | Lowercase SHA-256 digest; known-vector digest verified against an independent canonical-JSON reconstruction; negative-zero float hex normalization; validation/test rows not hashed. | `TestFingerprint` |
| 10 | Exact known-vector Markdown for the 1-segment pipeline; round-half-even six-decimal floats; `not-defined` correlations; one trailing newline; no blank lines inside tables; global Decimal context independence; injection probes (HTML, links, pipes, line breaks); invalid version/digest/ordering/counts/rates/totals/forged names rejected. | `TestRenderer` |
| 11 | No filesystem writes (open() monkeypatched); no timestamp/path fragments; V1 report builds and renders. | `TestSideEffects` |

## External integration

The complete suite requires real PostgreSQL and ran against a disposable
`postgres:16-alpine` container on localhost (non-default port, database and
user both suffixed `_test`). Full-suite result is recorded in the handoff
report.

The lockfile was not regenerated: no dependency was added.
