# RoadGuard AI — Authoritative Phase Plan

This file is the versioned execution authority for RoadGuard AI.
`docs/contracts.md` defines data and behavioural semantics; this plan assigns
ownership, entry/exit criteria, and status. The README is an overview only
and cannot override either document.

## Status vocabulary

- **planned**: scope is frozen but no implementation starts.
- **active**: RED-first work is underway on a dedicated branch.
- **accepted-local**: all required local gates and independent review pass.
- **published**: the accepted commit is a fast-forward-safe descendant of the
  canonical remote `main` and has been pushed normally.

No phase may start until its predecessor is at least **accepted-local**. A
phase is not **published** merely because an equivalent patch exists on a
different remote history.

## Cross-phase non-negotiables

- Preserve the V1 profile, start-of-day target semantics, target separation,
  natural keys, and point-in-time availability rules in `docs/contracts.md`.
- Keep cost, material consumption, target values, future events, latent
  simulation fields, and caller-supplied feature vectors out of ML features.
- Fit learned preprocessing on train data only; validation selects; the test
  partition is evaluated once after selection is frozen.
- Each phase requires RED evidence, targeted tests, full tests with branch
  coverage at least 80%, formatting, linting, strict type checking, diff
  checking, secret/generated-artifact inspection, and an independent review
  with no P1/P2 finding.
- An external integration counts as complete only when it has actually run.
  A skip is recorded as incomplete, never as a pass.
- Never expand V1 into YOLO/CV, LLM/RAG, Kafka, Spark, Kubernetes, Airflow,
  Redis, microservices, or cloud infrastructure without a new approved plan.

## Phases

| # | Single responsibility | Required output | Status |
| --- | --- | --- | --- |
| 1 | Foundation and locked system contracts | Configuration and immutable V1 contract | published |
| 2 | Segment master and event engine | Deterministic segments, events, accident timeline | published |
| 3 | Causal observation generation | Clean observation-core frame | published |
| 4 | Event-derived target generation | Separate target frame | published |
| 5 | Raw corruption, validation, and safe cleaning | Validated `CleaningResult` | published |
| 6 | Transactional PostgreSQL persistence | Fixed schema and safe read/write boundary | published |
| 7 | Point-in-time feature registry and generation | Frozen target-free feature frame | published |
| 8 | Chronological splitting and train-only preprocessing | 34/7/7 split and fitted transforms | published |
| 9 | Exploratory analysis and data card | Read-only reproducible EDA evidence | published |
| 10 | Baseline supervised evaluation | Baseline classifier/regressor metrics | published |
| 11 | Advanced classification | Validation-selected classifier | published |
| 12 | Advanced regression | Validation-selected regressor | planned |
| 13 | Frozen selection, artifacts, and risk mapping | Manifested selected artifacts and risk output | planned |
| 14 | Material forecasting | Rolling-origin network-month forecast | planned |
| 15 | Maintenance optimization | Constraint-validated optimization recommendation | planned |
| 16 | Inference runtime | Artifact-bound internal inference pipeline | planned |
| 17 | Explainability | Frozen-model explanation output | planned |
| 18 | FastAPI service | Validated serving API with health/readiness semantics | planned |
| 19 | Reproducible release | Sequential dashboard, container, CI/docs/release gates | planned |

## Phase 7 acceptance contract

**Entry:** Phase 6 is accepted locally; the Phase 7 implementation receives a
complete `RepositoryExport` and `DatasetSpec`.

**Output:** `build_feature_frame` returns only the exact key and feature
columns in `docs/contracts.md` section 14, sorted by natural key. Targets,
event keys, history, costs, materials, latent fields, labels, splitting,
imputation, encoding, scaling, models, and database I/O are out of scope.

**Temporal acceptance:** equivalent shuffled input must produce the same
canonical output; altering a valid future event together with freshly
re-derived valid targets cannot alter any feature value. Event-derived
observation fields must match strictly-prior event keys before use.

**Boundary acceptance:** invalid/forged source data fails after fresh Phase 5
validation; caller-owned frames are unchanged; the output cannot contain a
forbidden column.

**Exit:** all cross-phase gates pass, Phase 7 has independent review approval,
and its evidence report records the RED and GREEN commands/results.

## Phase 8 acceptance contract

**Entry:** Phase 7 is accepted locally; the Phase 8 implementation receives
the exact target-free Phase 7 feature frame plus its `DatasetSpec`.

**Output:** `roadguard.preprocessing` splits by sorted unique observation
dates into fixed 34/7/7 partitions and fits deterministic one-hot encoders
and scaling statistics on the training partition only. The public workflow
returns the partitions (keys preserved, canonically sorted), an immutable
train-only fitted state, and transformed train/validation/test frames whose
keys stay separate from finite `float64` model features. Models, tuning,
artifacts, EDA, imputation, clipping, and any later-phase behavior are out of
scope.

**Temporal acceptance:** equivalent shuffled input produces the same
partitions; validation/test rows, unseen categories, and extreme values never
alter the fitted state or the transformed training output.

**Boundary acceptance:** forged or invalid feature frames (wrong schema,
dtypes, keys, dates, nulls, non-finite values, incomplete V1 grid) fail
contextually; caller-owned frames are unchanged; keys and forbidden fields
never become model features.

**Exit:** all cross-phase gates pass, Phase 8 has independent review approval,
and its evidence report records the RED and GREEN commands/results.

## Phase 9 acceptance contract

**Entry:** Phase 8 is accepted locally; the Phase 9 implementation receives a
complete Phase 6 `RepositoryExport`, the exact canonical Phase 8
`ChronologicalSplit`, and their matching `DatasetSpec`.

**Output:** `roadguard.eda` fresh-validates and reproduces the Phase 7 feature
frame and Phase 8 split, joins the separate targets to the training keys only,
and returns an immutable `EDAReport`. `render_data_card` turns only that report
into deterministic in-memory Markdown. The exact report schema, statistics,
ordering, fingerprint, and rendering rules are frozen in
`docs/contracts.md` section 16. Filesystem/database writes, plots, notebooks,
HTML/CSV/JSON artifacts, transformed features, fitting, models, tuning,
selection, and all Phase 10 behavior are out of scope.

**Temporal acceptance:** feature and target statistics use only the canonical
34-date training partition. Validation and test may contribute only their
names, row/date counts, and date boundaries; their feature values and target
values never influence a summary, correlation, fingerprint, or rendered data
card. Equivalent shuffled upstream inputs followed by canonical rebuilding
produce equal `EDAReport` values and byte-identical Markdown.

**Boundary acceptance:** forged/mismatched exports, specs, splits, keys,
schemas, dtypes, dates, or target relationships fail contextually;
structurally invalid or internally contradictory report objects cannot be
rendered. Caller-owned objects are unchanged. Rendering performs no I/O,
contains no generated timestamp or environment-specific path, and cannot
claim validation/test or model-performance evidence.

**Exit:** all cross-phase gates pass, Phase 9 has independent review approval,
and its evidence report records the RED and GREEN commands/results. No Phase
10 code, dependency, metric, or artifact may be included.

## Phase 10 acceptance contract

**Entry:** Phase 9 is published. The Phase 10 implementation receives a
complete Phase 6 `RepositoryExport`, the exact canonical Phase 8
`ChronologicalSplit`, its matching immutable train-only `PreprocessorFit`, and
their matching `DatasetSpec`.

**Output:** `roadguard.baselines.evaluate_baselines` fresh-validates the
export, reproduces the Phase 7 feature frame and Phase 8 split, fits the
canonical train-only preprocessor and rejects a mismatched supplied fit,
trains exactly one locked training-prior dummy classifier and one locked
training-median dummy regressor, selects the classifier threshold on validation
only, and returns immutable validation/test metrics.
The exact API, estimators, parameters, result schema, metric semantics,
determinism policy, and failure rules are frozen in `docs/contracts.md`
section 17.

**Temporal acceptance:** no model or preprocessing statistic may use
validation/test rows. Validation is used only for the locked classifier
PR-AUC/F1/recall evidence, threshold selection, and locked regression
MAE/RMSE evidence. The test partition is transformed and evaluated exactly
once, only after both estimators and the decision threshold are frozen. Test
features or targets cannot affect fitting, validation metrics, threshold
selection, or any returned provenance field.

**Boundary acceptance:** forged/mismatched exports, specs, splits, keys,
schemas, dtypes, dates, target relationships, non-finite values, invalid
preprocessor state, degenerate classification partitions, constant test
regression targets, invalid probabilities, or non-finite predictions fail
contextually. Caller-owned objects are unchanged; keys, targets, future event
keys, costs, materials, and latent fields never become model features.

**Exit:** all cross-phase gates pass, Phase 10 has independent review approval,
and its evidence report records genuine RED and GREEN commands/results. No
advanced model/tuning, artifact persistence, risk mapping, forecasting,
optimization, inference, explainability, service, dashboard, container, or
later-phase behavior may be included.

## Phase 11 acceptance contract

**Entry:** Phase 10 is published. The Phase 11 implementation receives a
complete Phase 6 `RepositoryExport`, the exact canonical Phase 8
`ChronologicalSplit`, its matching immutable train-only `PreprocessorFit`,
their matching `DatasetSpec`, and an exact immutable `RoadGuardConfig`.

**Output:** `roadguard.classification.evaluate_advanced_classifier`
fresh-validates the export, reproduces the canonical Phase 7 feature frame,
Phase 8 split, and train-only preprocessing fit, then fits exactly two locked
feature-dependent classification candidates on training data only. It records
their validation evidence, selects exactly one candidate and its threshold by
the fixed validation ranking, and returns immutable selected-model test
metrics. The exact API, candidate constructors, seed derivation, selection
ranking, result schema, metric semantics, determinism policy, and failure
rules are frozen in `docs/contracts.md` section 18.

**Temporal acceptance:** no candidate or preprocessing statistic may use
validation/test rows while fitting. Validation is the sole source of each
candidate threshold and all model-selection evidence. After the winning
candidate and threshold are frozen, only that candidate may receive the test
matrix, exactly once inside one private test stage. Test features or targets
cannot affect candidate fit state, candidate validation metrics, selected
name, threshold, feature schema, or row counts.

**Boundary acceptance:** forged/mismatched exports, specs, splits, fits,
config seeds, keys, schemas, dtypes, dates, target relationships, non-finite
values, invalid probabilities, invalid model output, invalid metric output, or
degenerate classification partitions fail contextually. Caller-owned
objects are unchanged; keys, targets, future event keys, costs, materials,
and latent fields never become model features. The evaluator neither loads
configuration nor reads environment variables.

**Exit:** all cross-phase gates pass, Phase 11 has independent review
approval, and its evidence report records genuine RED and GREEN
commands/results. No advanced regression, calibration, hyperparameter search,
cross-validation, artifact persistence, risk mapping, forecasting,
optimization, inference, explainability, service, dashboard, container, or
later-phase behavior may be included.
