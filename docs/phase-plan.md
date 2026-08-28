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
| 12 | Advanced regression | Validation-selected regressor | published |
| 13 | Frozen selection, artifacts, and risk mapping | Manifested selected artifacts and risk output | published |
| 14 | Material forecasting | Rolling-origin network-month forecast | active |
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

## Phase 12 acceptance contract

**Entry:** Phase 11 is published. The Phase 12 implementation receives a
complete Phase 6 `RepositoryExport`, the exact canonical Phase 8
`ChronologicalSplit`, its matching immutable train-only `PreprocessorFit`,
their matching `DatasetSpec`, and an exact immutable `RoadGuardConfig`.

**Output:** `roadguard.regression.evaluate_advanced_regressor` fresh-validates
the export, reproduces the Phase 7 feature frame, Phase 8 split, and train-only
preprocessing fit, then fits exactly two locked feature-dependent regression
candidates on training data only. It records validation MAE/RMSE for both,
selects exactly one candidate by the fixed validation ranking, and returns
immutable selected-model test MAE/RMSE/R-squared metrics. The exact API,
candidate constructors, seed derivation, selection ranking, result schema,
metric semantics, determinism policy, and failure rules are frozen in
`docs/contracts.md` section 19.

**Temporal acceptance:** no candidate or preprocessing statistic may use
validation/test rows while fitting. Validation is the sole source of all model
selection evidence. After the winning candidate is frozen, only that candidate
may receive the test matrix, exactly once inside one private test stage. Test
features or targets cannot affect candidate fit state, candidate validation
metrics, selected name, feature schema, or row counts.

**Boundary acceptance:** forged/mismatched exports, specs, splits, fits,
config seeds, keys, schemas, dtypes, dates, target relationships, non-finite
values, invalid predictions, invalid model output, invalid metric output, or a
constant test regression target fail contextually. Caller-owned objects are
unchanged; keys, targets, future event keys, costs, materials, and latent
fields never become model features. The evaluator neither loads configuration
nor reads environment variables.

**Exit:** all cross-phase gates pass, Phase 12 has independent review
approval, and its evidence report records genuine RED and GREEN
commands/results. No calibration, hyperparameter search, cross-validation,
artifact persistence, risk mapping, forecasting, optimization, inference,
explainability, service, dashboard, container, or later-phase behavior may be
included.

## Phase 13 acceptance contract

**Entry:** Phase 12 is published. The Phase 13 implementation receives a
complete Phase 6 `RepositoryExport`, the exact canonical Phase 8 split and
matching train-only preprocessor fit, their matching `DatasetSpec`, and an
exact immutable `RoadGuardConfig`.

**Output:** `roadguard.artifacts.persist_selected_artifacts` reproduces the
exact Phase 11 classifier and Phase 12 regressor train/validation selections in
one private orchestration, retains the two train-fitted winners without
refitting, publishes their manifested local bundle atomically under
`config.artifacts_dir`, and returns immutable manifest provenance plus the
selected classifier's canonical test risk rows. The exact API, dependencies,
serialization, fingerprints, files, manifest, path policy, idempotence,
collision handling, schemas, and failure rules are frozen in
`docs/contracts.md` section 20.

**Temporal acceptance:** both selections use validation evidence only and are
frozen before test feature transformation. Test is transformed once; only the
selected classifier receives it once. Its captured positive probabilities are
reused for risk output. No regressor or losing classifier sees test. Complete
lower-phase validation may inspect test-target cells solely to enforce the
locked repository contract; after that boundary no test target is projected,
joined, hashed, serialized, scored, or otherwise used. This target-free pass is
risk-output generation, not a second test evaluation: it computes no test
metric and cannot affect selection. Persisted winners remain the exact
train-fitted instances; train-plus-validation refitting is forbidden.

**Artifact acceptance:** publication creates only `preprocessor.json`,
`classifier.joblib`, `regressor.joblib`, `test-risk.jsonl`, and `manifest.json`
inside the fixed digest-named directory. Payload sizes/hashes, training,
validation-selection, and target-free test-risk-input fingerprints, selected
names/threshold/seeds, feature schema, split dates/counts, runtime versions,
and risk bands are canonical and manifested. Publication is visibility-atomic,
byte-reproducible for the same locked environment, idempotent only for an exact
existing bundle, and fail-closed for tampering/collision. Absolute paths,
timestamps, host/user details, secrets, test targets, unselected models, and
mutable objects are never returned or persisted.

**Boundary acceptance:** exact input types, canonical export/split/fit
reproduction, target/key alignment, forbidden-feature exclusion, probability
shape/range, score/band mapping, UNC/network-syntax rejection, anchor-aware
containment, operation-boundary symlink/junction/reparse/changed-identity
rejection, fixed inventory,
serialization/write/hash/fsync/rename failure, concurrent collision, caller
immutability, and sanitized errors are tested adversarially. Phase 13 never
deserializes a model. Race-free protection against a concurrently privileged
local process is outside scope.

**Exit:** all cross-phase gates pass, Phase 13 has independent correctness,
security, temporal, provenance, and filesystem review approval, and its TDD
evidence records genuine RED/GREEN results. Model loading/registry, MLflow,
database prediction writes, calibration, train-plus-validation refitting,
forecasting, optimization, inference runtime, explainability, API/dashboard,
container, and all Phase 14+ work remain out of scope.

## Phase 14 acceptance contract

**Entry:** Phase 13 is published. The Phase 14 implementation receives a
complete Phase 6 `RepositoryExport`, exact realized maintenance-history rows,
and a fresh validated `DatasetSpec`. It reruns the locked cleaned-data
validation to authenticate the maintenance-event keys. For every event inside
the forecast history window, one matching realized-history row is mandatory;
absence cannot be guessed to mean zero consumption.
The authoritative PostgreSQL producer returns the export and history together
from one read-only repeatable-read snapshot; separately timed queries are not
an accepted runtime provenance path.

**Output:** `roadguard.forecasting.forecast_materials` returns immutable
per-material rolling-origin evidence and exactly one next-month
network-aggregate forecast for each of the four locked materials. The exact
API, schemas, candidates, metrics, timeline, completeness policy,
fingerprinting, and failure rules are frozen in `docs/contracts.md` section
21. Segment-level output and direct PostgreSQL persistence are excluded.

**Temporal acceptance:** the forecast timeline is independent of the
supervised 34/7/7 split. Each fold uses an expanding prefix strictly before
its target month. Both candidates see only validation origins; selection is
independent per material and freezes before the seven final origins. Only the
selected candidate sees those seven origins in one walk-forward pass. Prior
test actuals may enter a later test origin only after their month has elapsed,
which models operational rolling-origin availability and cannot alter the
already frozen selection. The final one-month forecast uses all accepted
history only after frozen test metrics are complete.

**Forecast acceptance:** the initial prefix is 24 months. V1 therefore has 17
validation origins, 7 frozen test origins, a 48-month by four-material dense
history, and four forecasts for the month immediately after the V1 cutoff.
Candidates are exactly seasonal-naive lag 12 and trailing-three-month mean.
Validation MAE, then validation RMSE, then candidate order select per
material. Test MAE/RMSE are reported per material; errors in unlike physical
units are never pooled. Negative or non-finite inputs, predictions, metrics,
or forecasts fail rather than being clipped.

**Boundary acceptance:** exact export/frame/spec types and schemas, fresh
lower-phase validation, complete event-to-history coverage inside the window,
natural keys, exact V1-start observation calendar, accounting values,
canonical zero-filled monthly aggregation, fold membership, prefix-only
candidate calls, selection freeze, one
selected-only test pass, next-period keys, fingerprints,
deterministic/shuffled equivalence, future/cost/segment/target isolation,
caller immutability, hostile values, and sanitized errors are tested
adversarially. The forecast workflow performs no database or external I/O;
its additive repository adapter performs one read-only snapshot and no write.
Phase 14 performs no environment, configuration, filesystem, artifact,
model-loading, or RNG operation.

**Exit:** all cross-phase gates pass, Phase 14 has independent correctness,
temporal, completeness, provenance, and numerical review approval, and its
TDD evidence records genuine RED/GREEN results. Database forecast writes,
segment forecasts, extra candidates, tuning, maintenance optimization,
inference, explainability, API/dashboard, container, and all Phase 15+ work
remain out of scope.
