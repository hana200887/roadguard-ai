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
| 1 | Foundation and locked system contracts | Configuration and immutable V1 contract | accepted-local |
| 2 | Segment master and event engine | Deterministic segments, events, accident timeline | accepted-local |
| 3 | Causal observation generation | Clean observation-core frame | accepted-local |
| 4 | Event-derived target generation | Separate target frame | accepted-local |
| 5 | Raw corruption, validation, and safe cleaning | Validated `CleaningResult` | accepted-local |
| 6 | Transactional PostgreSQL persistence | Fixed schema and safe read/write boundary | accepted-local |
| 7 | Point-in-time feature registry and generation | Frozen target-free feature frame | active |
| 8 | Chronological splitting and train-only preprocessing | 34/7/7 split and fitted transforms | planned |
| 9 | Exploratory analysis and data card | Read-only reproducible EDA evidence | planned |
| 10 | Baseline supervised evaluation | Baseline classifier/regressor metrics | planned |
| 11 | Advanced classification | Validation-selected classifier | planned |
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
canonical output; altering valid future targets or future events cannot alter
any feature value.

**Boundary acceptance:** invalid/forged source data fails after fresh Phase 5
validation; caller-owned frames are unchanged; the output cannot contain a
forbidden column.

**Exit:** all cross-phase gates pass, Phase 7 has independent review approval,
and its evidence report records the RED and GREEN commands/results.
