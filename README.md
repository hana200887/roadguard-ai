# RoadGuard AI

Predictive maintenance and risk intelligence for road infrastructure.

## Status

The versioned execution authority is [the Phase plan](docs/phase-plan.md);
the cross-phase semantic authority is [the system contract](docs/contracts.md).
The README is an overview and does not override either document.

Phase 1 (project foundation and system contracts) is implemented:

- Python 3.12 package scaffold (`src` layout) managed with `uv`.
- Locked V1 data and ML contract (`roadguard.contracts.V1Contract`): 300
  segments, 48 monthly observations per segment, 14,400 observations,
  70/15/15 chronological split locked to 34/7/7 dates, 30-day maintenance
  window, risk bands locked to exactly 0-30 / 31-60 / 61-80 / 81-100. The
  values are enforced and cannot be changed by YAML or environment
  configuration.
- Central typed runtime configuration: validated Pydantic model loaded from
  built-in defaults, an optional YAML file, and `ROADGUARD_*` environment
  variables. Unknown `ROADGUARD_*` variables are rejected.
- Pure target-semantics helpers (`roadguard.targets`): days until the next
  maintenance event (on or after the observation date, past events
  rejected) and the inclusive 30-day maintenance window.
- Probability-to-risk-score helper (`roadguard.risk`): deterministic
  decimal ROUND_HALF_UP (`Decimal(str(p)) * 100` quantized to 1), finite
  and in-range probabilities only, booleans rejected.
- System contract documentation in `docs/contracts.md` (dataset tables and
  columns, target semantics, 34/7/7 time split policy, model selection
  contracts, feature availability, artifacts and online inference, seed
  policy, generation methodology).

Phase 2 (segment master and maintenance-event engine) is implemented:

- `roadguard.segments.generate_segments`: deterministic static master for
  the V1 network with registry-based stable string IDs (e.g.
  `QL01-KM134-135`), contract columns, and latent simulation state
  (traffic base, heavy-vehicle ratio, weather exposure, deterioration,
  accident propensity, initial condition). `road_length_km` equals the KM
  marker span and is stable across seeds.
- `roadguard.events.generate_maintenance_events`: maintenance-event
  simulation as a monthly Bernoulli process (at most one event per
  segment-month) whose pure hazard (`roadguard.events.monthly_hazard`,
  `base_rate` validated and functional) depends on asset age, traffic,
  heavy-vehicle and weather exposure, condition deterioration, trailing
  accident history, and previous maintenance (renewal suppression via the
  pure `month_transition` state helper). Events never precede construction
  dates; months are simulated through the future-buffer horizon (longer
  buffers extend history, shorter ones are prefixes), and every segment is
  guaranteed a next event after the final observation date by continuing
  the hazard process (documented cap and explicit failure).
- `roadguard.events.generate_accident_timeline`: deterministic monthly
  accident counts per segment, reusable by later phases; hazard uses only
  past accidents.
- Per-segment RNG streams (`SeedSequence([seed, segment_id])`), so segment
  table row order never changes the output; same seed reproduces identical
  frames.

Phase 3 (observation-core synthetic generation) is implemented:

- `roadguard.observations.generate_observations`: the clean, complete
  `road_observations` table (16 documented columns; 14,400 V1 rows) with
  start-of-day snapshot semantics: as-of monthly traffic/weather aggregates,
  exact `road_age_days`, maintenance history strictly before `t`
  (`NEVER_MAINTAINED_DAYS_CAP = 3650` fallback), Phase 2 condition-replay
  scores, and literal 30/365-day accident windows from deterministically
  expanded monthly accident counts.
- Clean-core boundary: no missingness, invalid values, outliers, cleaning
  or imputation; no targets, cost, materials, anomaly flags, or engineered
  features; in-memory DataFrames only.
- Documented Phase 3 RNG namespaces (`SeedSequence([seed, segment_key,
  OBSERVATION_RNG_NAMESPACE, stream])`), row-order independent and isolated
  from Phase 2 streams.
- No targets, anomalies, material quantities or models yet;
  targets are never used during generation.

Phase 4 (event-derived supervised targets) is implemented:

- `roadguard.targets.derive_observation_targets`: derives
  `days_until_maintenance` and `maintenance_within_30_days` from the actual
  Phase 2 maintenance-event keys (first event on-or-after the snapshot;
  exact 0/30/31-day boundaries; no RNG). Missing tail events raise a
  contextual error instead of silent right censoring. Targets are returned
  in a frame physically separate from observation features, never containing
  cost, materials, static segment columns, or the internal
  `next_maintenance_date` state; future labels are forbidden model features.
- No models, API or dashboard yet.

Phase 5 (raw-data corruption, validation and safe cleaning) is implemented:

- `roadguard.data_quality.inject_observation_corruption`: deterministic
  corrupted-raw representation with controlled missing values (weather
  columns only), domain-valid outliers (`traffic_volume`, `rainfall_mm`) and
  exact duplicate rows, with an exact `CorruptionManifest` and a documented
  Phase 5 RNG namespace (row-order invariant, seed-controlled).
- `validate_raw_dataset` / `validate_cleaned_dataset`: staged schema, dtype,
  key, domain, cadence, cross-field and target/event validation producing a
  deterministic `ValidationReport` (severity, code, table, column, row key).
- `clean_raw_dataset`: removes exact duplicate rows, rejects conflicting
  keys, forward-fills permitted weather missingness from the same segment's
  strictly earlier values only (no global statistics), preserves outliers,
  and returns public frames (segments without latent simulation columns).
- Targets and maintenance events never enter the observation frame; latent
  segment fields never cross the cleaned boundary.
- No feature engineering, splitting, scaling, statistical imputation or
  models yet.
- Test harness: `pytest` with `pytest-cov` (branch coverage, enforced
  `fail_under = 80`), `ruff`, and `mypy` (strict).

Phase 6 (transactional PostgreSQL persistence) is implemented:

- A fixed `roadguard` schema with seven natural-key tables:
  `road_segments`, `road_observations`, `maintenance_events`,
  `observation_targets`, `maintenance_history`, `predictions`, and
  `material_forecasts`. Observation features and future-derived targets
  remain physically separate.
- `load_cleaning_result` deep-copies and freshly revalidates a Phase 5 result
  against an explicit `DatasetSpec`, then persists it in one transaction.
  Replays are insert-or-verify: identical and concurrent rows are idempotent,
  while a different row under the same natural key fails and rolls back the
  entire load. Realized cost/material rows are optional but, when supplied,
  must be complete; values are never fabricated from key-only events.
- `PostgresRepository` provides deterministic physical exports, point-in-time
  segment history (observations through `t`, maintenance events strictly
  before `t`), and monthly material aggregation from realized maintenance
  history only. Queries are built with bound SQLAlchemy expressions.
- Multi-query reads run as read-only `REPEATABLE READ` snapshots. Schema
  initialization verifies all seven existing tables and fails on drift rather
  than silently accepting a partial or incompatible schema.
- Phase 6 requires PostgreSQL with the synchronous `psycopg` driver. The URL
  is runtime-only, optional in configuration, and stored as a masked secret.

Phase 7 (point-in-time feature registry and generation) is published:

- `roadguard.features.build_feature_frame` accepts a complete Phase 6
  `RepositoryExport`, fresh-validates it, and returns the frozen target-free
  feature frame in canonical key order and dtypes.
- Event-derived maintenance features are checked against strictly-prior
  `maintenance_events`; targets, future event keys, cost/material facts, and
  latent fields cannot enter the frame.
- The phase deliberately does not split, impute, encode, scale, or train.

Phase 8 (chronological splitting and train-only preprocessing) is published:

- `roadguard.preprocessing.split_chronologically` splits the exact Phase 7
  feature frame into fixed 34/7/7 unique-date partitions with canonical key
  order and V1-exact row counts.
- `fit_preprocessor` accepts the complete provenance-checked split, verifies
  its chronological membership, and fits one-hot categories and scaling
  statistics from the canonical training partition only; `transform` applies
the immutable fitted state without refitting, keeping partition keys
  separate from finite `float64` model features.

Phase 9 (train-only EDA and deterministic data card) is published:

- `roadguard.eda.build_eda_report` fresh-validates the complete Phase 6
  export, rebuilds the Phase 7 feature frame and Phase 8 split, and computes
  Decimal-exact descriptive statistics, quartiles, IQR outliers, target
  correlations and a SHA-256 training fingerprint from the canonical 34-date
  training partition only.
- `render_data_card` renders the immutable `EDAReport` as deterministic
  in-memory Markdown with no timestamps, paths, later-partition statistics or
  I/O.
- The phase does not fit or apply preprocessing, train models, or persist
  artifacts.

Phase 10 baseline-supervised evaluation is published. It provides a
deterministic training-prior dummy classifier and training-median dummy
regressor, validation-only threshold selection, one frozen test evaluation,
and immutable metrics. Feature-dependent models and other advanced models,
model artifacts, API, dashboard, and Docker services remain future phases.

Phase 11 advanced classification is published. It provides two locked
feature-dependent candidates (L2-penalized logistic regression and histogram
gradient boosting) with derived per-candidate seeds, validation-only
selection, one frozen selected-model test evaluation, and immutable metrics.
Calibration, hyperparameter search, artifact persistence, and all later-phase
work remain outside Phase 11.

Phase 12 advanced regression is published. It provides locked Ridge-SVD and
histogram-gradient-boosting candidates, train-only fitting, validation
MAE/RMSE selection, one selected-only frozen test evaluation, and immutable
metrics. Artifact persistence, risk mapping, forecasting, optimization, and
all Phase 13+ work remain outside Phase 12.

Phase 13 frozen selection, artifact publication, and risk mapping is published.
It provides one private
train/validation selection orchestration, exact train-fitted winner retention,
atomic content-addressed local artifacts, canonical manifests, and target-free
test risk rows. Model loading/registry, live inference, optimization,
explainability, and all Phase 15+ work remain future phases.

Phase 14 network-month material forecasting is published. It provides
complete realized-history evidence, a separate expanding rolling-origin
timeline, deterministic per-material selection, one frozen test pass, and
four next-month forecast rows. Forecast persistence, maintenance optimization,
inference, explainability, and all Phase 15+ work remain future phases.

The Phase 15 maintenance-prioritization contract is now frozen; implementation
is underway. It defines an exact budget-only, single-date, offline
evaluation over Phase 13 risk rows authenticated against a separately trusted
manifest digest and explicit caller-asserted prospective cost-scenario values.
Phase 15 does not authenticate cost provenance. Phase 14 forecasts are not
treated as inventory constraints; fresh inference, persistence, serving, and
all Phase 16+ work remain future.

## Quickstart

```powershell
uv sync --frozen
uv run pytest --cov=roadguard --cov-branch --cov-report=term-missing --cov-fail-under=80
uv run ruff check .
uv run mypy src
```

## Configuration

Only runtime settings are configurable: `env`, `seed`, `data_dir`,
`artifacts_dir`, and the optional masked `database_url`. The V1 data/ML
contract is locked and cannot be overridden.

Configuration is resolved in this order (later wins):

1. Built-in defaults (documented in `src/roadguard/config.py`).
2. YAML file given as `load_config(path)` argument or via the
   `ROADGUARD_CONFIG_PATH` environment variable (reserved control variable).
3. Environment variables named `ROADGUARD_<FIELD>` (for example
   `ROADGUARD_SEED=7`).

Direct Pydantic model construction raises `pydantic.ValidationError` for
invalid or unknown settings. `load_config` instead raises
`roadguard.ConfigError` for invalid runtime values, unreadable or malformed
files, and unsupported `ROADGUARD_*` environment variable names; its public
errors never echo configuration values such as database credentials. Boolean
values are rejected for numeric fields; numeric strings such as
`ROADGUARD_SEED=42` are accepted. Phase 1 reads process environment variables
only; `.env` files are not parsed automatically. See `.env.example` for
supported variable names and safe example values.

## Contracts

All cross-phase guarantees (dataset schema, target semantics, time split
policy, feature availability, artifact/inference contracts, and the
reproducibility seed policy) are defined in `docs/contracts.md`.
