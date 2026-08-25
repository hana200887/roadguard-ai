# RoadGuard AI - System Contracts

This document defines the cross-phase guarantees for RoadGuard AI. Phase 1
implements the configuration (`src/roadguard/config.py`), the locked
contract models (`src/roadguard/contracts.py`), the target-semantics helpers
(`src/roadguard/targets.py`) and the risk-score helper
(`src/roadguard/risk.py`); the data, model, and inference phases must
implement the rest as specified here.

## 0. Locked V1 profile

The V1 production profile is fixed and enforced by `V1Contract`:

- **300** road segments.
- **48** monthly observations per segment.
- **14,400** observations in total.
- Chronological **70 / 15 / 15** train / validation / test split of the 48
  unique observation dates, giving locked date counts **34 / 7 / 7**
  (`train_date_count`, `validation_date_count`, `test_date_count`). The
  largest-remainder allocation is:
  - `48 * 0.70 = 33.6` -> floor 33
  - `48 * 0.15 = 7.2` -> floor 7
  - `48 * 0.15 = 7.2` -> floor 7
  - floors sum to 47; the remaining date goes to the train partition, giving
    **34 / 7 / 7** (and `34 + 7 + 7 = 48`).
- **30**-day maintenance window (`maintenance_window_days`).
- Risk bands locked to exactly **LOW 0-30, MEDIUM 31-60, HIGH 61-80,
  CRITICAL 81-100** (inclusive, contiguous). Any other layout is rejected,
  even one that is itself contiguous and covers 0-100.

These values are **not configurable**. Production configuration (YAML or
`ROADGUARD_*` environment variables) cannot change them: the keys are not
fields of the runtime configuration and are rejected if supplied. Small
dataset sizes used by unit tests or the generator must use the separate
`DatasetSpec` model, which is never loaded as the production profile.

## 1. Logical and physical tables

The logical data contract maps one-to-one to the fixed Phase 6 PostgreSQL
tables. There are no hidden storage-only tables in V1:

| Table                 | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| `road_segments`       | Static description of each road segment (one row per segment). |
| `road_observations`   | Monthly observation per segment (one row per segment per month). |
| `maintenance_events`  | Key-only maintenance-event timeline, including planned future events used to derive targets. |
| `observation_targets` | Event-derived targets per observation; physically and logically separate from model features. |
| `maintenance_history` | Fully realized cost and material facts for a subset of executed maintenance events. |
| `predictions`         | Model outputs: maintenance probability, risk score and band per (segment, date). |
| `material_forecasts`  | Network-month material quantity forecasts (derived, never features). |

`segment_id` is a stable string business identifier such as
`QL01-KM134-135`: road code `QL01` (national road) plus kilometre markers
134-135. `province` is a **separate** attribute; a road code such as `QL01`
is not a province. The V1 logical and physical contract uses the string
`segment_id` as its sole segment key: surrogate integer IDs are forbidden.

## 2. Column dictionary

Legend for the attribute columns:

- **Known at observation**: value is available from the start-of-day
  observation snapshot at time `t` (no future information).
- **Class/Reg feature**: may be used as a classification / regression model
  feature. `FORBIDDEN` means the column must never be a feature.
- **Target/forecast**: the column is a learning target or a forecast output
  rather than an input.

### `road_segments` (static per segment; no time-varying values)

| Column | Type | Unit | Raw nullability | Cleaned nullability | Range / categories | Provenance | Known at observation | Class feature | Reg feature | Target / forecast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `segment_id` | str | - | not null | not null | business code e.g. `QL01-KM134-135`, unique, stable, non-empty | road inventory registry | yes (identity) | no (identifier; grouping key only) | no (identifier) | no |
| `province` | str | - | not null | not null | province code from business registry (e.g. `NA`); a road code such as `QL01` is not a province | road inventory registry | yes | yes (categorical) | yes (encoded) | no |
| `road_type` | str | - | not null | not null | category set: highway / national / provincial / urban / rural | road inventory registry | yes | yes (categorical) | yes (encoded) | no |
| `construction_date` | date | - | not null | not null | <= first observation date | construction records | yes | yes | yes | no |
| `road_length_km` | float | km | not null | not null | > 0 | road inventory registry | yes | yes | yes | no |

### `road_observations` (one row per segment per month; time-varying values)

| Column | Type | Unit | Raw nullability | Cleaned nullability | Range / categories | Provenance | Known at observation | Class feature | Reg feature | Target / forecast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `segment_id` | str | - | not null | not null | same as `road_segments` | FK to `road_segments` | yes (identity) | no (identifier) | no (identifier) | no |
| `date` | date | - | not null | not null | monthly cadence; unique per (segment, date) | observation calendar | yes (snapshot time) | no (time key) | no (time key) | no |
| `traffic_volume` | int | vehicles / day | not null | not null | >= 0 | traffic surveys aggregated to month | yes | yes | yes | no |
| `heavy_vehicle_ratio` | float | ratio | not null | not null | 0.0 - 1.0 | traffic surveys aggregated to month | yes | yes | yes | no |
| `road_age_days` | int | days | derived | derived (not null) | >= 0 | derived per observation: `date - construction_date` | yes | yes | yes | no |
| `rainfall_mm` | float | mm / month | nullable | not null (imputed) | >= 0 | weather records aggregated to month | yes | yes | yes | no |
| `temperature` | float | degrees C (monthly mean) | nullable | not null (imputed) | -50 .. 60 | weather records | yes | yes | yes | no |
| `humidity` | float | percent (monthly mean) | nullable | not null (imputed) | 0 .. 100 | weather records | yes | yes | yes | no |
| `days_since_last_maintenance` | int | days | not null | not null | >= 0 (cap for never-maintained segments) | from `maintenance_events` strictly before `date` | yes | yes | yes | no |
| `previous_repairs` | int | count | not null | not null | >= 0 | count of `maintenance_events` strictly before `date` | yes | yes | yes | no |
| `road_condition_score` | int | score | not null | not null | 1 .. 100 (higher = better) | inspection survey at/before `t` | yes | yes | yes | no |
| `marking_condition_score` | int | score | not null | not null | 1 .. 100 | inspection survey at/before `t` | yes | yes | yes | no |
| `guardrail_condition_score` | int | score | not null | not null | 1 .. 100 | inspection survey at/before `t` | yes | yes | yes | no |
| `sign_condition_score` | int | score | not null | not null | 1 .. 100 | inspection survey at/before `t` | yes | yes | yes | no |
| `accident_count_30d` | int | count | not null | not null | >= 0, trailing 30 days | accident records | yes | yes | yes | no |
| `accident_count_365d` | int | count | not null | not null | >= 0, trailing 365 days | accident records | yes | yes | yes | no |

`road_age_days` is always derived per observation from
`road_segments.construction_date`; it is never stored or generated as an
independent value.

### `observation_targets` (one event-derived label row per observation)

| Column | Type | Unit | Raw nullability | Cleaned nullability | Range / categories | Provenance | Known at observation | Class feature | Reg feature | Target / forecast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `segment_id` | str | - | not null | not null | same as `road_segments` | FK to `road_segments` | yes (identity) | no | no | no |
| `date` | date | - | not null | not null | same as `road_observations` | FK to matching observation | yes (snapshot time) | no | no | no |
| `days_until_maintenance` | int | days | derived | derived (not null) | >= 0 | first `maintenance_events` key on or after the observation date | **no** (future) | no | no | **yes - regression target** |
| `maintenance_within_30_days` | int | - | derived | derived (not null) | 0 or 1 | derived from `days_until_maintenance` | **no** (future) | no | no | **yes - classification target** |

### `maintenance_history` (zero or one realized accounting row per event)

| Column | Type | Unit | Raw nullability | Cleaned nullability | Range / categories | Provenance | Known at observation | Class feature | Reg feature | Target / forecast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `segment_id` | str | - | not null | not null | same as `road_segments` | FK to `road_segments` | no | no | no | no |
| `maintenance_date` | date | - | not null | not null | any date | maintenance schedule/execution records | no | no | no | no (event key) |
| `maintenance_cost` | int | VND | not null | not null | > 0 | realized cost recorded at/after execution (accounting) | **no** | **FORBIDDEN** | **FORBIDDEN** | no (optimization input) |
| `thermoplastic_paint_kg` | float | kg | not null | not null | >= 0 | material consumption records at execution | no | **FORBIDDEN** | **FORBIDDEN** | no (forecast input) |
| `reflective_sheet_m2` | float | m2 | not null | not null | >= 0 | material consumption records at execution | no | **FORBIDDEN** | **FORBIDDEN** | no (forecast input) |
| `guardrail_meter` | float | m | not null | not null | >= 0 | material consumption records at execution | no | **FORBIDDEN** | **FORBIDDEN** | no (forecast input) |
| `traffic_sign_quantity` | int | count | not null | not null | >= 0 | material consumption records at execution | no | **FORBIDDEN** | **FORBIDDEN** | no (forecast input) |

**Material quantities** are actual historical consumption recorded from
maintenance events. Individual quantities may be **zero**: not every
maintenance action consumes every material, so no event is required to have
all four quantities positive. Material quantities are **forbidden as
classification and regression features**.

**`maintenance_cost`** is an **integer amount in VND**, a **realized** cost
recorded at or after the maintenance event (accounting provenance). It is
not guaranteed to be known before a maintenance decision, so it is
**excluded from the classification and regression feature allowlists**. It
maps to the `cost_vnd` input of the later maintenance optimization.

**Availability rule**: maintenance occurrence comes exclusively from
`maintenance_events`. At observation snapshot `t`, only event keys with
`maintenance_date < t` are available to features; events on or after `t`
(including future events) are forbidden. An event on `t` itself is the *next*
maintenance for that snapshot (see section 3). `maintenance_history` is
realized accounting/material provenance only and is never a model-feature
source.

### `predictions` (model outputs)

| Column | Type | Unit | Known at observation | Class feature | Reg feature | Target / forecast |
| --- | --- | --- | --- | --- | --- | --- |
| `segment_id` | str | - | yes | no | no | no |
| `date` | date | - | yes | no | no | no |
| `maintenance_probability` | float | probability | no (output) | no | no | yes (output) |
| `risk_score` | int | score | no (output) | no | no | yes (output) |
| `risk_band` | str | - | no (output) | no | no | yes (output) |

### `material_forecasts` (forecast outputs, network-month level)

| Column | Type | Unit | Known at observation | Class feature | Reg feature | Target / forecast |
| --- | --- | --- | --- | --- | --- | --- |
| `period` | date | - | yes | no | no | no |
| `material` | str | - | yes | no | no | no |
| `forecast_quantity` | float | kg / m2 / m / count | no (output) | no | no | yes (output) |

For V1, material forecasting is defined at **network-month** level
(`period`, `material`, `forecast_quantity`); `segment_id` is not required in
`material_forecasts`. Segment-level forecasting is not supported in V1
unless a separate forecast scope is explicitly documented. Forecast
quantities are outputs derived from maintenance-event history; they are
never used as features.

## 3. Target semantics

An observation is a **start-of-day snapshot**. `next_maintenance_date` is the
first maintenance event **on or after** the observation date:

```
next_maintenance_date >= observation_date
```

Then:

```
days_until_maintenance = next_maintenance_date - observation_date
```

```
maintenance_within_30_days =
    1 when 0 <= days_until_maintenance <= 30
    0 otherwise
```

Boundary behaviour (window 30): **0 days -> positive**, **30 days ->
positive**, **31 days -> negative**. Because the next event is selected on or
after the observation date, `days_until_maintenance` is never negative; a
past event is rejected (`ValueError` from `days_until_maintenance()`).

The helpers in `src/roadguard/targets.py` implement exactly this semantics
and are covered by unit tests for the boundary values.

Maintenance events are generated **first**; targets are derived from them
and never generated independently. Events are generated **beyond the final
observation date** so that the last observations are not mislabeled by right
censoring: an observation whose next event falls beyond the final
observation date still has a correctly computed `days_until_maintenance`.

## 4. Time split policy

Splits are chronological by unique observation dates with locked counts
34 / 7 / 7 (largest-remainder allocation from 70 / 15 / 15 over 48 dates,
section 0):

| Split      | Date count | Composition                          |
| ---------- | ---------- | ------------------------------------ |
| train      | 34         | first 34 unique dates                |
| validation | 7          | next 7 unique dates                  |
| test       | 7          | final 7 unique dates                 |

- Rows are never split randomly; the split boundary is defined by dates.
- Imputers, encoders, scalers, and outlier statistics are fit on training
  data only.
- Validation (or time-aware cross-validation) is used for tuning and model
  selection. The test set is evaluated exactly once on the frozen selected
  model.

## 5. Model selection contracts

Candidate selection never reads the test partition.

**Classification (maintenance_within_30_days)**
- Candidate ranking: **primary = validation PR-AUC** (precision-recall area
  under the curve).
- Decision threshold: maximize **validation F1**; tie-break on **higher
  validation recall**.
- Reported metrics (frozen model on test, once): accuracy, precision,
  recall, F1, ROC-AUC, and the confusion matrix.

**Regression (days_until_maintenance)**
- Candidate ranking: **primary = validation MAE**; tie-break on **lower
  validation RMSE**.
- Reported metrics (frozen model on test, once): MAE, RMSE, and R-squared.

**Forecasting (material quantities)**
- Forecasting uses a **separate rolling-origin timeline** and must **not**
  reuse the supervised 34 / 7 / 7 evaluation as its forecasting protocol.

## 6. Feature availability rules

- At observation time `t`, every model feature must use only information
  available at or before `t`. Features using future data are forbidden.
- Maintenance occurrence is available only from `maintenance_events` strictly
  before the observation date; future event keys are forbidden. Realized
  `maintenance_history`, costs, and materials are never ML features.
- Rolling features are computed **per segment** and **shifted before
  rolling**: `groupby(segment_id).shift(1).rolling(...)`. This prevents the
  current row from leaking into its own window.
- Material consumption is generated from maintenance events and is
  **forbidden** as a feature (section 2).
- `maintenance_cost` is a realized cost (section 2) and is **excluded** from
  ML features.
- `road_age_days` is derived from `construction_date`, never independently
  generated.

## 7. Artifact and inference contracts

- Generated data and model artifacts live under `data/` and `artifacts_dir`
  (runtime configuration) and are gitignored; only documentation (e.g.
  `data/README.md`) and intentional small fixtures are committable there.
- The inference pipeline loads models only from the managed artifact
  registry (MLflow in later phases). A model path supplied by an API user is
  never loaded.

### Risk score contract

```
risk_score = Decimal(str(maintenance_probability)) * 100,
             quantized to 1 with decimal ROUND_HALF_UP
```

- Deterministic **decimal ROUND_HALF_UP**: the probability is converted with
  `Decimal(str(p))` (never from the raw binary float), so binary
  floating-point representation cannot change `.5` behaviour. Python's
  banker's `round()` is not used, and no binary `floor(p * 100 + 0.5)`
  arithmetic is used.
- The probability is validated to be finite and within [0, 1]; values
  outside are rejected, not clamped, and booleans are rejected.
- The result is clamped to 0-100 only for floating-point numerical safety.
- Implemented once in `roadguard.risk.risk_score_from_probability`; every
  consumer must use it.
- Band classification uses the locked layout: LOW 0-30, MEDIUM 31-60, HIGH
  61-80, CRITICAL 81-100 (`V1_RISK_BANDS`).
- Model probabilities are only described as *calibrated* if calibration was
  actually implemented and evaluated. No calibration is implemented yet.

### Online inference contract

Online inference input is minimal:

```
segment_id, as_of_date
```

The runtime must, for a given `(segment_id, as_of_date)`:

- retrieve static segment attributes from PostgreSQL (`road_segments`);
- retrieve historical observations **up to and including** `as_of_date` and
  **exclude every observation after** `as_of_date`;
- build engineered features internally;
- **reject caller-supplied engineered feature vectors**;
- **reject caller-supplied model paths**;
- fail explicitly when the segment, its history, or required artifacts are
  unavailable;
- **never fill missing history with hard-coded defaults** (missing data is
  an explicit failure).

## 8. Reproducibility seed policy

- A single master seed is set through runtime configuration (`seed`, default
  42, validated positive, overridable via `ROADGUARD_SEED`).
- Every random number consumer (data generation, model training, splitting
  where applicable) must derive its stream deterministically from the master
  seed so that a given configuration reproduces identical data and results.
- The seed used for any generated dataset or trained artifact is recorded
  alongside the artifact metadata.
- Chronological splits depend on data, not on randomness; the seed governs
  the synthetic event and noise generation, not the split policy.

## 9. Generation methodology (Phase 2)

Implemented in `roadguard.segments` and `roadguard.events`.

**Observation calendar.** Monthly observations start at
`V1_OBSERVATION_START` (2022-01-01); the V1 window is the 48 first-of-month
dates from that start. `observation_dates(months, start)` reproduces the
calendar.

**Segment master.** `generate_segments(spec, seed)` produces the static
`road_segments` columns plus latent simulation state:
`traffic_base` (vehicles/day), `heavy_vehicle_ratio_base`, `weather_exposure`
(mean-rainfall multiplier), `deterioration_rate` (condition decay per month),
`accident_propensity`, `initial_condition` (1-100). Segment identifiers come
from a fixed registry (`ROAD_CODES`, `PROVINCES`) and `road_length_km`
equals the kilometre-marker span `end - start` of the business identifier;
both are deterministic and stable across seeds. Construction dates precede
the observation start by 5-30 years.

**Accident history.** `generate_accident_timeline(segments, spec, seed)`
produces deterministic monthly accident counts per segment (Poisson with
rate `accident_propensity * (traffic_base / 10000)^0.5 * 0.05`) for every
simulated month, reusable by later phases for accident features. The
maintenance hazard at month `t` uses the trailing accident history of the
12 months **strictly before** `t`; future accidents never influence hazard.

**Maintenance events.** `generate_maintenance_events(segments, spec, seed)`
simulates at most **one event per segment-month** via a Bernoulli draw with
probability `monthly_hazard(master, month_date, months_since_last_event,
condition, trailing_accidents, base_rate)`, the pure product of documented
latent drivers:

- base rate (`base_rate`, validated: not boolean, finite, positive);
- asset age (hazard grows with age since `construction_date`);
- traffic exposure (`(traffic_base / 10000)^0.2`);
- heavy-vehicle exposure (`1 + (ratio - 0.25) * 2`);
- weather exposure (`weather_exposure` with a calendar-month seasonal
  cycle);
- condition deterioration (a renewal proxy decaying by `deterioration_rate`
  per month via `decay_condition`, jumping toward `initial_condition` after
  each event; factor clamped 0.5-2.0x);
- trailing accident history (factor `1 + 0.3 * min(trailing_12m, 5)`);
- previous maintenance (renewal factor 0.3 / 0.6 / 1.0 for <6 / <12 /
  >=12 months since the last event, applied from the month **after** an
  event onward via the pure `month_transition` state helper).

The hazard is clamped to [0.02, 0.9]. Event day offsets are uniform within
the month; in the construction month, event days start at
`construction_date`. Simulation never runs before `construction_date`;
construction dates after the observation start are rejected. Months are
always simulated through the future-buffer horizon `sim_end`, so a longer
`future_buffer_months` extends the retained deterministic history and
shorter-buffer histories are preserved as prefixes; if no event falls after
the final observation date by `sim_end`, the same hazard and accident
processes continue until the first such event, subject to the documented
safety cap (`max_months_per_segment`, default 600) and an explicit
`GenerationError` failure. Targets are not generated in this phase and never
influence event generation.

**Determinism and row-order independence.** Each segment draws randomness
from its own stream `SeedSequence([seed, int.from_bytes(segment_id.encode("ascii"), "big")])` (no
Python `hash()`); child stream 0 is accidents, child stream 1 is maintenance
events. The segment master uses `SeedSequence(seed)` child stream 0. Same
seed => identical frames; different seeds change stochastic values (but not
segment IDs or lengths); shuffling the segment table never changes the
output frames.

## 10. Observation generation methodology (Phase 3)

Implemented in `roadguard.observations` (`generate_observations`). It builds
the clean observation-core `road_observations` table: one start-of-day
snapshot per (segment, month) with exactly the documented 16 columns, sorted
by `segment_id` and `date`. Inputs are never mutated; rows are generated
per segment and observation date in sorted order.

**Clean-core boundary.** Phase 3 produces a complete valid causal source:
no missing values, no invalid values, no injected outliers, no duplicates,
no cleaning, no imputation. Missingness, invalid values, outliers and
cleaning belong to later phases. No CSV/Parquet persistence and no target,
cost, material, anomaly, or engineered-feature columns are produced.

**As-of monthly traffic/weather.** Observation rows are labelled at
month-start as-of dates `t`. Traffic and weather represent the most recently
completed monthly aggregate available at the start of day `t` and are never
realized values of the future interval `[t, next_month)`. Per the
authoritative Phase 3 addendum, seasonality uses the as-of observation month
`m` and the trend uses the zero-based observation index `k`. Formulas
(named module constants, not configuration):

- traffic: `base * (1 + 0.003k) * (1 + 0.08 sin(2π(m-1)/12)) * exp(N(-σ²/2, σ))`
  with σ = 0.06, `np.rint` to a non-negative integer;
- heavy vehicles: `clip(base + 0.02 sin(2π(m-2)/12) + N(0, 0.012), 0, 1)`
  rounded to 4 decimals;
- rainfall: `max(0, 140 * exposure * (1 + 0.65 sin(2π(m-5)/12)) * exp(N(-σ²/2, σ)))`
  with σ = 0.22, 1 decimal;
- temperature: `clip(27 + 4 sin(2π(m-1)/12) - 0.8(exposure - 1) + N(0, 0.8), -50, 60)`,
  1 decimal;
- humidity: `clip(60 + 0.075 rainfall + 7(exposure - 1) + N(0, 2), 0, 100)`,
  1 decimal (humidity is positively dependent on rainfall by construction).

**Road age and maintenance history.** `road_age_days = (t -
construction_date).days` exactly. At snapshot `t`, `past_events` uses only
events with `maintenance_date < t`; `previous_repairs = len(past_events)`
and `days_since_last_maintenance = (t - latest past event).days`. With no
known prior event the value is `min(road_age_days, NEVER_MAINTAINED_DAYS_CAP =
3650)`; this means *no known event in simulated history*, not proof the
asset was never maintained. An event on `t` is excluded at `t` and becomes
historical only later.

**Condition replay and scores.** The latent condition replays the approved
Phase 2 `month_transition`/`decay_condition` chain from the pre-period
exactly as Phase 2 does, capturing the pre-transition state for each
observation month; an event on `t` cannot improve the score at `t` and its
effect first appears at the following snapshot. The four scores share the
latent condition (correlated) but use distinct documented modifiers and
noise, so they are never identical copies:

- road: `clip(rint(condition + N(0, 1.25)), 1, 100)`;
- marking: `clip(rint(condition - 4 - 0.00015 traffic - 0.008 rain + N(0, 1.5)), 1, 100)`;
- guardrail: `clip(rint(condition - 2 - 8 hgv - 2 log1p(accidents_365d) + N(0, 1.5)), 1, 100)`;
- sign: `clip(rint(condition - 3 - 0.04 max(humidity - 60, 0) - 0.004 rain + N(0, 1.5)), 1, 100)`.

**Exact accident windows.** Monthly counts from the Phase 2 timeline are
deterministically expanded into dated occurrences (a supporting internal
state, not a new V1 table): per (segment, year, month) bucket the RNG is
`SeedSequence([seed, key, OBSERVATION_RNG_NAMESPACE, ACCIDENT_DAY_STREAM,
year, month])`, offsets are drawn with replacement in `[0, days_in_month)`
(or from `construction_date.day - 1` in the construction month), and the
bucket's exact count is preserved. Windows are literal half-open intervals:

```
accident_count_30d  = count(t - 30d  <= accident_date < t)
accident_count_365d = count(t - 365d <= accident_date < t)
```

Not one/twelve calendar months. Complete monthly coverage over the required
history is validated; missing buckets raise an explicit error (never
silently zero). Accident counts are validated exactly (booleans, strings,
fractions and non-integral values rejected; integer-space preservation, no
float conversion) and capped at `MAX_ACCIDENTS_PER_SEGMENT_MONTH = 10_000`
before any expansion — the Decimal branch compares the exact Decimal against
0 and the cap before any `int` conversion, so an over-cap value is never
converted; every count validation error names the segment and month.
Expansion only covers months from `floor_month(first_obs - 365 days)`
through the final observation month: buckets strictly older than the
365-day window floor cannot affect any observation and are not expanded
(the boundary day `first_obs - 365 days` itself remains eligible), while the
condition replay continues to use the full required pre-period timeline.

**RNG namespaces.** Per segment: `SeedSequence([seed, key,
OBSERVATION_RNG_NAMESPACE, STREAM])` with streams TRAFFIC_STREAM,
WEATHER_STREAM, CONDITION_STREAM, plus the per-bucket ACCIDENT_DAY_STREAM
above, where `key = int.from_bytes(segment_id.encode("ascii"), "big")`. Draws per
observation date are consumed in the documented order (traffic noise,
heavy-vehicle noise; rainfall, temperature, humidity noise; road, marking,
guardrail, sign noise). No global NumPy state, no Python `hash()`, no Phase
2 RNG objects, no `spawn()` from a Phase 2 stream; changing one segment
never shifts another segment's stream.
## 11. Supervised target derivation (Phase 4)

Implemented in `roadguard.targets` (`derive_observation_targets`,
`TARGET_COLUMNS`).

**Purpose and separation.** Supervised labels are derived from the actual
Phase 2 maintenance-event keys; targets are never generated independently
and never influenced by random values (no seed, no RNG). The target frame
(`segment_id`, `date`, `days_until_maintenance`,
`maintenance_within_30_days`) is physically separate from the observation
feature table, and it never contains observation features, static segment
columns, cost, materials, or the internal `next_maintenance_date`
derivation state. Future labels are forbidden model features.

**Semantics.** For every observation snapshot at date `t`, the next
maintenance event is the earliest event for the same segment with
`maintenance_date >= t` (past events ignored):

- `days_until_maintenance = next_maintenance_date - t`;
- `maintenance_within_30_days = 1` when `0 <= days <= 30`, else 0.

Exact boundaries: same-day event -> days 0, class 1; 30 days -> class 1;
31 days -> class 0. The approved pure helpers
`days_until_maintenance` / `maintenance_within_30_days` are reused; the
derivation raises instead of reimplementing conflicting boundary rules.

**Right censoring is an error.** If any observation has no next event the
derivation raises a contextual `ValueError` naming the segment and
observation date; rows are never dropped, zero-filled, clipped, or treated
as negative class.

**Guarantees.** One target row per observation row, sorted by `segment_id`
and `date`; row-order invariant for both inputs; inputs never mutated;
explicit dtypes (object str / datetime64[ns] naive / int64 / int64). Date
values must be timezone-naive start-of-day (exactly midnight, including
zero nanoseconds) — non-midnight datetimes are rejected, never truncated —
and must be exactly representable as datetime64[ns]; duplicate required
column labels and unsupported object values are rejected contextually. Only
`segment_id` and `maintenance_date` are used from the event frame, so
later cost/material columns cannot change targets. Derivation uses a
per-segment sorted lookup (`searchsorted(side="left")`), never a Cartesian
join.

## 12. Raw-data corruption, validation and safe cleaning (Phase 5)

Implemented in `roadguard.data_quality`.

**Clean-core versus corrupted-raw boundary.** Phase 3 emits a clean causal
observation core and is never modified to emit dirty data. Phase 5 builds a
separate deterministic raw representation and structurally cleans it back to
a valid core. The public cleaned boundary contains exactly five segment
columns (`segment_id`, `province`, `road_type`, `construction_date`,
`road_length_km`); the latent simulation fields (`traffic_base`,
`heavy_vehicle_ratio_base`, `weather_exposure`, `deterioration_rate`,
`accident_propensity`, `initial_condition`) never cross it. Observations,
targets, segments and maintenance events stay physically separate; targets
never enter the observation frame.

**Corruption allowlists and manifest.** `inject_observation_corruption`
touches observation features only: missing values only in
`rainfall_mm`/`temperature`/`humidity` (never the first chronological
observation of a segment), domain-valid outliers only in
`traffic_volume`/`rainfall_mm` (multiplied by the documented
`outlier_multiplier`, default 8), and duplicates as exact copied rows (never
manufactured conflicting keys). Keys, dates, road age and all
target/future-event values are never corrupted. Every changed cell and
duplicated row is recorded in a deterministic `CorruptionManifest` (no full
row dumps). Rates default to `missing_rate = 0.02`,
`outlier_rate = 0.005`, `duplicate_rate = 0.002` and reject booleans,
strings, non-finite or negative values and rates above
`MAX_CORRUPTION_RATE = 0.25`; outlier arithmetic rejects overflow or
non-finite derived values before dtype conversion.

**RNG namespace and determinism.** Per-row streams
`SeedSequence([seed, segment_key, CORRUPTION_RNG_NAMESPACE, entropy])` (no
Python `hash()`, no global NumPy state). The entropy component is the day
offset from 1970-01-01 for dates on or after the epoch (existing historical
streams are preserved), and a disjoint non-negative pre-epoch namespace
(`2**62 + |offset|`) for earlier dates, so equal-distance pre- and
post-epoch dates can never share an entropy component. Same seed and data
produce identical corrupted output and manifest; different seeds change
affected keys; shuffled input rows produce identical canonical output; both
output and manifest are sorted deterministically; inputs are never mutated.

**Maintenance-event invariants.** Validation enforces the locked Phase 2
event contract with deterministic issue codes:

- `event_before_construction`: a maintenance date strictly before the
  segment's `construction_date` (events on the construction date itself are
  allowed);
- `event_month_conflict`: more than one event for the same segment and
  calendar month, even when the dates differ;
- `duplicate_key`: an exact duplicate `(segment_id, maintenance_date)` key.

All three are errors reported by both `validate_raw_dataset` and
`validate_cleaned_dataset`; `clean_raw_dataset` rejects such datasets.

**Validation severity rules.** `validate_raw_dataset` warns on permitted
weather missingness, fixed operational outlier thresholds
(`TRAFFIC_VOLUME_OUTLIER_MAX = 100_000`, `RAINFALL_OUTLIER_MAX = 1_000`)
and exact duplicate observation rows; it errors on missing keys, invalid
domains, non-finite values, bad foreign keys, conflicting keys, malformed
schemas/dtypes/dates/IDs, grid/cadence violations, cross-field violations
(`construction_date <= date`; `road_age_days == date - construction_date`;
`0 <= days_since_last_maintenance <= road_age_days`;
`0 <= heavy_vehicle_ratio <= 1`; weather ranges; integer condition scores in
[1, 100]; `0 <= accident_count_30d <= accident_count_365d`) and target
mismatches, including recomputing targets from the actual maintenance-event
keys (`target_event_inconsistency`). `validate_cleaned_dataset` escalates
remaining missing values and duplicate keys to errors. Reports carry exact,
deterministic issue counts and locations.

**Causal forward-fill policy.** Cleaning fills permitted weather
missingness using only the most recent strictly earlier non-missing value
from the same segment: group by segment, sort by date, forward-fill only,
never backward-fill, never another segment, never a global median/mean/mode;
a segment with no earlier valid value fails contextually. Exact duplicate
rows are removed; conflicting rows sharing an observation key are errors.
Domain-valid outliers are preserved and reported as warnings; clipping,
winsorizing or learned thresholds are not applied in Phase 5 — statistical
preprocessing must be fitted on training data only in a later phase.

**Frames and dtypes.** Cleaning returns public segments (5 columns,
`construction_date` normalized to datetime64[ns]), cleaned observations with
exactly `OBSERVATION_COLUMNS` sorted by `segment_id`/`date`, the unchanged
target frame, the unchanged maintenance-event frame, and the cleaned
validation report.

## 13. Transactional PostgreSQL persistence (Phase 6)

Phase 6 is limited to the physical PostgreSQL boundary between validated
Phase 5 data and later analytical phases. It does not engineer features,
split or scale data, train models, serve an API, or introduce a Docker
deployment.

**Fixed schema and keys.** A PostgreSQL schema named `roadguard` owns exactly
seven tables. All tables use their domain keys; no surrogate `id` is added:

- `road_segments`: primary key `segment_id`, with exactly the five public
  segment columns;
- `road_observations`: primary key (`segment_id`, `date`), with exactly
  `OBSERVATION_COLUMNS` and a foreign key to `road_segments`;
- `maintenance_events`: primary key (`segment_id`, `maintenance_date`), the
  key-only event timeline produced by earlier phases;
- `observation_targets`: primary key (`segment_id`, `date`), foreign key to
  the matching observation, with exactly `TARGET_COLUMNS`;
- `maintenance_history`: primary key (`segment_id`, `maintenance_date`) and
  foreign key to the matching event; cost and all four material fields are
  non-null realized facts;
- `predictions`: schema reserved for later prediction outputs, keyed by
  (`segment_id`, `date`);
- `material_forecasts`: schema reserved for later forecast outputs, keyed by
  (`period`, `material`).

The event and maintenance-history tables are deliberately separate. Earlier
phases know event keys but do not generate cost or material usage. Phase 6
therefore never invents those values; a maintenance-history row is accepted
only when every realized field is supplied.

**Configuration and initialization.** `database_url` is optional runtime
configuration, exposed as `ROADGUARD_DATABASE_URL` and held as a masked
`SecretStr`. Persistence requires the exact synchronous
`postgresql+psycopg` driver. Schema initialization is repeatable. Public
configuration and database errors do not echo credentials or SQL parameter
values. In particular, `load_config` converts YAML/environment validation
failures into a context-free `ConfigError` rather than exposing Pydantic's
raw input details.

**Transactional ETL.** `load_cleaning_result` accepts only a validated
`CleaningResult` plus its explicit `DatasetSpec`. It deep-copies and reruns the
complete cleaned Phase 5 validation on those exact frames immediately before
normalization, so a stale or forged report cannot authorize invalid labels.
It does not mutate caller-owned frames. Segments, observations, events,
targets, and optional realized history are written in one transaction. Every
table uses PostgreSQL conflict-safe insert-or-verify reconciliation: a missing
key is inserted, an identical persisted or concurrent row is counted as
existing, and a different row under the same key raises a conflict. Every
realized-history key must belong to the current validated event batch. Any
constraint or connection failure rolls back the whole load. PostgreSQL also
enforces the one-event-per-segment-calendar-month invariant.

**Read semantics.** Exports are ordered by natural key and preserve canonical
column order and dtypes; observations and targets are returned in separate
frames. A point-in-time segment query accepts a validated business identifier
and a date, returning observations at or before that date and maintenance
events strictly before it. Monthly material aggregation reads only complete,
realized `maintenance_history` rows. All caller values are passed through
bound SQLAlchemy expressions rather than interpolated SQL.
Multi-query exports and histories execute in one read-only `REPEATABLE READ`
transaction so concurrent commits cannot mix physical snapshots. A known
segment with no observation at or before the requested date fails explicitly.

**Schema drift.** Initialization creates the schema only when it is empty.
When tables already exist, their exact table set, column order/types/nullability,
primary and foreign keys, named check constraints, and indexes are verified
before use. Partial, extra, or incompatible objects fail safely; the
initializer does not repair or delete them implicitly.

## 14. Point-in-time feature registry and generation (Phase 7)

Phase 7 creates the deterministic offline feature frame for later modelling.
It accepts only a complete `RepositoryExport` from the Phase 6 read boundary
and its explicit `DatasetSpec`; it deep-copies and re-runs complete cleaned
validation before feature generation. It additionally verifies that
`previous_repairs` and `days_since_last_maintenance` equal the values derived
from strictly-prior `maintenance_events` (including the locked
never-maintained cap). A prior validation report or an equivalent-looking
collection of frames is not a bypass token.

**Scope boundary.** Phase 7 joins validated public segment attributes to
validated observation rows and produces a registry-defined frame only. It
does not write a database table, derive a lag or rolling feature, split rows,
impute, encode categoricals, scale, select/evaluate a model, train, serve an
API, or include a target, event key, cost, material fact, anomaly field, or
latent simulation value in its output.

**Feature frame.** The output is canonically sorted by (`segment_id`, `date`)
and has these two key columns, which are never model features:

```
segment_id, date
```

The frozen V1 Phase 7 registry contains exactly these model-feature columns:

```
province, road_type, construction_date, road_length_km,
traffic_volume, heavy_vehicle_ratio, road_age_days,
rainfall_mm, temperature, humidity,
days_since_last_maintenance, previous_repairs,
road_condition_score, marking_condition_score,
guardrail_condition_score, sign_condition_score,
accident_count_30d, accident_count_365d
```

`province` and `road_type` remain raw categorical values and
`construction_date` remains a raw datetime in this phase. Their train-only
encoding/transformation belongs to Phase 8. The registry's feature eligibility
does not authorize an unlisted engineered feature.

**Provenance and determinism.** Targets and maintenance-event keys are read
solely to revalidate the exported source boundary; they never influence any
output value. The generator never mutates caller-owned frames and normalizes
the output keys, categoricals, datetimes, integers, and floats to the exact
canonical Phase 6 dtypes. Equivalent shuffled input frames produce the same
canonical frame.

## 15. Chronological splitting and train-only preprocessing (Phase 8)

Phase 8 accepts only the exact target-free Phase 7 feature frame from
`docs/contracts.md` section 14 and its `DatasetSpec`. It splits by sorted
unique observation dates and fits deterministic preprocessing on the training
partition only. It does not join targets, engineer lag/rolling features,
impute, clip, train a model, tune, persist artifacts, write a database, or
serve an API.

**Input boundary.** The frame must have exactly the `FEATURE_FRAME_COLUMNS`
schema in exact order: `segment_id` and `date` are natural keys, never model
features. Missing, extra, reordered, or duplicate-labelled columns, duplicate
natural keys, malformed/timezone-aware/non-midnight dates, null values,
non-finite numeric values, and invalid dtypes are rejected contextually. The
frame must reproduce the complete `DatasetSpec` observation grid (exactly
`dataset_months_per_segment` unique dates, `dataset_segments` segments per
date, `dataset_observations` rows). For the V1 profile this means exactly 48
unique dates, 300 segments per date, and 14,400 rows. Caller-owned frames are
never mutated.

**Chronological split.** `split_chronologically` sorts the unique observation
dates and assigns the first 34 to training, the next 7 to validation, and the
final 7 to testing. Partitions are disjoint, contiguous, and together
reproduce the complete input; each is canonically sorted by (`segment_id`,
`date`). Shuffling valid input rows never changes any partition. For V1 the
exact row counts are 10,200 train, 2,100 validation, and 2,100 test.

**Train-only preprocessing.** `fit_preprocessor` accepts only the complete
provenance-checked `ChronologicalSplit`. It reconstructs and validates the
full 48-date source and its partition membership, then fits only the canonical
first 34 dates. A caller cannot present an arbitrary future-contaminated
34-date frame as training data through the public API. The function returns
an immutable `PreprocessorFit`:

- `province` and `road_type` are one-hot encoded using the sorted unique
  training categories only, producing stable columns named
  `province_<category>` and `road_type_<category>`; unknown
  validation/test categories encode as an all-zero row and never change the
  fitted schema.
- `construction_date` becomes the deterministic numeric day count since
  1970-01-01 (column `construction_date_days`) and is scaled with training
  statistics.
- The remaining numeric Phase 7 features are scaled with training mean and
  population standard deviation (zero-variance training columns transform to
  a constant zero).

`transform` applies only the fitted state to any frame: it never refits and
never consults validation/test statistics. Every transformed feature value
is a finite `float64`; partition keys are returned separately from model
features. The stable transformed feature names are exposed as
`PreprocessorFit.transformed_feature_columns`.
