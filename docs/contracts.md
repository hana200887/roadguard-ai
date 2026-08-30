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

## 16. Train-only exploratory analysis and data card (Phase 9)

Phase 9 produces reproducible descriptive evidence without weakening the
frozen-test policy. Its public module is `roadguard.eda` and its only workflow
is:

```python
build_eda_report(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    spec: DatasetSpec,
) -> EDAReport
render_data_card(report: EDAReport) -> str
```

`EDAError` is the contextual `ValueError` subclass for invalid Phase 9 data.
The module `roadguard.eda.__all__` contains exactly that error, the two
workflow functions, and these result types: `EDAReport`, `SplitInventory`,
`DataQualitySummary`, `NumericSummary`, `CategoricalLevel`,
`CategoricalSummary`, `DateSummary`, `ClassificationBalance`, and
`TargetCorrelation`. The package root additionally exposes the same Phase 9
symbols while preserving every locked Phase 1-8 export. Each result type is a
`@dataclass(frozen=True)` whose fields use only the exact built-in Python
scalar types and tuples specified below; NumPy/Pandas scalar substitutes are
not stored. No result exposes a caller-owned mutable frame or mapping.

### Input and leakage boundary

`build_eda_report` accepts exact instances of `RepositoryExport`,
`ChronologicalSplit`, and `DatasetSpec`. It deep-copies caller-owned frames,
fresh-runs the complete cleaned-data validation, rebuilds the exact Phase 7
feature frame, runs `split_chronologically`, and requires the supplied split
to equal that canonical result in schema, dtypes, keys, dates, values, and
partition membership. A previous validation report, transformed frame,
standalone training frame, or lookalike object is not accepted.

Targets remain a separate source frame and are joined one-to-one by
(`segment_id`, `date`) only after validation. Statistics and the training
fingerprint use exactly the 34 training dates. Validation and test contribute
only `SplitInventory` metadata: partition name, row count, unique-date count,
first date, and last date. Their feature values and target values never
influence any statistic, category inventory, correlation, fingerprint, or
rendered text. Reading later-partition targets solely as part of fresh source
integrity validation is not evaluation and must not expose them in the result.

Phase 9 does not fit or apply preprocessing, impute, clip, engineer/select a
feature, train/tune/select/evaluate a model, calculate model metrics, set a
decision threshold, assign a risk band, persist an artifact, query or write a
database, or write to the filesystem. In particular it reports no validation
or test distribution, target prevalence, correlation, or model evidence.

### Frozen report schema and ordering

`EDAReport` contains exactly these fields in this order:

1. `contract_version: str`, exactly `roadguard.phase9.v1`;
2. `training_fingerprint: str`, the lowercase 64-character SHA-256 defined
   below;
3. `feature_columns: tuple[str, ...]`, exactly `FEATURE_COLUMNS`;
4. `split_inventory: tuple[SplitInventory, ...]`, ordered train, validation,
   test;
5. `data_quality: DataQualitySummary`;
6. `numeric_features: tuple[NumericSummary, ...]`, in numeric Phase 7 registry
   order;
7. `categorical_features: tuple[CategoricalSummary, ...]`, in categorical
   Phase 7 registry order;
8. `datetime_features: tuple[DateSummary, ...]`, in datetime Phase 7 registry
   order;
9. `regression_target: NumericSummary`, for `days_until_maintenance`;
10. `classification_target: ClassificationBalance`, for
    `maintenance_within_30_days`;
11. `target_correlations: tuple[TargetCorrelation, ...]`, ordered first by
    numeric Phase 7 registry order and then by `TARGET_COLUMNS[2:]` order. It
    contains exactly one element for every Cartesian-product pair of those
    two collections, with no missing or duplicate pair, so its cardinality is
    `len(numeric_features) * len(TARGET_COLUMNS[2:])`.

The nested dataclass fields and annotations are exactly:

```python
SplitInventory(
    name: Literal["train", "validation", "test"],
    row_count: int,
    date_count: int,
    first_date: date,
    last_date: date,
)
DataQualitySummary(
    row_count: int,
    segment_count: int,
    date_count: int,
    duplicate_key_count: int,
    missing_cell_count: int,
    non_finite_numeric_count: int,
)
NumericSummary(
    column: str,
    count: int,
    missing_count: int,
    mean: float,
    population_std: float,
    minimum: float,
    q1: float,
    median: float,
    q3: float,
    maximum: float,
    iqr_outlier_count: int,
    iqr_outlier_rate: float,
    zero_variance: bool,
)
CategoricalLevel(value: str, count: int, proportion: float)
CategoricalSummary(
    column: str,
    count: int,
    missing_count: int,
    cardinality: int,
    levels: tuple[CategoricalLevel, ...],
)
DateSummary(
    column: str,
    count: int,
    missing_count: int,
    unique_count: int,
    minimum: date,
    maximum: date,
)
ClassificationBalance(
    column: str,
    negative_count: int,
    positive_count: int,
    positive_rate: float,
)
TargetCorrelation(feature: str, target: str, pearson_r: float | None)
```

`DataQualitySummary` is calculated on the canonically sorted training
feature/target join; missing cells cover every joined column and non-finite
counts cover every numeric feature and target cell. Its last three values must
be zero after fresh validation; they remain explicit evidence rather than
being omitted.

`NumericSummary` contains `column`, `count`, `missing_count`, `mean`,
`population_std`, `minimum`, `q1`, `median`, `q3`, `maximum`,
`iqr_outlier_count`, `iqr_outlier_rate`, and `zero_variance`.
`CategoricalLevel` contains `value`, `count`, and `proportion`;
`CategoricalSummary` contains `column`, `count`, `missing_count`,
`cardinality`, and `levels`. Levels are ordered by descending count and then
ascending Unicode code-point value. `DateSummary` contains `column`, `count`,
`missing_count`, `unique_count`, `minimum`, and `maximum` as `date` values.
`ClassificationBalance` contains `column`, `negative_count`, `positive_count`,
and `positive_rate`. `TargetCorrelation` contains `feature`, `target`, and
`pearson_r: float | None`.

All counts are exact integers. Every descriptive calculation first converts
each canonically ordered numeric value with `Decimal.from_float` for floats or
`Decimal(value)` for integers, then operates sequentially in that order inside
a fresh `decimal.localcontext` with precision 80, `ROUND_HALF_EVEN`,
`Emin=-999999999`, `Emax=999999999`, and traps enabled for
`InvalidOperation`, `DivisionByZero`, and `Overflow`. This context, not
pandas/NumPy reductions or binary-float accumulation, is authoritative.

Means and proportions use all training rows. The mean is the Decimal sum
divided by the integer count. Population variance is the Decimal sum of
squared deviations from that mean divided by count; population standard
deviation is its Decimal square root (`ddof=0`). Quartiles use canonically
sorted Decimal values and positions `(count - 1) * p` for exact Decimal
`p` values `0.25`, `0.50`, and `0.75`; a non-integral position linearly
interpolates its floor/ceiling values. An IQR outlier is strictly below
`q1 - 1.5 * (q3 - q1)` or strictly above `q3 + 1.5 * (q3 - q1)` in Decimal
arithmetic; its rate is the count divided by `count`. Before any reduction,
an exact constant fast path checks whether every source Decimal equals the
first. For a constant column, the mean/minimum/quartiles/maximum are that
exact Decimal, variance and population standard deviation are exact Decimal
zero, outlier count/rate are zero, and `zero_variance` is true. For a
non-constant column, a computed zero variance is an arithmetic failure and
raises `EDAError`; otherwise `zero_variance` is false.

Pearson correlation uses Decimal means and the direct paired-row formula
`sum((x-x_mean)*(y-y_mean)) / sqrt(sum((x-x_mean)^2)*sum((y-y_mean)^2))`;
it is `None` when either source sequence is exactly constant according to the
same pre-reduction check. A zero variance sum for a non-constant input raises
`EDAError`. Each derived Decimal stored in a report is converted once with
Python `float`; a non-finite or unrepresentable result, or any trapped Decimal
operation, raises `EDAError` instead of emitting partial evidence. Source
probes include huge finite constant and non-constant values near the
`float64` limits; a huge constant must produce exact `population_std == 0.0`
and `zero_variance is True`. No presentation rounding occurs in the report
object, and every stored float is finite.

### Canonical training fingerprint

The fingerprint is SHA-256 over UTF-8 canonical JSON produced with sorted
object keys, separators `(',', ':')`, ASCII escaping enabled, and non-finite
values forbidden. The payload has exactly this shape (placeholders show the
value type, not literal content):

```json
{
  "columns": ["FEATURE_FRAME_COLUMNS + TARGET_COLUMNS[2:]"],
  "contract": "roadguard.phase9.v1",
  "spec": {
    "dataset_months_per_segment": 0,
    "dataset_observations": 0,
    "dataset_segments": 0
  },
  "split": {
    "test": ["YYYY-MM-DD"],
    "train": ["YYYY-MM-DD"],
    "validation": ["YYYY-MM-DD"]
  },
  "train_rows": [["canonical scalar"]]
}
```

The displayed indentation is illustrative; canonical bytes use the compact
separators specified above. The real `columns` array is exactly
`FEATURE_FRAME_COLUMNS + TARGET_COLUMNS[2:]`; the real spec integers and all
ordered split dates replace the placeholders. `train_rows` is sorted by
(`segment_id`, `date`) and each row is an array in the exact declared column
order after the one-to-one target join.

Dates and datetimes canonicalize to `YYYY-MM-DD`, strings remain strings,
integers remain JSON integers, and finite floats canonicalize to lowercase
`float.hex()` strings with negative zero normalized to positive zero. The
fingerprint therefore binds the training evidence and split provenance but
does not hash validation/test feature or target values.

### Deterministic data card

`render_data_card` accepts only an exact, internally consistent `EDAReport`
and rejects invalid contract versions, columns, ordering, malformed digests,
non-finite values, invalid counts/rates, or contradictory totals. It returns
UTF-8-compatible Markdown with `\n` line endings and exactly one trailing
newline. The title and headings are exactly:

```text
# RoadGuard AI - Phase 9 Train-Only Data Card
## Scope and leakage guard
## Provenance
## Split inventory
## Training data quality
## Training feature summaries
### Numeric features
### Categorical features
### Datetime features
## Training target summaries
### Regression target
### Classification target
## Train-only target correlations
## Limitations
```

The scope section contains exactly these bullets in order:

```text
- Statistics and correlations use only the canonical 34-date training partition.
- Validation and test are represented only by row counts, date counts, and date boundaries.
- No preprocessing was fit or applied, and no model was trained, selected, or evaluated.
```

The provenance section contains these bullets in order: ``- Contract:
`roadguard.phase9.v1` ``; ``- Training fingerprint: `<digest>` ``; and
``- Feature columns: `column_1`, `column_2`, ...`` using the exact report order
and one code span per column. The split table columns are
`Partition | Rows | Dates | First date | Last date`. The data-quality table
columns are `Rows | Segments | Dates | Duplicate keys | Missing cells |
Non-finite numeric cells`. The numeric-feature and regression-target tables
use `Column | Count | Missing | Mean | Population std | Min | Q1 | Median | Q3
| Max | IQR outliers | IQR outlier rate | Zero variance`. The categorical
table uses `Column | Level | Count | Proportion`; the datetime table uses
`Column | Count | Missing | Unique | Min | Max`; the classification table uses
`Column | Negative | Positive | Positive rate`; and the correlation table uses
`Feature | Target | Pearson r`. Each table uses the conventional Markdown
delimiter row made only from `---` cells. Every table row begins and ends with
`|` and uses one ASCII space between a pipe and its cell content.

The headings, bullets, tables, and dynamic rows specified here are exhaustive;
the renderer adds no other prose, headings, notes, blank blocks, or metadata.
The categorical table contains exactly one row for every `CategoricalLevel`
in report order. Rows follow report ordering. Integers use unsigned base-10 notation, dates use
`YYYY-MM-DD`, booleans use lowercase `true` or `false`, and displayed floats
are produced inside a fresh `decimal.localcontext` with precision 1100,
`ROUND_HALF_EVEN`, `Emin=-999999999`, `Emax=999999999`, and traps enabled for
`InvalidOperation`, `DivisionByZero`, and `Overflow`. Inside that context the
renderer applies
`Decimal.from_float(value).quantize(Decimal("0.000001"),
rounding=ROUND_HALF_EVEN)`, then fixed-point formatting with exactly six
decimal places; displayed negative zero is normalized to `0.000000`.
Rendering failure raises `EDAError`, never partial Markdown. Tests replace the
caller's global Decimal context and render summaries containing finite values
near both `float64` limits. Undefined correlation is written `not-defined`.
There is exactly one blank line between headings, prose/list blocks, and
tables, and no blank line inside a table.

The limitations section contains exactly these bullets in order:

```text
- This card is descriptive train-only evidence; it is not causal analysis or model-performance evidence.
- Validation and test feature/target distributions were not summarized.
- The SHA-256 fingerprint is an equality/integrity identifier, not anonymization, authentication, or a digital signature.
```

The renderer includes no wall-clock timestamp, host/path, random value,
environment detail, database credential, raw row, or later-partition
statistic. It requires the exact fixed field/column/target names, exact split
names `train`, `validation`, and `test`, and categorical levels drawn only from
the matching locked `PROVINCES` or `ROAD_TYPES` registry; forged dynamic text
is rejected rather than interpolated. Equivalent shuffled source frames
followed by a canonical Phase 7/8 rebuild produce equal `EDAReport` values,
the same fingerprint, and byte-identical Markdown across calls.

## 17. Baseline supervised evaluation (Phase 10)

Phase 10 establishes one deterministic learned baseline for each supervised
target without performing candidate tuning or publishing model artifacts. Its
public module is `roadguard.baselines` and its only workflow is:

```python
evaluate_baselines(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
) -> BaselineEvaluation
```

`BaselineEvaluationError` is the contextual `ValueError` subclass for invalid
Phase 10 input, estimator, prediction, or metric state. The module `__all__`
contains exactly `evaluate_baselines`, that error, `BaselineEvaluation`,
`ClassificationBaselineMetrics`, `RegressionBaselineMetrics`, and these
constants: `BASELINE_CONTRACT_VERSION`, `BASELINE_CLASSIFIER_NAME`,
and `BASELINE_REGRESSOR_NAME`. The package root exposes the same symbols while
preserving every locked Phase 1-9 export.

Wrong top-level argument types and lookalike objects raise `TypeError` before
any field is read. After all four arguments pass exact-type validation,
expected lower-phase `FeatureInputError` and `PreprocessingError` failures,
plus expected scikit-learn estimator/metric `ValueError` or arithmetic
failures, are translated to
`BaselineEvaluationError` with a fixed phase/context message and suppressed
exception chaining. That public message contains no raw value, row, object
representation, filesystem path, environment value, database URL, or
credential. Phase 10's own domain checks raise `BaselineEvaluationError`
directly. Unexpected programming failures and system-exiting exceptions are
not relabeled as input errors.

### Locked estimators and dependency

Phase 10 adds the runtime dependency `scikit-learn>=1.5,<2`; `uv.lock` must pin
the resolved version and transitive dependencies. Scikit-learn is used for
the estimators and metrics below rather than maintaining custom numerical ML
or metric implementations.

The classifier is exactly `sklearn.dummy.DummyClassifier(strategy="prior")`.
It learns only the training-label class prior, ignores feature values, uses no
RNG, and its public name is exactly `dummy_prior`.

The regressor is exactly `sklearn.dummy.DummyRegressor(strategy="median")`.
It learns only the training-target median, ignores feature values, uses no
RNG, and its public name is exactly `dummy_median`.

These two estimators are neutral benchmarks, not Phase 11/12 candidate models.
Phase 10 records their validation metrics but does not use those metrics to
choose between estimators. Feature-dependent learning and validation-ranked
candidate selection begin only in the later advanced-model phases.

`BASELINE_CONTRACT_VERSION` is exactly `roadguard.phase10.v1`. Estimator
objects, coefficients, predictions, and fitted preprocessing state remain
private to the call and are never returned or persisted. Phase 10 has no
random consumer and therefore introduces no seed argument or RNG namespace.

### Input, provenance, and temporal boundary

`evaluate_baselines` accepts exact instances of `RepositoryExport`,
`ChronologicalSplit`, `PreprocessorFit`, and `DatasetSpec`. It deep-copies
caller-owned frames, fresh-runs the complete cleaned-data validation through
`build_feature_frame`, rebuilds the canonical `split_chronologically` result,
and requires the supplied split to equal it in schema, dtypes, keys, dates,
values, and partition membership. It calls `fit_preprocessor` on that complete
canonical split, requires the supplied fit to equal the freshly reproduced fit
field-for-field, and transforms train and validation internally. A caller
cannot supply a transformed feature matrix, standalone target array,
estimator, threshold, or lookalike input object.

Targets remain separate and join one-to-one to each partition solely by the
canonical (`segment_id`, `date`) keys after validation. Joined target order
must exactly equal transformed key order. The classifier uses only
`maintenance_within_30_days`; the regressor uses only
`days_until_maintenance`. Keys, either target, future event keys,
`maintenance_history`, costs, materials, latent generation fields, and every
column outside `PreprocessorFit.transformed_feature_columns` are forbidden
model features.

Both estimators fit exactly once on the transformed 34-date training
partition. The fitted estimators produce validation predictions exactly once;
validation selects only the classifier threshold and records the locked
metrics below. Only after both estimators and the threshold are frozen may the
test partition be transformed and passed exactly once to one private
test-evaluation boundary. Before that point, test feature/target values may be
read only for fresh integrity and canonical-split validation; they cannot be
transformed, predicted, summarized, ranked, or used by any fit/selection
operation. There is no public API that accepts a frozen model and evaluates a
test partition again. “Exactly once” means one private test stage per pure
workflow invocation; it does not introduce a global counter, consumable token,
filesystem marker, or other cross-call state.

Changing only test feature/target values in a freshly valid matching export
and split may change only test metrics. It cannot change fitted state,
validation predictions/metrics, the threshold, estimator names,
feature schema, or row counts. Changing only validation feature/target values
may change validation metrics and the threshold but cannot change fitted
preprocessing or estimator state. Equivalent shuffled upstream frames followed
by canonical rebuilding produce equal results, subject only to the exact
locked numerical library versions in `uv.lock`.

### Threshold and metric semantics

Classifier probabilities are the positive-class column from
`predict_proba`. Each estimator scalar is converted once to an exact built-in
`float` and must be finite and in `[0.0, 1.0]`; invalid output is rejected,
never clipped. Candidate thresholds are `0.0`, `1.0`, and each distinct
validation probability. A hard prediction is one
when `probability >= threshold` and zero otherwise. Candidate thresholds are
ranked by higher validation F1, then higher validation recall, then higher
threshold; the first candidate under that total ordering is frozen. Training,
validation, and test classification targets must each contain both integer
classes 0 and 1. The test-label class check occurs only inside the frozen test
stage.

Classification metrics use scikit-learn with target class 1 as positive:

- validation PR-AUC is `average_precision_score` on probabilities;
- hard-label classification metrics use the frozen threshold over
  `predict_proba`; `DummyClassifier.predict` is never used;
- validation F1 and recall are `f1_score` and `recall_score` on hard labels,
  both with `zero_division=0`;
- test accuracy, precision, recall, and F1 use `accuracy_score`,
  `precision_score`, `recall_score`, and `f1_score`, with
  `zero_division=0` where accepted;
- test ROC-AUC is `roc_auc_score` on probabilities;
- the test confusion matrix is `confusion_matrix(labels=(0, 1))`, stored as
  `((true_negative, false_positive), (false_negative, true_positive))`.

Regression prediction scalars are converted once to finite built-in floats
and evaluated raw: they are never clipped or rounded even when negative.
Validation MAE and RMSE and test MAE and RMSE use `mean_absolute_error` and
`root_mean_squared_error`. Test R-squared uses
`r2_score(force_finite=False)`. The test regression target must contain at
least two distinct values; a constant test target is rejected instead of
allowing a forced or non-finite R-squared value.

All returned float metrics and the selected threshold must be exact finite
built-in floats. Counts must be exact built-in integers. Any non-finite
estimator state or prediction, invalid metric output, invalid shape, or
unexpected class ordering raises `BaselineEvaluationError`; no partial result
is returned.

### Frozen result schema

All result types are `@dataclass(frozen=True)` and expose no mutable frame,
array, estimator, mapping, or caller-owned object. Their fields and order are
exactly:

```python
ClassificationBaselineMetrics(
    validation_pr_auc: float,
    decision_threshold: float,
    validation_f1: float,
    validation_recall: float,
    test_accuracy: float,
    test_precision: float,
    test_recall: float,
    test_f1: float,
    test_roc_auc: float,
    test_confusion_matrix: tuple[tuple[int, int], tuple[int, int]],
)
RegressionBaselineMetrics(
    validation_mae: float,
    validation_rmse: float,
    test_mae: float,
    test_rmse: float,
    test_r2: float,
)
BaselineEvaluation(
    contract_version: str,
    classifier_name: str,
    regressor_name: str,
    feature_columns: tuple[str, ...],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    classification: ClassificationBaselineMetrics,
    regression: RegressionBaselineMetrics,
)
```

The contract version and estimator names equal the constants above;
`feature_columns` equals the fitted
`PreprocessorFit.transformed_feature_columns`; and row counts equal the exact
canonical partitions (10,200/2,100/2,100 for V1). Results contain no split
dates or data summaries already owned by earlier phases, no train metrics,
predictions, coefficients, feature importance, calibration claim, risk score,
risk band, filesystem path, timestamp, environment detail, or database
credential.

### Failure, testing, and scope boundary

Invalid schemas/dtypes/keys/dates/values, duplicate or missing keys, target
misalignment, forged partition membership or preprocessing state,
null/non-finite features, invalid targets, degenerate classification
partitions, constant test regression targets, invalid
estimator/probability/prediction state, or
non-finite metrics fail contextually without mutating caller-owned objects.

RED-first tests must cover the exact public surface and frozen schemas; known
classification/regression vectors against direct scikit-learn references;
threshold primary and both tie-break rules; train-only fit; validation-only
threshold selection; the single frozen test boundary; shuffled-input
determinism; caller immutability; target/key alignment; every forbidden-field
class; forged split/fit/lookalike inputs; non-finite and out-of-range estimator
output; single-class partitions; constant test regression targets; and a
complete V1-profile evaluation. The V1 test asserts schema,
finite/ranged metrics, exact row counts, and reproducibility, not a locked
performance value or superiority claim.

Phase 10 is in-memory and introduces no direct PostgreSQL or filesystem I/O;
there is no new Phase 10 database integration gate. Existing full-suite
PostgreSQL tests still run as a cross-phase gate. Phase 10 does not perform
feature-dependent learning, candidate/model/hyperparameter tuning,
cross-validation, advanced classification/regression, calibration,
artifact/model persistence, model
registration, risk mapping, forecasting, optimization, inference,
explainability, API/dashboard/container work, or any Phase 11+ behavior.

## 18. Advanced classification (Phase 11)

Phase 11 selects one fixed, feature-dependent classifier using validation
evidence. It does not persist a model or perform any later-phase work. Its
public module is `roadguard.classification` and its only workflow is:

```python
evaluate_advanced_classifier(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
    config: RoadGuardConfig,
) -> AdvancedClassificationEvaluation
```

`AdvancedClassificationError` is the contextual `ValueError` subclass for
invalid Phase 11 input, estimator, prediction, selection, or metric state.
The module `__all__` contains exactly `evaluate_advanced_classifier`, that
error, `CandidateValidationMetrics`, `TestClassificationMetrics`,
`AdvancedClassificationEvaluation`, and these constants:
`ADVANCED_CLASSIFIER_CONTRACT_VERSION`, `ADVANCED_CLASSIFIER_RNG_NAMESPACE`,
and `CANDIDATE_CLASSIFIER_NAMES`. The package root exposes the same symbols
while preserving every locked Phase 1-10 export.

All five arguments must be exact instances of their declared types; wrong
top-level types and lookalike objects raise `TypeError` before any field is
read. The evaluator never calls `load_config` and never reads a configuration
file or an environment variable. After exact-type validation, expected
lower-phase `FeatureInputError` and `PreprocessingError`, plus expected
scikit-learn estimator/metric `ValueError` or arithmetic failures, are
translated to `AdvancedClassificationError` with a fixed phase/context message
and suppressed exception chaining. Its public message contains no raw value,
row, object representation, filesystem path, environment value, database URL,
or credential. Phase 11 domain checks raise that error directly. Unexpected
programming failures and system-exiting exceptions are not relabeled as input
errors.

### Locked candidates, dependency, and seed derivation

Phase 11 adds no dependency: it uses the exact `scikit-learn>=1.5,<2`
environment already locked by Phase 10. The constants are exactly:

```python
ADVANCED_CLASSIFIER_CONTRACT_VERSION = "roadguard.phase11.v1"
ADVANCED_CLASSIFIER_RNG_NAMESPACE = 0x5247311
CANDIDATE_CLASSIFIER_NAMES = ("logistic_l2", "hist_gradient_boosting")
```

The candidate order above is observable selection provenance and is also the
final tie-break order. It is not caller configurable. For candidate index zero
or one in that exact order, the evaluator derives exactly one built-in `int`
seed from:

```python
int(
    np.random.SeedSequence(
        [config.seed, ADVANCED_CLASSIFIER_RNG_NAMESPACE, candidate_index]
    ).generate_state(1, dtype=np.uint32)[0]
)
```

It passes that derived seed as `random_state` to the corresponding constructor.
There is no public seed argument, global RNG use, or unseeded stochastic
operation. The supplied `RoadGuardConfig` is not returned, persisted, or
rendered. Only `config.seed` participates in evaluation; its `env`,
`data_dir`, `artifacts_dir`, and `database_url` fields are ignored and cannot
change a result. After exact top-level type validation, the evaluator reads
only `config.seed`, requires its exact built-in type to be `int` (not `bool`
or a subclass) and its value to be at least one. A forged or invalid seed
raises `AdvancedClassificationError` with a fixed configuration-seed message;
no other config field is validated, read, or represented in an error.

The `logistic_l2` candidate is exactly:

```python
LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    tol=1e-8,
    fit_intercept=True,
    class_weight=None,
    random_state=derived_seed,
)
```

The `hist_gradient_boosting` candidate is exactly:

```python
HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=derived_seed,
)
```

No caller-supplied estimator, parameter, feature matrix, threshold, or seed
is accepted. There is no grid/random/Bayesian search, cross-validation,
resampling, class weighting, calibration, clipping, or candidate beyond this
two-element set. The locked numerical-library versions in `uv.lock` bound
reproducibility.

### Input, provenance, and temporal boundary

The evaluator deep-copies caller-owned export frames, fresh-runs complete
cleaned-data validation through `build_feature_frame`, rebuilds the canonical
`split_chronologically` result, and requires the supplied split to equal it in
schema, dtypes, keys, dates, values, and partition membership. It calls
`fit_preprocessor` on the complete canonical split, requires the supplied fit
to equal the freshly reproduced fit field-for-field, and internally transforms
train and validation. A caller cannot supply transformed features, a target
array, a model, a threshold, a seed, or a lookalike input object.

Targets join one-to-one to each transformed partition solely by canonical
(`segment_id`, `date`) keys, and joined target order must exactly equal key
order. The sole target is `maintenance_within_30_days`. Keys, both target
columns, future event keys, `maintenance_history`, costs, materials, latent
generation fields, and every column outside
`PreprocessorFit.transformed_feature_columns` are forbidden model features.

Each candidate fits exactly once on the transformed 34-date training partition
and receives the transformed seven-date validation partition exactly once via
`predict_proba`. Validation is used only for the candidate records and the
selection rule below. Only after both candidate records, the selected name,
and the selected threshold are frozen may the test partition be transformed
exactly once and passed to one private test-evaluation boundary. The selected
candidate receives that test matrix exactly once through `predict_proba`; all
test hard labels and metrics derive solely from that one captured probability
vector. The loser must never receive a test matrix, produce a test prediction,
rank, or otherwise consume a test value. There is no public API that accepts a
fitted candidate and evaluates a test partition again, and no cross-call test
counter or state.

Changing only test feature/target values in a freshly valid matching export and
split may change only the selected candidate's test metrics. It cannot change
either fitted candidate state, any candidate validation record, the selected
name, threshold, feature schema, or row counts. Changing only validation
feature/target values may change validation records, selection, threshold, and
downstream selected-test metrics, but cannot change fitted candidate state or
the transformed training output. Equivalent shuffled upstream frames followed
by canonical rebuilding produce equal results for the same exact config and
locked numerical-library versions. Changing `config.seed` may alter only
stochastic candidate state and downstream evidence; it never changes source
validation, split, preprocessing fit, feature schema, or row counts.

### Validation selection and metric semantics

Training, validation, and test targets must each contain exactly integer
classes `[0, 1]`; the test-class check occurs only inside the frozen private
test stage. Every `predict_proba` output must be an `ndarray` of shape
`(partition_rows, 2)`, expose exact class order `[0, 1]`, and convert each
positive-class scalar exactly once to a finite built-in `float` in `[0.0, 1.0]`.
Invalid output is rejected, never clipped. `predict` is never used for a
reported hard-label metric.

For each candidate, threshold candidates are `0.0`, `1.0`, and each distinct
validation probability; a hard prediction is one when
`probability >= threshold`. The threshold is ranked by higher validation F1,
then higher validation recall, then higher threshold. Its candidate record
contains `average_precision_score` validation PR-AUC on probabilities plus F1
and recall at that candidate's selected threshold, both with `zero_division=0`.

The selected candidate is the record with higher validation PR-AUC, then higher
validation F1, then higher validation recall, then earlier position in
`CANDIDATE_CLASSIFIER_NAMES`. This is the complete ranking: no test metric,
baseline comparison, threshold value, random draw, or model internals may
break a tie. The selected candidate's frozen threshold produces test hard
labels. Test metrics are `accuracy_score`, `precision_score`, `recall_score`,
and `f1_score` (`zero_division=0` where accepted), `roc_auc_score` on positive
probabilities, and `confusion_matrix(labels=(0, 1))` stored as
`((true_negative, false_positive), (false_negative, true_positive))`.

All returned metrics and thresholds are finite built-in `float`s; counts are
exact built-in `int`s. Each validation PR-AUC/F1/recall and each test
accuracy/precision/recall/F1/ROC-AUC is in `[0.0, 1.0]`. The test confusion
matrix is exactly a two-by-two tuple of built-in non-negative `int`s whose sum
equals `test_rows`. Malformed shape, unexpected class ordering, invalid
estimator output, invalid metric output, out-of-range metric, or non-finite
metric raises `AdvancedClassificationError`; no partial result is returned.

### Frozen result schema

All result types are `@dataclass(frozen=True)` and expose no mutable frame,
array, estimator, mapping, prediction, seed, configuration, or caller-owned
object. Their fields and order are exactly:

```python
CandidateValidationMetrics(
    classifier_name: str,
    validation_pr_auc: float,
    decision_threshold: float,
    validation_f1: float,
    validation_recall: float,
)

TestClassificationMetrics(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float,
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]],
)

AdvancedClassificationEvaluation(
    contract_version: str,
    selected_classifier_name: str,
    feature_columns: tuple[str, ...],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    candidates: tuple[CandidateValidationMetrics, CandidateValidationMetrics],
    test: TestClassificationMetrics,
)
```

`contract_version` equals `ADVANCED_CLASSIFIER_CONTRACT_VERSION`, candidate
records preserve the exact locked order, `selected_classifier_name` equals the
`classifier_name` of exactly one candidate record, and `feature_columns`
equals the reproduced `PreprocessorFit.transformed_feature_columns`. V1 row
counts are exactly 10,200/2,100/2,100. Results contain no model, model parameters, coefficients,
feature importance, calibration claim, risk score/band, artifact path,
timestamp, environment detail, database credential, train metric, raw
prediction, or test-derived selection evidence.

### Failure, testing, and scope boundary

Invalid schemas/dtypes/keys/dates/values, duplicate or missing keys, target
misalignment, forged partition membership or preprocessing state, invalid
config seed, null/non-finite features, invalid targets, degenerate
classification partitions, invalid seed derivation, invalid estimator output, or
non-finite metrics fail contextually without mutating caller-owned objects.

RED-first tests must cover the absent public module before implementation;
exact public/package surface and frozen schema; both exact constructors and
derived seeds; direct scikit-learn known-vector metric comparisons; every
threshold tie-break; candidate selection PR-AUC, F1, recall, and fixed-order
tie-breaks; exact train-only fit inputs; validation-only selection; one private
selected-only test stage with exactly one test transform, exactly one selected
`predict_proba` call, and no loser test call; shuffled-input and same-config
determinism; changed-seed and ignored-config-field boundaries; caller
immutability; target/key alignment; every forbidden-field class; forged
export/split/fit/config-seed/lookalike inputs;
malformed, non-finite, and out-of-range probabilities; invalid metric output;
out-of-range score/confusion-matrix output; single-class partitions; and a
complete V1 evaluation. The V1 test asserts schema, row counts, finite/ranged
metrics, and reproducibility; it does not
lock performance values, require either candidate to win, or claim superiority
over Phase 10 baselines.

Phase 11 is in-memory and introduces no direct PostgreSQL or filesystem I/O;
the existing full-suite PostgreSQL tests remain a cross-phase gate. It does not
perform advanced regression, artifact/model persistence or registration, risk
mapping, material forecasting, optimization, inference, explainability,
API/dashboard/container work, or any Phase 12+ behavior.

## 19. Advanced regression (Phase 12)

Phase 12 selects one fixed, feature-dependent regressor using validation
evidence. It does not persist a model or perform any later-phase work. Its
public module is `roadguard.regression` and its only workflow is:

```python
evaluate_advanced_regressor(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
    config: RoadGuardConfig,
) -> AdvancedRegressionEvaluation
```

`AdvancedRegressionError` is the contextual `ValueError` subclass for invalid
Phase 12 input, estimator, prediction, selection, or metric state. The module
`__all__` contains exactly `evaluate_advanced_regressor`, that error,
`CandidateRegressionValidationMetrics`, `TestRegressionMetrics`,
`AdvancedRegressionEvaluation`, and these constants:
`ADVANCED_REGRESSOR_CONTRACT_VERSION`, `ADVANCED_REGRESSOR_RNG_NAMESPACE`,
and `CANDIDATE_REGRESSOR_NAMES`. The package root exposes the same symbols
while preserving every locked Phase 1-11 export.

All five arguments must be exact instances of their declared types; wrong
top-level types and lookalike objects raise `TypeError` before any field is
read. The evaluator never calls `load_config` and never reads a configuration
file or an environment variable. After exact-type validation, expected
lower-phase `FeatureInputError` and `PreprocessingError`, plus expected
scikit-learn estimator/metric `ValueError` or arithmetic failures, are
translated to `AdvancedRegressionError` with a fixed phase/context message and
suppressed exception chaining. Its public message contains no raw value, row,
object representation, filesystem path, environment value, database URL, or
credential. Phase 12 domain checks raise that error directly. Unexpected
programming failures and system-exiting exceptions are not relabeled as input
errors.

### Locked candidates, dependency, and seed derivation

Phase 12 adds no dependency: it uses the exact `scikit-learn>=1.5,<2`
environment already locked by Phase 10. The constants are exactly:

```python
ADVANCED_REGRESSOR_CONTRACT_VERSION = "roadguard.phase12.v1"
ADVANCED_REGRESSOR_RNG_NAMESPACE = 0x5247312
CANDIDATE_REGRESSOR_NAMES = ("ridge_l2", "hist_gradient_boosting")
```

The candidate order above is observable selection provenance and is also the
final tie-break order. It is not caller configurable. The supplied
`RoadGuardConfig` is not returned, persisted, or rendered. Only `config.seed`
participates in evaluation; its `env`, `data_dir`, `artifacts_dir`, and
`database_url` fields are ignored and cannot change a result. After exact
top-level type validation, the evaluator reads only `config.seed`, requires
its exact built-in type to be `int` (not `bool` or a subclass), and its value
to be at least one. A forged or invalid seed raises
`AdvancedRegressionError` with a fixed configuration-seed message; no other
config field is validated, read, or represented in an error.

`ridge_l2` is exactly:

```python
Ridge(
    alpha=1.0,
    fit_intercept=True,
    solver="svd",
    tol=1e-8,
    positive=False,
)
```

It is deterministic and receives no random state. For
`hist_gradient_boosting`, whose locked candidate index is one, the evaluator
derives exactly one built-in `int` seed from:

```python
int(
    np.random.SeedSequence(
        [config.seed, ADVANCED_REGRESSOR_RNG_NAMESPACE, candidate_index]
    ).generate_state(1, dtype=np.uint32)[0]
)
```

It passes that derived seed as `random_state` to this exact constructor:

```python
HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=derived_seed,
)
```

There is no public seed argument, global RNG use, unseeded stochastic
operation, caller-supplied estimator, parameter, feature matrix, target,
prediction, or candidate beyond this two-element set. The locked
numerical-library versions in `uv.lock` bound reproducibility.

### Input, provenance, and temporal boundary

The evaluator deep-copies caller-owned export frames, fresh-runs complete
cleaned-data validation through `build_feature_frame`, rebuilds the canonical
`split_chronologically` result, and requires the supplied split to equal it in
schema, dtypes, keys, dates, values, and partition membership. It calls
`fit_preprocessor` on the complete canonical split, requires the supplied fit
to equal the freshly reproduced fit field-for-field, and internally transforms
train and validation. A caller cannot supply transformed features, a target
array, a model, a prediction, a seed, or a lookalike input object.

Targets join one-to-one to each transformed partition solely by canonical
(`segment_id`, `date`) keys, and joined target order must exactly equal key
order. The sole target is `days_until_maintenance`. Keys, both target columns,
future event keys, `maintenance_history`, costs, materials, latent generation
fields, and every column outside `PreprocessorFit.transformed_feature_columns`
are forbidden model features.

Each candidate fits exactly once on the transformed 34-date training partition
and receives the transformed seven-date validation partition exactly once via
`predict`. Validation is used only for candidate records and the selection
rule below. Only after both candidate records and the selected name are frozen
may the test partition be transformed exactly once and passed to one private
test-evaluation boundary. The selected candidate receives that test matrix
exactly once through `predict`; all test metrics derive solely from that one
captured prediction vector. The loser must never receive a test matrix,
produce a test prediction, rank, or otherwise consume a test value. There is
no public API that accepts a fitted candidate and evaluates a test partition
again, and no cross-call test counter or state.

Changing only test feature/target values in a freshly valid matching export
and split may change only the selected candidate's test metrics. It cannot
change either fitted candidate state, any candidate validation record, the
selected name, feature schema, or row counts. Changing only validation
feature/target values may change validation records, selection, and downstream
selected-test metrics, but cannot change fitted candidate state or the
transformed training output. Equivalent shuffled upstream frames followed by
canonical rebuilding produce equal results for the same exact config and
locked numerical-library versions. Changing `config.seed` may alter only the
histogram-gradient-boosting candidate state and downstream evidence; it never
changes source validation, split, preprocessing fit, feature schema, or row
counts.

### Regression selection and metric semantics

Every `predict` output must be an `ndarray` of shape `(partition_rows,)`.
Each prediction scalar is converted exactly once to a finite built-in `float`.
Invalid output is rejected, never clipped or rounded; negative finite
predictions remain raw model output. Validation targets must be integer
`days_until_maintenance` values. Test targets must contain at least two
distinct values, and that test-target check occurs only inside the frozen
private test stage.

Each candidate record contains `mean_absolute_error` validation MAE and
`root_mean_squared_error` validation RMSE on its captured validation
predictions. The selected candidate is the record with lower validation MAE,
then lower validation RMSE, then earlier position in
`CANDIDATE_REGRESSOR_NAMES`. This is the complete ranking: no test metric,
baseline comparison, random draw, model internals, prediction magnitude, or
candidate name may break a tie.

The selected candidate's captured test predictions produce `mean_absolute_error`
test MAE, `root_mean_squared_error` test RMSE, and
`r2_score(force_finite=False)` test R-squared. MAE and RMSE are finite built-in
floats greater than or equal to zero. R-squared is a finite built-in float,
may be negative, and has no artificial `[0, 1]` range restriction. Invalid
shape, non-finite or non-numeric prediction, invalid metric output,
out-of-domain MAE/RMSE, non-finite R-squared, or a constant test target raises
`AdvancedRegressionError`; no partial result is returned.

### Frozen result schema

All result types are `@dataclass(frozen=True)` and expose no mutable frame,
array, estimator, mapping, prediction, seed, configuration, or caller-owned
object. Their fields and order are exactly:

```python
CandidateRegressionValidationMetrics(
    regressor_name: str,
    validation_mae: float,
    validation_rmse: float,
)

TestRegressionMetrics(
    mae: float,
    rmse: float,
    r2: float,
)

AdvancedRegressionEvaluation(
    contract_version: str,
    selected_regressor_name: str,
    feature_columns: tuple[str, ...],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    candidates: tuple[
        CandidateRegressionValidationMetrics,
        CandidateRegressionValidationMetrics,
    ],
    test: TestRegressionMetrics,
)
```

`contract_version` equals `ADVANCED_REGRESSOR_CONTRACT_VERSION`, candidate
records preserve the exact locked order, `selected_regressor_name` equals the
`regressor_name` of exactly one candidate record, and `feature_columns` equals
the reproduced `PreprocessorFit.transformed_feature_columns`. V1 row counts
are exactly 10,200/2,100/2,100. Results contain no model, model parameters,
coefficients, feature importance, calibration claim, risk score/band, artifact
path, timestamp, environment detail, database credential, train metric, raw
prediction, or test-derived selection evidence.

### Failure, testing, and scope boundary

Invalid schemas/dtypes/keys/dates/values, duplicate or missing keys, target
misalignment, forged partition membership or preprocessing state, invalid
config seed, null/non-finite features, invalid targets, a constant test
regression target, invalid seed derivation, invalid estimator output, or
non-finite metrics fail contextually without mutating caller-owned objects.

RED-first tests must cover the absent public module before implementation;
exact public/package surface and frozen schema; both exact constructors and
the histogram-gradient-boosting derived seed; direct scikit-learn known-vector
metric comparisons; MAE, RMSE, and fixed-order candidate-selection ties; exact
train-only fit inputs; validation-only selection; one private selected-only
test stage with exactly one test transform, exactly one selected `predict`
call, and no loser test call; shuffled-input and same-config determinism;
changed-seed and ignored-config-field boundaries; caller immutability;
target/key alignment; every forbidden-field class; forged
export/split/fit/config-seed/lookalike inputs; malformed, non-finite, and
wrong-shape predictions; invalid metric output; constant test targets; and a
complete V1 evaluation. The V1 test asserts schema, row counts, finite/raw
metrics, and reproducibility; it does not lock performance values, require
either candidate to win, or claim superiority over Phase 10 baselines.

Phase 12 is in-memory and introduces no direct PostgreSQL or filesystem I/O;
the existing full-suite PostgreSQL tests remain a cross-phase gate. It does not
perform calibration, hyperparameter search, cross-validation, artifact/model
persistence or registration, risk mapping, material forecasting, optimization,
inference, explainability, API/dashboard/container work, or any Phase 13+
behavior.

## 20. Frozen selection, artifacts, and risk mapping (Phase 13)

Phase 13 is the only V1 workflow that retains the exact train-fitted winners,
publishes their local artifact bundle, and maps the selected classifier's
captured test probabilities to the locked risk score and band. Its public
module is `roadguard.artifacts` and its only workflow is:

```python
persist_selected_artifacts(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
    config: RoadGuardConfig,
) -> FrozenSelectionResult
```

The module `__all__` contains exactly `persist_selected_artifacts`,
`FrozenSelectionError`, `ArtifactPersistenceError`, `ArtifactFile`,
`RiskOutput`, `SelectedArtifactManifest`, `FrozenSelectionResult`, and these
constants:

```python
FROZEN_SELECTION_CONTRACT_VERSION = "roadguard.phase13.v1"
ARTIFACT_FILENAMES = (
    "preprocessor.json",
    "classifier.joblib",
    "regressor.joblib",
    "test-risk.jsonl",
    "manifest.json",
)
RISK_BAND_NAMES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
```

The package root exposes the same symbols while preserving every locked Phase
1-12 export. `FrozenSelectionError` is the contextual `ValueError` subclass
for invalid inputs, reproduced selection state, probabilities, risk output, or
manifest values. `ArtifactPersistenceError` is the contextual `RuntimeError`
subclass for path, serialization, hashing, write, flush, synchronization,
collision, inventory, or atomic-publication failures.

All five arguments must be exact instances of their declared types; wrong
top-level types and lookalikes raise `TypeError` before any field is read.
Expected lower-phase validation/preprocessing errors, expected scikit-learn
estimator/metric failures, expected serialization failures, and expected
`OSError` or path failures are translated at their narrow boundary to one of
the two Phase 13 errors with a fixed context message and suppressed exception
chaining. Public errors contain no raw value, row, object or estimator
representation, absolute path, environment value, database URL, credential,
serialized byte, or temporary name. Unexpected programming failures and
system-exiting exceptions are not relabeled. The complete allowed public
message set is fixed to `Phase 13 input validation failed.`, `Phase 13
selection failed.`, `Phase 13 risk mapping failed.`, `Phase 13 artifact
serialization failed.`, `Phase 13 artifact path validation failed.`, `Phase 13
artifact write failed.`, `Phase 13 artifact verification failed.`, and `Phase
13 artifact publication failed.` No dynamic suffix or interpolated value is
permitted.

### Dependency, configuration, and path boundary

Phase 13 adds exactly one direct runtime declaration, `joblib>=1.5,<2`; the
already locked transitive `joblib 1.5.3` remains authoritative, so the lockfile
need not resolve to a new package version. No other dependency is added. Model
payloads use exactly `joblib.dump(value, binary_file, compress=0, protocol=5)`.
Phase 13 exposes no joblib/pickle load or deserialization API and never calls a
load function.

Only `config.seed` and `config.artifacts_dir` are read. The seed must be the
exact positive built-in `int` required by Phases 11-12. `artifacts_dir` must
satisfy `type(config.artifacts_dir) is type(Path())`, must not denote a
filesystem root, and is resolved only internally. `env`, `data_dir`, and
`database_url` are ignored and cannot change
any selected state, artifact byte, manifest, risk row, return value, or error.
The workflow never calls `load_config`, reads an environment variable, accepts
a caller path/filename/model/byte stream, or changes the current directory.

UNC/network syntax is rejected. A filesystem anchor such as `C:\` or `/` is a
permitted ancestor but cannot itself be `artifacts_dir`; every configured
component below the anchor and every created descendant is rejected when it is
a symlink, junction, reparse point, non-anchor mount point, or special file.
The root may already be an ordinary directory or may be created as one leaf
below verified existing ancestors. Unresolved traversal, device/reserved
targets, and any fixed descendant that resolves outside the root are rejected.
At each filesystem operation boundary, containment and file identity are
rechecked. Staging and digest destinations are siblings beneath the same
verified contract-version directory and must report the same device/volume.
Whether a non-UNC mounted filesystem is actually remote is a documented caller
precondition because Python path APIs cannot determine that portably. The
workflow may create only the contract-version directory, one fresh staging
directory, one digest directory, and the fixed names in `ARTIFACT_FILENAMES`.

### One private selection and capture orchestration

Phase 11 and Phase 12 public results intentionally contain metrics only; they
expose no fitted estimator or captured prediction. Phase 13 therefore does not
call either public evaluator and does not accept either result as input.
Instead, one private orchestration shares or extracts the already locked
selection primitives while preserving the public behavior of both earlier
modules. An internal refactor is permitted only when all existing Phase 11 and
Phase 12 tests remain unchanged and green.

The workflow deep-copies the four caller-owned export frames, fresh-runs the
complete cleaned-data and feature validation, reproduces the canonical 34/7/7
split, requires the supplied split to match in schema/dtypes/keys/dates/values
and membership, reproduces the train-only `PreprocessorFit`, and requires the
supplied fit to match field-for-field. Train and validation are each
transformed once for the shared Phase 13 orchestration.

The exact Phase 11 classifier constructors, two candidate seeds, positive
class semantics, validation threshold search, candidate metrics, and ranking
remain unchanged. The exact Phase 12 regressor constructors, HGB-only seed,
validation MAE/RMSE records, and ranking remain unchanged. Each of the two
classifiers and two regressors fits exactly once on the same canonical
train-only matrix and its own key-joined target. Each classifier receives the
validation matrix exactly once through `predict_proba`; each regressor receives
it exactly once through `predict`. The classifier and regressor selections are
both frozen before any test feature is transformed.

The retained selected classifier and regressor are the same train-fitted
instances that produced their selected validation records. Neither is refit
on train plus validation or on any other rows. Immediately after both
selections freeze and before any test transformation, the exact preprocessor,
selected-classifier, and selected-regressor payload bytes are captured in the
verified staging area. Those three payloads can therefore depend only on
train/validation selection state and never on a test feature or target. Losing
candidates are discarded and are never serialized. No tuning,
cross-validation, calibration, ensembling, baseline comparison, or new
candidate is performed.

After both selections are frozen, the canonical test feature partition is
transformed exactly once. Only the selected classifier receives it, exactly
once through `predict_proba`; the captured positive-class probability vector
is the sole probability source for all Phase 13 risk rows. Neither losing
classifier nor either regressor receives a test matrix. Complete lower-phase
validation may inspect test-target cells solely to enforce the locked
repository contract. After that validation boundary, Phase 13 never projects,
joins, hashes, serializes, scores, or otherwise uses test targets for
selection, artifacts, risk output, or returned values. Regression test
evaluation remains the responsibility of Phase 12 and is not repeated here.

### Risk output

Risk output contains exactly one row per canonical test key, in ascending
(`segment_id`, `date`) order. V1 therefore produces exactly 2,100 rows. The
selected classifier must retain exact `classes_ == ndarray([0, 1])`, and its
single test `predict_proba` output must be an `ndarray` of shape
`(test_rows, 2)`. Every captured positive-class scalar is converted exactly
once to a finite built-in `float` in `[0, 1]`; booleans, numeric-looking
strings, malformed shapes/classes, and non-finite or out-of-range values are
rejected, never clipped or rounded.

For each probability, `risk_score_from_probability` is called exactly once.
The resulting exact built-in `int` assigns the band only by these inclusive
ranges:

```text
LOW       0-30
MEDIUM   31-60
HIGH     61-80
CRITICAL 81-100
```

No alternative rounding or band implementation is permitted. In particular,
scores at probabilities `0.305`, `0.605`, and `0.805` are respectively 31,
61, and 81. The raw probability, score, and band must remain mutually
consistent. Probabilities are not described as calibrated because Phase 13
does not implement or evaluate calibration.

The public and persisted risk row fields are exactly `segment_id`, `date`,
`maintenance_probability`, `risk_score`, and `risk_band`. Targets, hard class
labels, regression predictions, thresholds, model internals, feature vectors,
and test metrics are absent.

### Canonical provenance fingerprints

`training_fingerprint` is exactly the lowercase SHA-256 defined by Phase 9. It
binds the canonical training feature/target evidence, full split-date
provenance, and `DatasetSpec`, while excluding validation/test values.

`selection_fingerprint` is SHA-256 over UTF-8 canonical JSON with sorted
object keys, compact separators `(',', ':')`, ASCII escaping, and non-finite
values forbidden. Its payload is exactly:

```json
{
  "columns": ["FEATURE_FRAME_COLUMNS + TARGET_COLUMNS[2:]"],
  "contract": "roadguard.phase13.v1",
  "spec": {
    "dataset_months_per_segment": 0,
    "dataset_observations": 0,
    "dataset_segments": 0
  },
  "validation_dates": ["YYYY-MM-DD"],
  "validation_rows": [["canonical scalar"]]
}
```

The real ordered columns, exact spec integers, validation dates, and rows
replace the placeholders. Rows are sorted by (`segment_id`, `date`). Scalar
canonicalization is exactly Phase 9: dates/datetimes use `YYYY-MM-DD`, strings
remain strings, integers remain JSON integers, and finite floats use lowercase
`float.hex()` strings with negative zero normalized to positive zero. The
fingerprint includes both supervised targets because both validation
selections depend on them. It excludes every test feature, test target, test
probability, risk value, filesystem value, and serialized artifact byte.

`risk_input_fingerprint` is SHA-256 over the same canonical JSON rules and
scalar encoding. Its payload is exactly:

```json
{
  "columns": ["FEATURE_FRAME_COLUMNS"],
  "contract": "roadguard.phase13.v1",
  "spec": {
    "dataset_months_per_segment": 0,
    "dataset_observations": 0,
    "dataset_segments": 0
  },
  "test_dates": ["YYYY-MM-DD"],
  "test_rows": [["canonical scalar"]]
}
```

The real ordered `FEATURE_FRAME_COLUMNS`, exact spec integers, test dates, and
rows replace the placeholders. Rows are sorted by (`segment_id`, `date`). The
fingerprint binds every target-free test feature consumed by risk mapping and
excludes both test targets, captured probabilities, risk values, filesystem
values, and serialized artifact bytes. Any valid test-feature change must
therefore change `risk_input_fingerprint`, the manifest bytes/digest, and the
relative artifact directory even when the selected classifier happens to emit
identical probabilities.

### Exact payload files and canonical bytes

The four payload files are produced completely before the manifest:

1. `preprocessor.json` contains exactly the immutable `PreprocessorFit` fields
   `scaled_columns`, `means`, `stds`, `province_categories`, and
   `road_type_categories`; tuples encode as arrays and floats encode as the
   Phase 9 canonical `float.hex()` strings.
2. `classifier.joblib` contains only the internally constructed selected
   train-fitted classifier. Its selected name, decision threshold, seed, and
   feature schema live in the manifest rather than in caller-controlled
   wrapper state.
3. `regressor.joblib` contains only the internally constructed selected
   train-fitted regressor.
4. `test-risk.jsonl` contains the exact risk rows. Each row is compact
   ASCII-escaped JSON with keys sorted lexicographically, a JSON-number raw
   finite probability, an ISO date string, and one `\n`; the file has no blank
   line and ends in exactly one `\n` when non-empty.

`manifest.json`, `preprocessor.json`, and every JSONL row use exactly
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`. The resulting text is encoded as UTF-8 and followed by
exactly one `\n`; every file uses `\n` line endings. Every persisted date uses
ISO `YYYY-MM-DD`. Public result floats remain built-in floats; manifest float
fields are stored as canonical lowercase `float.hex()` strings. No file
contains an absolute path, timestamp, hostname, username, environment name,
database value, credential, caller representation, test target, or unselected
estimator.

Each payload file is closed and synchronized before its exact byte size and
lowercase 64-character SHA-256 are recorded. `ArtifactFile` records preserve
this exact role order: `preprocessor`, `classifier`, `regressor`,
`test_risk`. The manifest does not list or hash itself. Its canonical bytes are
hashed after all four payload records are frozen; that digest is
`manifest_sha256` and cannot appear inside `manifest.json`, avoiding a
self-referential digest.

Runtime provenance records exactly these names in this order: `python` from
`platform.python_version()`, then `numpy`, `pandas`, `scikit-learn`, and
`joblib` from `importlib.metadata.version(...)` using those exact distribution
names. `platform_tag` is exactly `sysconfig.get_platform()`. Exact input/config,
platform tag, Python version, and locked library versions must produce
byte-identical payloads, manifest, digest, relative directory, and risk rows
after canonical rebuilding.

### Manifest and frozen result schemas

All public result types are `@dataclass(frozen=True)` and expose no estimator,
mutable frame/array/mapping, raw serialized bytes, absolute path, configuration
object, database value, target, regression prediction, or test metric. Their
fields and order are exactly:

```python
ArtifactFile(
    role: Literal["preprocessor", "classifier", "regressor", "test_risk"],
    filename: str,
    sha256: str,
    size_bytes: int,
)

RiskOutput(
    segment_id: str,
    date: date,
    maintenance_probability: float,
    risk_score: int,
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"],
)

SelectedArtifactManifest(
    contract_version: str,
    training_fingerprint: str,
    selection_fingerprint: str,
    risk_input_fingerprint: str,
    classifier_contract_version: str,
    regressor_contract_version: str,
    selected_classifier_name: str,
    selected_regressor_name: str,
    classifier_decision_threshold: float,
    master_seed: int,
    classifier_seed: int,
    regressor_seed: int | None,
    feature_columns: tuple[str, ...],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    train_dates: tuple[date, ...],
    validation_dates: tuple[date, ...],
    test_dates: tuple[date, ...],
    classification_candidates: tuple[
        CandidateValidationMetrics,
        CandidateValidationMetrics,
    ],
    regression_candidates: tuple[
        CandidateRegressionValidationMetrics,
        CandidateRegressionValidationMetrics,
    ],
    risk_bands: tuple[
        tuple[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"], int, int],
        ...,
    ],
    risk_output_rows: int,
    runtime_versions: tuple[tuple[str, str], ...],
    platform_tag: str,
    artifacts: tuple[ArtifactFile, ArtifactFile, ArtifactFile, ArtifactFile],
)

FrozenSelectionResult(
    manifest_sha256: str,
    relative_artifact_directory: str,
    manifest: SelectedArtifactManifest,
    risk_output: tuple[RiskOutput, ...],
)
```

`classifier_seed` is the exact selected Phase 11 candidate seed.
`regressor_seed` is `None` only for `ridge_l2`; otherwise it is the exact
selected Phase 12 HGB seed. Candidate records preserve their locked order and
the selected names/threshold reproduce the exact validation-only rules.
`feature_columns` equals the reproduced preprocessor columns; V1 rows are
10,200/2,100/2,100 and dates are 34/7/7. `risk_bands` is exactly
`(("LOW", 0, 30), ("MEDIUM", 31, 60), ("HIGH", 61, 80),
("CRITICAL", 81, 100))`. Runtime version order is fixed above.

The manifest JSON is the direct canonical JSON projection of
`SelectedArtifactManifest`; dataclass objects become objects, tuples become
arrays, dates become ISO `YYYY-MM-DD` strings, and floats use the canonical
rule above. `relative_artifact_directory` is exactly
`roadguard.phase13.v1/<manifest_sha256>` with forward slashes. The physical
final directory is that fixed relative directory beneath the configured
artifact root.

### Atomic publication, collision, and idempotence

All files are first created exclusively in a fresh verified sibling staging
directory on the same filesystem. Its concurrency-safe cryptographic nonce is
outside the section 8 model/data RNG policy: it never enters or
changes a payload, manifest, digest, relative directory, risk row, or returned
value. After serialization, close/sync, independent size/hash verification,
canonical manifest creation, and manifest close/sync, the complete staging
directory is published with one atomic directory rename. Atomicity means
visibility of the complete bundle to cooperating readers, not guaranteed
crash durability; the parent directory is synchronized when the platform
supports it. A success result is constructed only after the final directory
and exact five file inventory have been re-opened as bytes and verified without
deserializing models.

If the digest-named final directory already exists, Phase 13 reads bytes only,
requires exactly the five fixed regular files with no extras, validates each
expected size before hashing, and streams hashes with a fixed bounded buffer;
it never allocates from an attacker-controlled file size. It verifies the
canonical manifest plus every recorded size/hash. A byte-identical valid bundle
is idempotent and returned without overwrite. Missing, extra, link/reparse,
special, truncated, changed, or conflicting content raises
`ArtifactPersistenceError`; it is never overwritten, repaired, merged, or
deleted. A concurrent rename collision follows the same verify-identical or
fail-closed rule. Before an idempotent return or collision failure, the current
call's staging directory is removed through the safe cleanup rule below.

On any pre-publication failure, no result is returned and no new or modified
final bundle becomes visible; any pre-existing bundle remains untouched.
Cleanup may remove only the exact staging directory handle created by that call
after rechecking its identity, containment, and non-link status at the cleanup
boundary. It never removes or modifies the configured root, a completed
bundle, caller-owned content, or a path whose identity has changed.
Staging directories left by process termination are ignored and never adopted,
published, or removed merely because their names match the staging pattern.

### Isolation, failure, testing, and scope

Changing only valid test targets cannot change any returned or persisted Phase
13 value: complete lower-phase validation may inspect them for validity, but
no downstream Phase 13 computation consumes them. Changing only valid test
features must change `risk_input_fingerprint`, manifest bytes/digest, and the
relative directory; it may also change captured probabilities, risk rows,
`test-risk.jsonl`, and its artifact record. It cannot change train-fitted
model/preprocessor bytes, training/selection fingerprints, candidate
validation records, selected names/threshold/seeds, feature schema, or
row/date counts. Changing validation values may change selection evidence,
selected artifacts, and downstream risk output but cannot change preprocessing
or any candidate's train input/state before selection. Equivalent shuffled
upstream frames reproduce identical output after canonical rebuilding.

RED-first tests must cover the absent public module; exact public/package
surface and frozen schemas; direct joblib dependency and exact serialization
calls; exact four train fits and four validation predictions; exact Phase
11/12 constructors, seeds, metrics, thresholds, rankings, and retained winner
identity; both selections frozen before one test transform; one selected-only
classifier test call and no other test-model call; test probability vector
reuse; target-free 2,100-row V1 risk output; every score/band boundary and sole
use of `risk_score_from_probability`; training/selection fingerprints; exact
canonical bytes, file names/inventory/sizes/hashes, manifest digest, and
relative path; reproducibility and shuffle invariance; training, selection,
and risk-input fingerprints; test-feature and test-target isolation; ignored
config fields; caller immutability; malformed probabilities and forged inputs;
poisoned nested methods; secret/error
sanitization; operation-boundary path traversal, containment, symlink,
junction/reparse, special-file, and changed-identity rejection; partial writes
and every flush/hash/rename failure; collision, tampering, concurrency and
idempotence; and a complete V1 publication in two
isolated roots with byte-identical outputs. Tests patch every joblib/pickle load
entry point to fail if Phase 13 attempts deserialization.

Phase 13 performs local artifact writes only under the configured root. It
does not load models, register with MLflow, write predictions or artifacts to
PostgreSQL, accept API input, perform online/batch inference beyond the locked
test risk output, calibrate probabilities, refit on train plus validation,
tune/cross-validate, forecast materials, optimize maintenance, explain models,
serve an API/dashboard, build containers, or perform any Phase 14+ behavior.

Protection against a concurrently privileged local process replacing
filesystem components between an operation-boundary check and use is outside
the Phase 13 threat model. Phase 13 must still fail closed whenever such a
replacement is observable at a required boundary; it makes no race-free
filesystem guarantee against that actor.

## 21. Rolling-origin material forecasting (Phase 14)

Phase 14 is the only V1 workflow that turns complete realized material facts
into a deterministic next-month network forecast. Its public module is
`roadguard.forecasting` and its only workflow is:

```python
forecast_materials(
    dataset: RepositoryExport,
    maintenance_history: pd.DataFrame,
    spec: DatasetSpec,
) -> MaterialForecastEvaluation
```

The module `__all__` contains exactly `forecast_materials`,
`MaterialForecastError`, `ForecastCandidateMetrics`,
`MaterialForecastMetrics`, `MaterialForecast`,
`MaterialForecastEvaluation`, and these constants:

```python
MATERIAL_FORECAST_CONTRACT_VERSION = "roadguard.phase14.v1"
FORECAST_MATERIAL_NAMES = (
    "thermoplastic_paint_kg",
    "reflective_sheet_m2",
    "guardrail_meter",
    "traffic_sign_quantity",
)
FORECAST_CANDIDATE_NAMES = (
    "seasonal_naive_12",
    "trailing_mean_3",
)
INITIAL_TRAIN_MONTHS = 24
FROZEN_TEST_ORIGIN_COUNT = 7
FORECAST_HORIZON_MONTHS = 1
```

The package root exposes the same symbols while preserving every locked Phase
1-13 export. Phase 14 adds no dependency, configuration field, environment
variable, seed, or RNG use. The pure forecast workflow performs no database
or external I/O. The only Phase 14 database operation is the additive
read-only snapshot adapter defined below; Phase 14 performs no database write,
filesystem operation, artifact/model load, model serialization, or network
operation.

`MaterialForecastError` is the contextual `ValueError` subclass for invalid
inputs, completeness, folds, candidate output, metrics, fingerprints, or
forecast rows. All three arguments must be exact instances of their declared
types; lookalikes, subclasses, and wrong top-level types raise `TypeError`
before any field, label, or scalar is read. Expected validation, arithmetic,
or metric failures are translated at their narrow boundary with suppressed
exception chaining. Public errors contain no raw value, row, object
representation, path, environment value, database URL, credential, or SQL.
The complete allowed public messages are `Phase 14 input validation failed.`,
`Phase 14 rolling-origin evaluation failed.`, and `Phase 14 forecast output
failed.` Input/type-adjacent schema, lower-phase validation, completeness,
normalization, aggregation, calendar, fingerprint, and fold-construction
failures map to the input message. Candidate dispatch/output or metric
failures during validation or frozen test map to the rolling-origin message.
Final candidate dispatch/output, final row construction, or final result
invariant failures map to the forecast-output message. No dynamic suffix is
permitted. Unexpected programming failures and system-exiting exceptions are
not relabeled.

### Authoritative PostgreSQL input snapshot

Phase 14 adds exactly one method to `PostgresRepository`:

```python
export_material_forecast_inputs() -> tuple[RepositoryExport, pd.DataFrame]
```

The method reads `road_segments`, `road_observations`,
`observation_targets`, `maintenance_events`, and `maintenance_history` inside
one read-only `REPEATABLE READ` transaction. It never composes separately
timed calls to `export_dataset` and `aggregate_monthly_material_usage`, and it
never reads or writes `material_forecasts`. The returned `RepositoryExport`
has the exact existing canonical columns, dtypes, and natural-key ordering.
The history frame has exactly `MAINTENANCE_HISTORY_COLUMNS`, is ordered by
(`segment_id`, `maintenance_date`), and has object segment IDs,
`datetime64[ns]` dates, `int64` cost/sign quantities, and `float64` remaining
material quantities. Empty history preserves those exact columns and dtypes.

The existing `export_dataset`, history query, and monthly aggregation APIs are
unchanged. SQLAlchemy failures are raised as `PersistenceError` with exact
message `PostgreSQL material forecast input export failed` from the original
exception, matching the locked Phase 6 database-error boundary; no partial
pair is returned. Real-PostgreSQL tests prove
one-snapshot behavior under a concurrent committed change. The pure
`forecast_materials` function accepts the returned pair as its first two
arguments but never receives a repository or engine and never opens a
connection. Direct exact frames remain valid for deterministic unit fixtures,
subject to the same complete validation.

### Immutable result schemas

All result types are frozen dataclasses. `ForecastCandidateMetrics` fields are
exactly `candidate_name`, `validation_mae`, and `validation_rmse`.
`MaterialForecastMetrics` fields are exactly `material`, `candidates`,
`selected_candidate_name`, `test_mae`, and `test_rmse`; `candidates` is the
two-record tuple in `FORECAST_CANDIDATE_NAMES` order.
`MaterialForecast` fields are exactly `period`, `material`, and
`forecast_quantity`. `MaterialForecastEvaluation` fields are exactly:

```text
contract_version
forecast_input_fingerprint
history_start
history_end
forecast_period
validation_origins
test_origins
material_metrics
forecasts
```

Dates are built-in `date` values. Names are exact strings from the locked
tuples. Quantities and metrics are finite built-in `float` values and are
non-negative. `material_metrics` and `forecasts` each contain exactly four
records in `FORECAST_MATERIAL_NAMES` order. All nested collections are tuples;
no caller frame, mutable mapping, estimator, fold target, per-origin
prediction, database object, or internal array is exposed.

### Exact input and completeness boundary

`spec` is fresh-revalidated from its public fields and must have at least 32
months, because 24 initial months, at least one validation origin, and seven
frozen test origins are mandatory. The workflow deep-copies all four exact
`RepositoryExport` frames and fresh-runs the complete locked cleaned-data
validation against that spec. A stale/forged export, report-equivalent
lookalike, missing event, inconsistent point-in-time observation field, or
inconsistent target fails before forecast aggregation. Validation may inspect
supervised values solely to authenticate the lower-phase contract; after this
boundary, Phase 14 never projects, aggregates, hashes, scores, predicts from,
or returns an observation, supervised target, or segment attribute.

Fresh lower-phase validation is necessary but not sufficient for the Phase 14
calendar. The sorted unique authenticated observation-date tuple must equal
`observation_dates(spec.dataset_months_per_segment, V1_OBSERVATION_START)`
exactly. A consecutive first-of-month calendar shifted to any other start is
rejected rather than mapped onto the V1 history window. This exact calendar
check occurs before event/history filtering and is covered by a shifted-frame
RED test.

The history window begins at `V1_OBSERVATION_START` and contains exactly
`spec.dataset_months_per_segment` consecutive calendar months. `history_end`
is the first day of the final month and `forecast_period` is the first day of
the immediately following month.

The authenticated event frame has exactly `EVENT_COLUMNS` in order.
`maintenance_history` has exactly `MAINTENANCE_HISTORY_COLUMNS` in order.
Labels must be exact built-in strings with no duplicate-label trick. The
history frame is deep-copied before normalization. Keys are exact ASCII
segment IDs and timezone-free built-in dates or midnight timezone-free
`Timestamp` values. Duplicate natural keys are rejected. Every history key,
including a key outside the forecast window, must belong to the authenticated
event frame.

Inside the history window, the event-key set and realized-history key set must
be exactly equal. This deliberately strengthens the Phase 6 optional-history
boundary for Phase 14: a missing accounting row is incomplete evidence and is
never interpreted as zero. Events/history before `history_start` or after
`history_end` are validated but ignored and cannot change any Phase 14 return
value or call.

`maintenance_cost` is an exact positive built-in integer. The sign quantity is
an exact non-negative built-in integer. The other three material quantities
are exact built-in `int` or `float` values, excluding booleans, converted once
to finite non-negative floats. Phase 14 validates cost for record integrity
but never projects, aggregates, hashes, predicts from, or returns it. Changing
only a valid cost cannot change any result.

After completeness is proven, in-window realized rows are first sorted by
(`segment_id`, `maintenance_date`). Each row is assigned to the first day of
its event's calendar month. For each (`period`, `material`) group, values are
converted to built-in floats in that canonical row order and summed exactly
with `math.fsum`; the result is converted once to a built-in float and
negative zero is normalized to positive zero. In particular, the exact sign
count is converted once from built-in integer to float before `math.fsum`.
The workflow then constructs exactly one row for every
(`period`, `material`) pair in the history window. A missing pair at this
stage is a proven zero because every in-window event has complete realized
facts. All-zero histories and zero quantities are valid. Canonical history is
sorted by (`period`, `material`), contains exactly four materials for every
period, and has columns exactly `period`, `material`, and `quantity`. For V1
this is 48 periods and 192 rows. Segment identity, event count, event day,
cost, supervised observations/targets, and Phase 13 artifacts/risk rows are
not forecast inputs.

### Separate expanding rolling-origin protocol

The supervised 34/7/7 split is never imported, reproduced, or inspected.
Phase 14 uses one-step expanding rolling origins over each canonical material
series. Period index is zero-based over the accepted history:

- indices `0..23` are the initial 24-month prefix;
- validation target indices are `24..(month_count - 8)` inclusive;
- frozen test target indices are `(month_count - 7)..(month_count - 1)`;
- the final forecast target index is `month_count`.

V1 therefore has 17 validation origins (indices 24 through 40), seven frozen
test origins (41 through 47), and one final forecast for index 48. Every
candidate prediction for target index `t` receives only quantities at indices
strictly below `t`. Earlier validation actuals may enter later validation
prefixes. Selection freezes after all validation predictions and before any
test-candidate call. During the single test pass, only the selected candidate
for that material predicts each of its seven origins. An earlier test actual
may enter a later test prefix only after its period has elapsed; this is the
locked operational walk-forward protocol, not candidate reselection. Test
values never change candidate metrics or names. After test metrics are fixed,
the selected candidate uses the complete accepted history once to forecast
the immediately following month.

Changing or appending any valid event/history value after a fold's target
period cannot change that fold's candidate input or prediction. Changing only
the seven frozen test quantities may change test metrics and the final
forecast, but cannot change validation metrics or selected names. Changing
only the final history month cannot change any earlier-origin prediction.

### Candidates, metrics, and selection

Selection is independent per material; raw errors in kg, m2, metres, and count
are never added, averaged, weighted, or compared across materials. Candidate
order and formulas are exact:

1. `seasonal_naive_12` returns the quantity exactly 12 indices before the
   target.
2. `trailing_mean_3` returns `math.fsum` of the last three prefix quantities
   divided by `3.0`.

Candidates have no learned preprocessing, fit state, parameter search,
seasonality inference, clipping, rounding, random seed, or fallback. Each
candidate is called exactly once per validation origin and material. The
selected candidate alone is called exactly once per frozen test origin and
once for the final forecast. A returned scalar must be an exact built-in
`float`, finite and non-negative; invalid output fails rather than being
clipped or replaced.

Every candidate call, including the final forecast, is routed at call time
through the exact private seam
`_candidate_forecast(candidate_name: str, prefix: tuple[float, ...]) -> float`.
The seam dispatches only the two locked names. It returns
`float(prefix[-12])` for seasonal naive and
`float(math.fsum(prefix[-3:]) / 3.0)` for trailing mean, normalizing negative
zero to positive zero. Tests monkeypatch this seam to inspect immutable prefix
values, stage ordering, candidate names, exact call counts, and invalid
outputs; no function tuple may capture an unpatchable stale reference.

For an ordered actual vector `y` and forecast vector `p` of length `n`, metrics
use these exact formulas after every scalar has passed validation:

```text
MAE  = math.fsum(abs(y_i - p_i) for i in range(n)) / n
RMSE = math.sqrt(math.fsum((y_i - p_i) ** 2 for i in range(n)) / n)
```

Each result is converted once to a finite non-negative built-in float.
Validation ranking is lower MAE, then lower RMSE, then the earlier index in
`FORECAST_CANDIDATE_NAMES`. Test MAE/RMSE never affect selection. A valid
all-zero series deterministically selects `seasonal_naive_12` by the final
tie-break and forecasts zero. Forecast quantities remain floats, including
for `traffic_sign_quantity`, matching the locked database output schema.

### Canonical forecast-input fingerprint

`forecast_input_fingerprint` is lowercase SHA-256 over UTF-8 canonical JSON
with sorted object keys, compact separators `(',', ':')`, ASCII escaping, and
non-finite values forbidden. Dates use `YYYY-MM-DD`; strings remain strings;
integers remain JSON integers; finite floats use lowercase `float.hex()` with
negative zero normalized to positive zero. Its payload is exactly:

```json
{
  "candidates": [
    {"lag_months": 12, "name": "seasonal_naive_12"},
    {"name": "trailing_mean_3", "window_months": 3}
  ],
  "columns": ["period", "material", "quantity"],
  "contract": "roadguard.phase14.v1",
  "forecast_horizon_months": 1,
  "history_end": "YYYY-MM-DD",
  "history_rows": [["canonical scalar"]],
  "history_start": "YYYY-MM-DD",
  "initial_train_months": 24,
  "materials": ["FORECAST_MATERIAL_NAMES"],
  "spec": {
    "dataset_months_per_segment": 0,
    "dataset_observations": 0,
    "dataset_segments": 0
  },
  "test_origins": ["YYYY-MM-DD"],
  "validation_origins": ["YYYY-MM-DD"]
}
```

Real constants, spec integers, dates, material strings, and canonical rows
replace the placeholders. The fingerprint excludes segment IDs, event days,
event counts, cost, ignored out-of-window rows, metrics, selections,
predictions, forecasts, paths, configuration, environment, and database
state. Equivalent shuffled input and different valid row-index types reproduce
the same fingerprint and complete result.

### Isolation, RED tests, and scope

RED-first tests must cover the absent module/API; exact exports, signatures,
constants, frozen schemas, and package surface; exact input types/columns;
complete event-to-history coverage and missing-versus-proven-zero behavior;
natural keys, costs, material scalar rules, window boundaries, month
aggregation, four-series densification, and canonical ordering; minimum month
count, exact V1-start observation calendar, shifted-calendar rejection, and
exact validation/test/final origins; exact candidate formulas and
call counts; prefix-only validation, frozen per-material selection, one
selected-only test pass, and final forecast ordering; known manual metric and
tie vectors; no cross-unit aggregation; all-zero histories; negative and
non-finite candidate traps; V1 192-row history and four-row forecast;
fingerprint bytes and sensitivity/isolation; shuffle/index invariance; caller
immutability; hostile labels/scalars/frame subclasses; sanitized errors; and
poisoned environment, configuration, database, filesystem, artifact,
supervised-split, Phase 13 load, and RNG entry points around the pure workflow.
Separate unit and real-PostgreSQL tests cover the exact combined-snapshot
adapter, dtypes/order, empty history, concurrent-snapshot isolation, sanitized
database failure, and the absence of any `material_forecasts` read or write.

Phase 14 returns in-memory forecast evidence only. Complete lower-phase validation is
the only permitted inspection of observations and targets; valid changes that
preserve the authenticated event/history evidence cannot change any Phase 14
fingerprint, metric, selection, forecast, or call. Phase 14 does not write
`material_forecasts`, persist/load artifacts, register models, accept a
repository/engine/config/path/seed, forecast per segment, consume costs as a
forecast input, alter classification/regression features, tune candidates,
perform maintenance optimization, serve inference, explain models, expose an
API/dashboard, build containers, or perform any Phase 15+ behavior.

## 22. Exact offline maintenance prioritization (Phase 15)

Phase 15 is the only V1 workflow that converts the authenticated final Phase
13 held-out risk snapshot and an explicit caller-asserted prospective cost
scenario into an exact budget-feasible set of segments for human planning
review. It is an offline evaluation, not live inference or automated
maintenance execution.
Its public module is `roadguard.optimization` and its only workflow is:

```python
optimize_maintenance(
    selection: FrozenSelectionResult,
    expected_manifest_sha256: str,
    candidate_costs: tuple[MaintenanceCostInput, ...],
    budget_vnd: int,
) -> MaintenanceOptimizationResult
```

The module `__all__` contains exactly `optimize_maintenance`,
`MaintenanceOptimizationError`, `MaintenanceCostInput`,
`MaintenanceRecommendation`, `MaintenanceOptimizationResult`, and these
constants:

```python
MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION = "roadguard.phase15.v1"
MAINTENANCE_OPTIMIZATION_USE_CASE = "OFFLINE_EVALUATION_ONLY"
MAX_EXACT_VND = 2**63 - 1
V1_OPTIMIZATION_CANDIDATE_COUNT = 300
```

The package root exposes the same symbols while preserving every locked Phase
1-14 export. Phase 15 adds no dependency, configuration field, environment
variable, seed, or RNG use. The workflow performs no database, repository,
filesystem, artifact-load, model, forecast, network, or external I/O.

`MaintenanceOptimizationError` is the contextual `ValueError` subclass for
invalid authenticated source data, cost scenarios, fingerprints, optimization
state, or returned recommendations. All four arguments must be exact
instances of their declared types: `selection` must have exact type
`FrozenSelectionResult`, `expected_manifest_sha256` must have exact built-in
type `str`, `candidate_costs` must have exact type `tuple` and contain only
exact `MaintenanceCostInput` records, and `budget_vnd` must have exact built-in
type `int`. The function preflights all top-level types and every cost-tuple
element type before reading any cost-record field. It similarly requires exact
nested Phase 13 public record and tuple types before reading their fields.
Lookalikes, subclasses, booleans, NumPy scalars, mutable lists/mappings, and
other type violations raise `TypeError`; later value/domain violations map to
`Phase 15 input validation failed.` No hostile field method is invoked.

The recursive type preflight also requires exact `SelectedArtifactManifest`,
`ArtifactFile`, `RiskOutput`, classification/regression candidate-metric
records, every declared tuple container including each runtime-version pair,
and exact built-in date/string/integer/float leaf types before scalar-domain
validation. The observable order is recursive type checks, scalar-domain
checks, then comparisons, sorting, canonical projection, and hashing. A
missing or malformed field on an otherwise exact forged record is an input
validation failure; no supplied value controls attribute lookup behavior.

Expected validation, canonicalization, arithmetic, or optimization failures
are translated at their narrow boundary with suppressed exception chaining.
The complete allowed `MaintenanceOptimizationError` messages are `Phase 15
input validation failed.`, `Phase 15 optimization failed.`, and `Phase 15
output validation failed.` No dynamic suffix is permitted. Public errors
contain no raw segment, cost, budget, row, object representation, serialized
byte, path, environment value, database URL, credential, or internal state.
Unexpected programming failures and system-exiting exceptions are not
relabeled.

### Scientific and temporal meaning

The source Phase 13 probabilities predict occurrence of the locked
`maintenance_within_30_days` label and are explicitly uncalibrated. Phase 15
uses only their derived exact integer `risk_score` values as prioritization
utilities. Neither probability nor score is a deterioration hazard, causal
effect of an intervention, prevented failure count, avoided loss, expected
savings, or monetary benefit.

The evidence date is derived internally as the greatest exact date in the
authenticated `selection.manifest.test_dates`; callers cannot choose or shift
it. Exactly the risk row for that evidence date is a candidate decision for
each segment. Earlier held-out dates are authenticated as source provenance
but are never separate actions, objective terms, costs, or constraints. The
inclusive risk window ends 30 calendar days after the evidence date, preserving
the locked target boundary `days_until_maintenance <= 30`.

V1's latest Phase 13 evidence date and Phase 14 next-month forecast period do
not describe one common decision horizon. Phase 14 output predicts expected
network-level consumption, not inventory, procurement availability, or
capacity, and no per-segment action-to-material requirement exists. Phase 15
therefore does not accept or inspect `MaterialForecastEvaluation` and cannot
use forecast quantities as resource constraints. Adding material constraints
requires a later separately frozen contract with available-capacity and
per-action-demand semantics.

### Exact immutable schemas

All public records are `@dataclass(frozen=True)` with fields in exactly this
order:

```python
MaintenanceCostInput(
    segment_id: str,
    cost_vnd: int,
    cost_as_of_date: date,
)

MaintenanceRecommendation(
    priority_rank: int,
    segment_id: str,
    evidence_date: date,
    maintenance_probability: float,
    risk_score: int,
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    cost_vnd: int,
    cost_as_of_date: date,
)

MaintenanceOptimizationResult(
    contract_version: str,
    use_case: Literal["OFFLINE_EVALUATION_ONLY"],
    source_manifest_sha256: str,
    source_risk_input_fingerprint: str,
    optimization_input_fingerprint: str,
    evidence_date: date,
    risk_window_end: date,
    budget_vnd: int,
    selected_cost_vnd: int,
    remaining_budget_vnd: int,
    candidate_count: int,
    selected_count: int,
    total_risk_score: int,
    recommendations: tuple[MaintenanceRecommendation, ...],
)
```

Every exposed scalar has its exact annotated built-in type. Collections are
tuples. No caller object, frame, array, estimator, model payload, target,
feature vector, regression output, threshold, forecast, mutable mapping,
configuration, repository/database object, path, or credential is exposed.

`MaintenanceCostInput.cost_vnd` is an explicit prospective planning-scenario
value asserted by the caller to have been available on `cost_as_of_date`.
Historical `maintenance_cost` establishes its exact integer VND meaning but is
realized accounting data and is never automatically projected forward,
averaged, filled, or otherwise converted into an estimate by Phase 15. Every
`cost_as_of_date` must be an exact built-in `date` no later than the evidence
date. Phase 15 cannot authenticate the estimate's source, method, availability,
freshness, or lineage. The caller owns those preconditions; Phase 15 only makes
the asserted as-of date and exact value fingerprint-visible.

### Complete source and scenario validation

`expected_manifest_sha256` is a trust anchor supplied separately from
`selection`. The caller must obtain and approve it out of band from the Phase
13 publication boundary; copying an untrusted digest out of the same supplied
object does not meet this precondition. Phase 15 authenticates the complete
Phase 13 result against that trusted digest in memory before reading the
decision snapshot. It requires the exact Phase 13 contract version,
canonical manifest fields and artifact inventory, lowercase 64-character
digests, exact equality between the expected and returned manifest digests,
and the exact relative contract-version/digest directory. It
reprojects the manifest with the locked Phase 13 canonical JSON rules and
requires its SHA-256 to equal both digest values. It also serializes
every `RiskOutput` with the locked Phase 13 canonical JSONL rules and requires
the exact byte length and SHA-256 to match the manifest's `test_risk` artifact
record. No artifact file is opened.

The risk tuple must have the exact manifested row count, ascending
(`segment_id`, `date`) order, exact manifest test-date membership, no duplicate
key, and the same complete segment set on every test date. Phase 15 is V1-only:
it requires exactly seven test dates, exactly 2,100 risk rows, and exactly
`V1_OPTIMIZATION_CANDIDATE_COUNT` (300) unique segments on every date. It
therefore derives exactly 300 final-date candidates and rejects a larger or
smaller source before optimization.
Every segment ID, date, built-in finite probability in `[0, 1]`, exact
built-in integer score, and exact band is fresh-validated. The score must equal
the sole `risk_score_from_probability` result and the band must match the
locked inclusive ranges. Boundaries including probabilities `0.305`, `0.605`,
and `0.805` remain scores 31, 61, and 81.

`candidate_costs` must contain exactly one record for every final-date segment
and no missing, duplicate, or extra segment. Input tuple order is irrelevant;
canonical processing sorts by exact ASCII `segment_id`. Each `cost_vnd` is an
exact positive built-in integer at most `MAX_EXACT_VND`. `budget_vnd` is an
exact non-negative built-in integer at most `MAX_EXACT_VND`. Values are never
coerced, rounded, scaled, clipped, parsed, or converted through floating-point
arithmetic. All internal cost sums use exact Python integers and are checked
against the budget and signed-64-bit boundary.

### Exact objective, constraints, and algorithm

There is one indivisible binary choice per final-date segment: prioritize it
for human review or defer it. Every action is optional. There is no partial
action, repeated segment action, mandatory work, action type, scheduled date,
crew/geographic/fairness quota, or material constraint.

Among all subsets whose exact summed cost does not exceed `budget_vnd`, the
unique result is chosen by this lexicographic objective:

1. maximize the exact sum of selected integer `risk_score` values;
2. among equal-score subsets, minimize exact selected `cost_vnd`;
3. among equal-score, equal-cost subsets, scan candidates in ascending
   `segment_id` order and prefer selecting the candidate at the first differing
   binary decision.

The final tie rule is equivalent to the lexicographically smallest sorted
selected-ID tuple at this boundary because all candidate costs are strictly
positive; an equal-cost strict superset is impossible. Raw probabilities and
risk bands remain audit and display fields and never provide separate objective
weights.

The implementation is exact deterministic 0/1 dynamic programming indexed by
attainable total risk score, never by the VND budget. Candidates are processed
in ascending segment ID order. For each risk total, it retains only the state
with lower exact cost, then the state preferred by the final binary tie rule.
Risk states update in descending order for each candidate. With exactly 300 V1
candidates and score at most 100, the state domain has at most 30,001 values
and a full-domain scan has at most 9,000,300 candidate-state visits. The
selected set is reconstructed and every objective and constraint is recomputed
independently before return.
No greedy ratio, external solver, floating objective, heuristic, approximation,
timeout, randomization, fallback, or best-effort result is permitted.

A candidate transition is computed with an exact Python integer. If its cost
exceeds `budget_vnd`, it is discarded as infeasible without error, including
when adding two individually valid costs would exceed `MAX_EXACT_VND`. Every
stored state and returned cost is budget-feasible and therefore no greater
than `MAX_EXACT_VND`. Valid high-cost inputs never become an overflow error.

The empty subset is valid and is the unique result for zero budget, no
affordable positive-score candidate, or an all-zero-risk snapshot. A candidate
with score zero is never selected because it cannot improve the first
objective and strictly worsens the second. A budget that can afford every
positive-score candidate selects all such candidates. No valid input is
infeasible because all actions are optional.

Selected recommendations are presentation-ranked by descending `risk_score`,
then descending `maintenance_probability`, then ascending `cost_vnd`, then
ascending `segment_id`; `priority_rank` is the contiguous exact sequence from
1. This order does not change the optimized subset. Totals are recomputed from
the returned records: selected cost never exceeds budget, remaining budget is
the exact difference, selected segments are unique input candidates, and total
risk score is exact.

### Canonical optimization-input fingerprint

`optimization_input_fingerprint` is lowercase SHA-256 over UTF-8 canonical
JSON with sorted object keys, compact separators `(',', ':')`, ASCII escaping,
and non-finite values forbidden. Dates use `YYYY-MM-DD`; strings remain
strings; integers remain JSON integers; finite floats use lowercase
`float.hex()` with negative zero normalized to positive zero exactly as in the
Phase 9 canonical scalar rules. Its payload is exactly:

```json
{
  "budget_vnd": 0,
  "candidates": [["segment_id", "evidence_date", "probability_hex", 0,
                  "risk_band", 0, "cost_as_of_date"]],
  "columns": ["segment_id", "evidence_date", "maintenance_probability",
              "risk_score", "risk_band", "cost_vnd", "cost_as_of_date"],
  "contract": "roadguard.phase15.v1",
  "objective": ["maximize_total_risk_score", "minimize_total_cost_vnd",
                "prefer_selected_lower_segment_id_at_first_difference"],
  "source_manifest_sha256": "lowercase sha256",
  "source_risk_input_fingerprint": "lowercase sha256",
  "use_case": "OFFLINE_EVALUATION_ONLY"
}
```

Real values replace the placeholders and candidate rows are sorted by
`segment_id`. The fingerprint binds the full Phase 13 manifest by digest and
therefore indirectly binds all authenticated earlier risk dates, while the
candidate rows bind every final-date risk/cost value actually optimized. It is
sensitive to source, latest risk, cost, as-of date, and budget, and invariant
to input cost tuple order. It excludes output selection, metrics, paths,
configuration, environment, database state, Phase 14 forecasts, and all later
phase inputs.

### Isolation, RED tests, and scope

RED-first tests must cover the absent module/API; exact exports, signature,
constants, frozen field order, and package surface; exact top-level and nested
types; manifest digest and risk JSONL size/hash authentication; risk row counts,
dates, keys, ordering, probability/score/band consistency, and boundary scores;
complete cost coverage and as-of rules; exact VND zero/positive/maximum/overflow
boundaries; zero budget, all unaffordable, all-zero risk, exact fit, and all
positive-score candidates affordable; known manual optima and a greedy
counterexample; exhaustive small fixed-case oracle comparison; all three
objective tie-breaks; reconstruction and total invariants; 300-candidate scale;
fingerprint bytes, sensitivity, and isolation; shuffled-cost equivalence;
caller immutability; hostile values and sanitized failures; and poisoned
environment, configuration, database/repository, filesystem, artifact loading,
model prediction/training, forecasting, RNG, network, inference, explainability,
and API entry points.

Phase 15 returns in-memory offline recommendation evidence only. It does not
estimate costs, read realized future cost, consume material forecasts, load or
run models, infer new risk, use regression outputs, estimate causal effects,
schedule work, allocate crews/materials, solve multiple periods, persist or
load recommendations, add or repurpose a PostgreSQL table, accept a
repository/engine/config/path/seed, expose an API/dashboard, build containers,
or perform any Phase 16+ behavior.
