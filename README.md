# RoadGuard AI

Predictive maintenance and risk intelligence for road infrastructure.

## Status

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
- No observations, targets, anomalies, material quantities, database or
  models yet; targets are never used during event generation.
- Test harness: `pytest` with `pytest-cov` (branch coverage, enforced
  `fail_under = 80`), `ruff`, and `mypy` (strict).

Not yet implemented (future phases): synthetic data generation, database
access, ML models, API, dashboard, and Docker services.

## Quickstart

```powershell
uv sync --frozen
uv run pytest --cov=roadguard --cov-branch --cov-report=term-missing --cov-fail-under=80
uv run ruff check .
uv run mypy src
```

## Configuration

Only runtime settings are configurable: `env`, `seed`, `data_dir`, and
`artifacts_dir`. The V1 data/ML contract is locked and cannot be overridden.

Configuration is resolved in this order (later wins):

1. Built-in defaults (documented in `src/roadguard/config.py`).
2. YAML file given as `load_config(path)` argument or via the
   `ROADGUARD_CONFIG_PATH` environment variable (reserved control variable).
3. Environment variables named `ROADGUARD_<FIELD>` (for example
   `ROADGUARD_SEED=7`).

Invalid or unknown settings raise `pydantic.ValidationError`; unreadable
files and unsupported `ROADGUARD_*` environment variable names raise
`roadguard.ConfigError`. Boolean values are rejected for numeric fields;
numeric strings such as `ROADGUARD_SEED=42` are accepted. Phase 1 reads
process environment variables only; `.env` files are not parsed
automatically. See `.env.example` for supported variable names and safe
example values.

## Contracts

All cross-phase guarantees (dataset schema, target semantics, time split
policy, feature availability, artifact/inference contracts, and the
reproducibility seed policy) are defined in `docs/contracts.md`.
