# Phase 7 TDD Evidence — Point-in-time feature frame

## Source and scope

The journeys and acceptance criteria are derived from
[`docs/phase-plan.md`](../phase-plan.md) and the Phase 7 addendum in
[`docs/contracts.md`](../contracts.md). Phase 7 is limited to a frozen,
target-free feature registry and deterministic feature frame from a complete
Phase 6 `RepositoryExport`. Splitting, imputation, encoding, scaling, model
work, API work, database writes, and unlisted engineered features are out of
scope.

## User journeys

- As a modelling workflow, I need a canonical point-in-time feature frame from
  the Phase 6 export so later phases can split and preprocess it safely.
- As a reviewer, I need targets, future events, costs, materials, and latent
  fields excluded so a caller cannot leak a label into a model feature.
- As an operator, I need invalid or forged exported data to fail before a
  feature frame is returned.

## RED and GREEN evidence

| Guarantee | Test evidence | Result |
| --- | --- | --- |
| The Phase 7 public module and registry must exist before implementation. | `uv run pytest tests/test_features.py -q` | RED: `ModuleNotFoundError: roadguard.features` before implementation. |
| Label-encoded `previous_repairs` is rejected, and datetime output is canonical. | `uv run pytest tests/test_features.py -k "target_encoded or normalizes_phase6_datetime" -q` | RED: 2 failures before the provenance/dtype fix. |
| The frozen registry, target exclusion, canonical ordering, input immutability, future-event/target invariance, provenance guard, dtype normalization, public export, and 14,400-row V1 path work. | `uv run pytest tests/test_features.py tests/test_package.py -q` | GREEN: 12 passed in 18.46s. |
| The non-PostgreSQL project suite remains compatible with Phase 1–6. | `uv run pytest --ignore=tests/test_postgres_integration.py --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80` | GREEN: 590 passed in 222.17s; total coverage 85.21%, `roadguard.features` 93%. |

## Guarantees covered

| # | What is guaranteed | Test |
| --- | --- | --- |
| 1 | Registry names and source order are immutable and explicit. | `test_registry_has_exact_source_feature_order` |
| 2 | The public builder returns only keys plus approved features. | `test_build_feature_frame_has_exact_target_free_schema` |
| 3 | Row order does not affect output and input frames are untouched. | `test_build_feature_frame_is_canonical_and_does_not_mutate_inputs` |
| 4 | A valid future event with targets freshly re-derived cannot change features. | `test_valid_future_event_and_rederived_targets_cannot_change_feature_values` |
| 5 | A forged target frame fails fresh validation. | `test_build_feature_frame_revalidates_forged_export` |
| 6 | A maintenance feature encoded from the label fails the strictly-prior event provenance check. | `test_build_feature_frame_rejects_target_encoded_maintenance_feature` |
| 7 | Object-date source values produce canonical `datetime64[ns]` output without mutating the source. | `test_build_feature_frame_normalizes_phase6_datetime_output` |
| 8 | The V1 feature frame preserves all 14,400 canonical rows. | `test_v1_feature_frame_preserves_all_canonical_rows` |

## Remaining external gate

The exact complete-suite command was run in this environment:

```text
uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80
```

It reached 590 passing tests and 85.21% coverage, then reported 28 deliberate
errors because `TEST_DATABASE_URL` is unset and the PostgreSQL integration
fixture fails closed with `real PostgreSQL required; Phase 6 INCOMPLETE`.
This is not recorded as a pass. A real disposable PostgreSQL URL is required
to complete that external gate and move Phase 7 from `active` to
`accepted-local`.

`pip-audit` was not installed in the locked environment, so no new dependency
audit result is asserted here.
