# Phase 13 frozen artifacts — TDD evidence

## Source and scope

The journeys and acceptance criteria come from `docs/contracts.md` section 20
and the Phase 13 acceptance contract in `docs/phase-plan.md`. This evidence
covers only frozen train-fitted winner retention, canonical local artifact
publication, manifested provenance, and target-free test risk mapping. Phase
14 and later behavior is excluded.

## RED

Tests were added before `roadguard.artifacts` existed.

```text
uv run pytest tests/test_artifacts.py -q
4 failed in 3.81s
ModuleNotFoundError: No module named 'roadguard.artifacts'
```

The failure was caused by the absent Phase 13 public module, not by a broken
fixture, missing dependency, or unrelated regression. The RED checkpoint is
commit `cf0d0e9`.

## GREEN progression

The first implementation run reached `4 passed, 2 failed`; both failures were
the same canonical manifest serialization defect (`regressor_seed=None` was
not yet projected as JSON null). After the minimal correction, the original
focused suite passed. Subsequent adversarial tests added provenance,
idempotence, tamper, bounded-read, path, staging-identity, no-clobber publish,
and orchestration call-order coverage.

```text
uv run pytest tests/test_artifacts.py tests/test_package.py -q
16 passed, 1 skipped
```

The inherited scikit-learn warning concerns the locked Phase 11 logistic
constructor's deprecated `penalty` argument; Phase 13 does not change that
locked constructor.

## Guarantees exercised

| Guarantee | Evidence |
| --- | --- |
| Exact module/package surface, constants, frozen dataclasses, and manifest field order | `tests/test_artifacts.py`, `tests/test_package.py` |
| Same input/runtime produces byte-identical bundles in isolated roots | two-root reconstruction test |
| Existing identical bundle is idempotent; tampered bundle fails closed | idempotence and tamper tests |
| A valid test-feature change changes only test-input provenance and downstream manifest identity | risk-input fingerprint isolation test |
| Winner payloads are captured after train/validation selection and before the single selected-classifier test pass | traced orchestration test |
| Phase 11/12 public evaluators and joblib/pickle load APIs are never used | poisoned-entry-point orchestration test |
| Publication is no-clobber; changed-identity staging is not removed | filesystem adversarial tests |
| Manifest reads are bounded and existing payload sizes/hashes are verified | oversized-manifest and bundle verification tests |
| Filesystem roots and non-anchor mount points are rejected | path-policy tests |
| Wrong top-level input types fail before fields are read | exact-type boundary test |

## Regression and coverage evidence

The final non-PostgreSQL regression gate completed as follows:

```text
uv run pytest --ignore=tests/test_postgres_integration.py \
  --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80
997 passed, 1 skipped, 63 warnings in 854.86s
Total coverage: 86.29%
roadguard.artifacts: 80%
roadguard._artifact_io: 81%
```

After Docker was enabled, the targeted disposable-PostgreSQL gate completed:

```text
28 passed in 13.11s
```

The final complete suite against a real disposable PostgreSQL instance then
completed as follows:

```text
1025 passed, 1 skipped, 63 warnings in 892.18s
Total coverage: 91.71%
roadguard.artifacts: 80%
roadguard._artifact_io: 81%
```

## Review history

Terra High independently reviewed the implementation and repaired four
filesystem findings: overwrite-capable rename, changed-identity staging
cleanup, unbounded existing-manifest reads, and non-anchor mount acceptance.
A second focused pass added temporal call-order and prohibited-loader probes.
The final review fixes added ancestor validation before creating an artifact
root leaf and parent-directory synchronization after atomic rename with
sanitized publication-failure mapping. The final blocker-only review is
APPROVED with no P1/P2 findings. Final coverage, PostgreSQL, static, and
blocker-only review gates are green; staged-diff, fast-forward, and push checks
remain.
