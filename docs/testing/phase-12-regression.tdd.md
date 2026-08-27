# Phase 12 advanced-regression TDD evidence

Authority: `docs/contracts.md` section 19 and the Phase 12 acceptance contract
in `docs/phase-plan.md`.

## RED

The public contract tests were written before the production module and run
with:

```powershell
uv run pytest tests/test_regression.py tests/test_regression_adversarial.py tests/test_package.py -q
```

Pytest stopped during collection with two errors:

```text
ModuleNotFoundError: No module named 'roadguard.regression'
2 errors during collection
```

The RED checkpoint is commit `6e118cc` (`test: add RED Phase 12 regression
contract`).

Independent review later found that numeric-looking strings and poisoned exact
`DataFrame` instances crossed two defensive boundaries. New adversarial tests
were added before each repair and produced genuine intermediate RED results:

```text
3 failed, 20 passed
2 failed, 23 passed
```

## GREEN

After the minimum implementation and review fixes:

```powershell
uv run pytest tests/test_regression.py tests/test_regression_adversarial.py tests/test_package.py -q
```

```text
51 passed in 36.52s
```

The full cross-phase gate used the disposable PostgreSQL test service through
`TEST_DATABASE_URL` (the credential is intentionally not recorded here):

```powershell
uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80
```

```text
1011 passed, 48 warnings in 766.54s
Total coverage: 93.20%
roadguard.regression: 84% with branch coverage
```

The warnings are the existing scikit-learn 1.9 deprecation warning for the
locked Phase 11 logistic-regression `penalty` argument; Phase 12 adds no warning
and may not change that frozen Phase 11 constructor.

## Review guarantees

- Both exact locked candidates fit once on the reproduced train matrix and
  target vector, then predict validation once.
- Selection uses validation MAE, validation RMSE, then locked candidate order.
- The selected candidate alone receives the test matrix once after selection;
  the loser never receives test data.
- Fresh-valid test-target and test-feature changes cannot affect candidate
  records, selection, feature schema, or row counts.
- Ridge receives no seed; HGB receives exactly one seed derived from the locked
  namespace and candidate index one.
- Numeric-looking strings, malformed or non-finite predictions, invalid metric
  outputs, and constant test targets fail contextually.
- Trusted unbound pandas operations prevent instance-poisoned exact
  `DataFrame` methods from leaking raw exception values.
- Caller-owned exports, split frames/dates, fit, spec, and config remain
  unchanged; forbidden keys, targets, future-event and latent fields never
  enter the model matrix.

Two independent read-only re-reviews approved the repaired implementation with
no remaining P1/P2 findings. A possible future service-level resource cap was
recorded as P3 and is outside the frozen in-memory Phase 12 scope.

## Diagnostic note

A non-gate probe using `--cov=roadguard.regression` triggered a Windows NumPy
double-import loader error during collection. Re-running with the repository's
official package-level `--cov=roadguard` command completed normally; both the
targeted suite and the required full coverage gate are green.
