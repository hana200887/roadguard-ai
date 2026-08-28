# Phase 14 forecasting TDD evidence

This record tracks the RED-first implementation of the authoritative Phase 14
contract in `docs/contracts.md` section 21. Phase 14 remains limited to
deterministic rolling-origin network-month material forecasting and its
read-only PostgreSQL input snapshot adapter.

## RED

The contract tests and repository-adapter tests were written before any Phase
14 production module or method existed. The first targeted run was:

```text
uv run pytest tests/test_forecasting.py -q
```

It failed during collection with:

```text
ModuleNotFoundError: No module named 'roadguard.forecasting'
1 error
```

An independent rerun reproduced the same missing-module failure. This is the
intended RED condition: the Phase 14 public implementation was absent, while
the test files compiled successfully. It was not caused by a broken fixture,
missing test dependency, or environment failure.

## GREEN

Terra High implemented the pure workflow, root exports, and the additive
PostgreSQL snapshot adapter. Sol High then independently reviewed the
implementation and added adversarial regression coverage. The review found
and fixed two P2 error-boundary defects: huge built-in material integers could
leak `OverflowError`, and fingerprint failures were mapped to the output stage
instead of the locked input stage. A second review found no remaining P1/P2
finding.

The final targeted commands passed:

```text
uv run pytest tests/test_forecasting.py tests/test_package.py tests/test_database_unit.py -q
35 passed

uv run pytest tests/test_postgres_integration.py -k phase14 -q
4 passed, 28 deselected
```

The PostgreSQL tests used a disposable PostgreSQL 16 instance and exercised
the read-only repeatable-read snapshot, concurrent snapshot isolation, exact
empty/non-empty dtypes, sanitized failure, and absence of any
`material_forecasts` access.

The final full-suite gate was:

```text
uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80
1051 passed, 1 skipped
Total coverage: 91.93%
roadguard.forecasting: 96%
```

The one skip is the pre-existing Windows policy gate for symlink/junction
creation; it is unrelated to Phase 14. Frozen dependency sync, Ruff lint,
Ruff format checking, strict mypy over `src`, lock checking, and
`git diff --check` also passed before the implementation commit.
