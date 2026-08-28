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

Pending implementation and independent review. Exact targeted, PostgreSQL,
full-suite, coverage, lint, format, type-check, and diff-check results will be
recorded here only after they have run successfully.
