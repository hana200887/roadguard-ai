# Phase 15 maintenance optimization — RED evidence

```powershell
uv run pytest tests/test_optimization.py tests/test_package.py -q
```

Result: RED — `15 failed, 1 passed in 3.08s`.

The 14 Phase 15 tests failed with `ModuleNotFoundError: No module named
'roadguard.optimization'`; `tests/test_package.py::test_public_api_surface`
failed because `MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION` is not yet exported
from `roadguard`.

## GREEN evidence

```powershell
uv run pytest tests/test_optimization.py tests/test_package.py -q
```

Initial Terra GREEN result: `16 passed in 2.97s`.

```powershell
uv run coverage run --branch -m pytest tests/test_optimization.py tests/test_package.py -q
uv run coverage report --include=src/roadguard/optimization.py --show-missing --fail-under=80
```

Initial Terra result: `16 passed in 7.30s`; focused
`roadguard.optimization` coverage was 87% (337 statements, 158 branches).

```powershell
uv run ruff check src/roadguard/optimization.py src/roadguard/__init__.py tests/test_optimization.py tests/test_package.py
uv run ruff format --check src/roadguard/optimization.py src/roadguard/__init__.py tests/test_optimization.py tests/test_package.py
uv run mypy src
git diff --check
```

Result: Ruff reported `All checks passed!`, all four checked files were
formatted, mypy reported `Success: no issues found in 32 source files`, and
the diff check passed.

## Independent Sol review and fixes

The first GREEN implementation passed its tests but independent adversarial
review found malformed segment acceptance; partial Phase 13 manifest
validation; late fixed-shape rejection; provenance rereads; shared-solver
output validation; inexact output leaf types; and a hostile undeclared
attribute path through generic deep copying. Sol replaced generic copying with
declared-field snapshots, split canonical manifest validation into a private
helper, added a structurally separate sparse optimality oracle, and replaced
RNG-based seed authentication with a locked pure-integer verifier. Regression
coverage includes every finding and poisons both hostile deep-copy hooks and
NumPy `SeedSequence` during a valid workflow.

```powershell
uv run pytest tests/test_optimization.py tests/test_optimization_adversarial.py tests/test_package.py -q
uv run coverage erase
uv run coverage run --branch -m pytest tests/test_optimization.py tests/test_optimization_adversarial.py tests/test_package.py -q
uv run coverage report --include='src/roadguard/optimization.py,src/roadguard/_optimization_manifest.py' --show-missing --fail-under=80
```

Final focused result: `21 passed in 3.24s`; coverage-run result:
`21 passed in 8.11s`. Combined Phase 15 production coverage is 89% over 481
statements and 208 branches, above the required 80% branch-aware threshold.
Independent security/provenance re-review approved the corrected implementation
with no P1/P2 findings remaining.

## Final cross-phase gate

With a disposable PostgreSQL 16 container available only on local port 54329:

```powershell
$env:TEST_DATABASE_URL='<redacted disposable local PostgreSQL URL>'
uv run pytest --cov=roadguard --cov-report=term-missing --cov-branch --cov-fail-under=80
```

Result: `1070 passed, 1 skipped, 63 warnings in 806.00s`; total branch-aware
coverage was 91.63%. The single skip is the existing Windows-policy junction/
symlink case. The warnings are the existing scikit-learn 1.9 deprecation notice
for the locked Phase 11 logistic-regression `penalty` argument; no Phase 15
failure or warning was emitted.
