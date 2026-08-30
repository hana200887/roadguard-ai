# Phase 15 maintenance optimization — RED evidence

```powershell
uv run pytest tests/test_optimization.py tests/test_package.py -q
```

Result: RED — `15 failed, 1 passed in 3.08s`.

The 14 Phase 15 tests failed with `ModuleNotFoundError: No module named
'roadguard.optimization'`; `tests/test_package.py::test_public_api_surface`
failed because `MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION` is not yet exported
from `roadguard`.
