"""Probability to risk score mapping for the locked V1 contract."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

__all__ = ["risk_score_from_probability"]


def risk_score_from_probability(probability: float) -> int:
    """Map a maintenance probability to the V1 integer risk score (0-100).

    Uses decimal ROUND_HALF_UP arithmetic, so binary floating-point
    representation cannot change .5 behaviour:

        risk_score = Decimal(str(probability)) * 100  quantized to 1 with
        ROUND_HALF_UP

    Booleans are rejected, the probability must be finite and between 0 and
    1 inclusive, and the result is clamped to 0-100 only for floating-point
    numerical safety. The locked bands in :mod:`roadguard.contracts`
    classify the resulting score.
    """
    if isinstance(probability, bool):
        raise ValueError("probability must be a numeric value, not a boolean")
    if not math.isfinite(probability):
        raise ValueError("probability must be finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1 inclusive")
    score = (Decimal(str(probability)) * Decimal("100")).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return max(0, min(100, int(score)))
