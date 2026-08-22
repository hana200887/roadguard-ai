"""Tests for the probability-to-risk-score mapping."""

from __future__ import annotations

import pytest

from roadguard import risk_score_from_probability


class TestRiskScoreMapping:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.0, 0),
            (0.005, 1),
            (0.01, 1),
            (0.30, 30),
            (0.305, 31),
            (0.31, 31),
            (0.60, 60),
            (0.605, 61),
            (0.61, 61),
            (0.80, 80),
            (0.805, 81),
            (0.81, 81),
            (1.0, 100),
        ],
        ids=[
            "zero",
            "half_up_floor_case",
            "low_lower_bound",
            "low_upper_bound",
            "medium_lower_bound_halves",
            "medium_lower_bound",
            "medium_upper_bound",
            "high_lower_bound_halves",
            "high_lower_bound",
            "high_upper_bound",
            "critical_lower_bound_halves",
            "critical_lower_bound",
            "one",
        ],
    )
    def test_boundary_mapping(self, probability: float, expected: int) -> None:
        assert risk_score_from_probability(probability) == expected

    def test_round_half_up_is_not_banker_s_rounding(self) -> None:
        assert risk_score_from_probability(0.305) == 31

    def test_float_safety_clamp_stays_within_range(self) -> None:
        assert 0 <= risk_score_from_probability(0.999_999_999_999_999_9) <= 100

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.145, 15),
            (0.285, 29),
            (0.565, 57),
            (0.575, 58),
        ],
        ids=["0p145", "0p285", "0p565", "0p575"],
    )
    def test_decimal_round_half_up_not_binary_float(
        self, probability: float, expected: int
    ) -> None:
        assert risk_score_from_probability(probability) == expected


class TestInvalidProbabilities:
    @pytest.mark.parametrize(
        "probability",
        [float("nan"), float("inf"), float("-inf"), -0.001, -1.0, 1.001, 2.0],
    )
    def test_invalid_probability_rejected(self, probability: float) -> None:
        with pytest.raises(ValueError):
            risk_score_from_probability(probability)

    @pytest.mark.parametrize("value", [True, False])
    def test_boolean_rejected(self, value: bool) -> None:
        with pytest.raises(ValueError):
            risk_score_from_probability(value)
