"""Tests for pure target-semantics helpers."""

from __future__ import annotations

from datetime import date

import pytest

from roadguard import days_until_maintenance, maintenance_within_30_days


class TestDaysUntilMaintenance:
    def test_same_day_is_zero_days(self) -> None:
        assert days_until_maintenance(date(2026, 1, 1), date(2026, 1, 1)) == 0

    def test_thirty_days(self) -> None:
        assert days_until_maintenance(date(2026, 1, 1), date(2026, 1, 31)) == 30

    def test_thirty_one_days(self) -> None:
        assert days_until_maintenance(date(2026, 1, 1), date(2026, 2, 1)) == 31

    def test_across_year_boundary(self) -> None:
        assert days_until_maintenance(date(2025, 12, 1), date(2026, 1, 30)) == 60

    def test_past_event_rejected(self) -> None:
        with pytest.raises(ValueError):
            days_until_maintenance(date(2026, 1, 31), date(2026, 1, 1))


class TestMaintenanceWithin30Days:
    def test_zero_days_is_positive(self) -> None:
        assert maintenance_within_30_days(0) is True

    def test_thirty_days_is_positive(self) -> None:
        assert maintenance_within_30_days(30) is True

    def test_thirty_one_days_is_negative(self) -> None:
        assert maintenance_within_30_days(31) is False

    def test_mid_window_is_positive(self) -> None:
        assert maintenance_within_30_days(15) is True

    def test_negative_days_is_negative(self) -> None:
        assert maintenance_within_30_days(-1) is False

    def test_far_future_is_negative(self) -> None:
        assert maintenance_within_30_days(1_000) is False
