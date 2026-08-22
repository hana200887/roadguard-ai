"""Pure target-semantics helpers for the locked V1 data contract.

Observations are start-of-day snapshots. ``next_maintenance_date`` is the
first maintenance event on or after the observation date, so
``days_until_maintenance`` is never negative.
"""

from __future__ import annotations

from datetime import date

from roadguard.contracts import V1_MAINTENANCE_WINDOW_DAYS

__all__ = ["days_until_maintenance", "maintenance_within_30_days"]


def days_until_maintenance(observation_date: date, next_maintenance_date: date) -> int:
    """Return calendar days from the observation snapshot to the next event.

    ``next_maintenance_date`` must be on or after ``observation_date``;
    a past event is a contract violation and raises ``ValueError``.
    """
    if next_maintenance_date < observation_date:
        raise ValueError("next_maintenance_date must be on or after observation_date")
    return (next_maintenance_date - observation_date).days


def maintenance_within_30_days(days: int) -> bool:
    """Return True when ``days`` falls inside the 30-day maintenance window.

    The window is inclusive on both ends: 0 and 30 are positive, 31 is not.
    """
    return 0 <= days <= V1_MAINTENANCE_WINDOW_DAYS
