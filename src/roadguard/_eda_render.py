"""Private Phase 9 Markdown data-card renderer and report validation.

Imported by ``roadguard.eda``; never part of the public API surface.
"""

from __future__ import annotations

import decimal
import math
import re
from collections.abc import Sequence
from datetime import date

from roadguard.eda import (
    _CATEGORICAL_FEATURE_COLUMNS,
    _CORRELATION_PAIRS,
    _DATETIME_FEATURE_COLUMNS,
    _NUMERIC_FEATURE_COLUMNS,
    _RENDER_PRECISION,
    _TARGET_VALUE_COLUMNS,
    CategoricalLevel,
    CategoricalSummary,
    ClassificationBalance,
    DataQualitySummary,
    DateSummary,
    EDAError,
    EDAReport,
    NumericSummary,
    SplitInventory,
    TargetCorrelation,
    _fresh_decimal_context,
)
from roadguard.features import FEATURE_COLUMNS
from roadguard.segments import PROVINCES, ROAD_TYPES

_NUMERIC_HEADERS = (
    "Column",
    "Count",
    "Missing",
    "Mean",
    "Population std",
    "Min",
    "Q1",
    "Median",
    "Q3",
    "Max",
    "IQR outliers",
    "IQR outlier rate",
    "Zero variance",
)


def _format_float(value: float) -> str:
    with decimal.localcontext(_fresh_decimal_context(_RENDER_PRECISION)):
        try:
            quantized = decimal.Decimal.from_float(value).quantize(
                decimal.Decimal("0.000001"), rounding=decimal.ROUND_HALF_EVEN
            )
        except decimal.DecimalException as exc:
            raise EDAError("cannot render float value in the data card") from exc
        if quantized.is_nan() or quantized.is_infinite():
            raise EDAError("cannot render a non-finite float value")
    text = format(quantized, "f")
    if text == "-0.000000":
        text = "0.000000"
    return text


def _format_date(value: date) -> str:
    return value.isoformat()


def _format_integer(value: int) -> str:
    return str(value)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _numeric_table_row(summary: NumericSummary) -> list[str]:
    return [
        summary.column,
        _format_integer(summary.count),
        _format_integer(summary.missing_count),
        _format_float(summary.mean),
        _format_float(summary.population_std),
        _format_float(summary.minimum),
        _format_float(summary.q1),
        _format_float(summary.median),
        _format_float(summary.q3),
        _format_float(summary.maximum),
        _format_integer(summary.iqr_outlier_count),
        _format_float(summary.iqr_outlier_rate),
        _format_bool(summary.zero_variance),
    ]


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def _validate_report(report: EDAReport) -> None:
    if type(report) is not EDAReport:
        raise TypeError("report must be an EDAReport")
    if report.contract_version != "roadguard.phase9.v1":
        raise EDAError(f"invalid contract version {report.contract_version!r}")
    if type(report.feature_columns) is not tuple or report.feature_columns != FEATURE_COLUMNS:
        raise EDAError("invalid feature column schema")
    if (
        type(report.training_fingerprint) is not str
        or re.fullmatch(r"[0-9a-f]{64}", report.training_fingerprint) is None
    ):
        raise EDAError("invalid training fingerprint digest")

    inventory = report.split_inventory
    if (
        type(inventory) is not tuple
        or len(inventory) != 3
        or any(type(item) is not SplitInventory for item in inventory)
        or tuple(item.name for item in inventory) != ("train", "validation", "test")
    ):
        raise EDAError("invalid split inventory")
    expected_date_counts = (34, 7, 7)
    for item, expected_dates in zip(inventory, expected_date_counts, strict=True):
        if item.row_count <= 0 or item.date_count != expected_dates:
            raise EDAError("invalid split row or date counts")
        if item.first_date > item.last_date:
            raise EDAError("invalid split date boundaries")
    if not (
        inventory[0].last_date < inventory[1].first_date
        and inventory[1].last_date < inventory[2].first_date
    ):
        raise EDAError("split partitions are not chronologically ordered")

    quality = report.data_quality
    if type(quality) is not DataQualitySummary:
        raise EDAError("invalid data quality summary")
    if (
        quality.row_count <= 0
        or quality.segment_count <= 0
        or quality.date_count != 34
        or quality.duplicate_key_count != 0
        or quality.missing_cell_count != 0
        or quality.non_finite_numeric_count != 0
    ):
        raise EDAError("contradictory data quality totals")
    if inventory[0].row_count != quality.row_count:
        raise EDAError("split train row count contradicts data quality totals")
    for item in inventory:
        if item.row_count != item.date_count * quality.segment_count:
            raise EDAError("split row counts contradict the segment/date grid")

    numeric = report.numeric_features
    if (
        type(numeric) is not tuple
        or any(type(item) is not NumericSummary for item in numeric)
        or tuple(item.column for item in numeric) != _NUMERIC_FEATURE_COLUMNS
    ):
        raise EDAError("invalid numeric feature ordering")
    for summary in numeric:
        _validate_numeric_summary(summary, quality.row_count)

    categorical = report.categorical_features
    if (
        type(categorical) is not tuple
        or any(type(item) is not CategoricalSummary for item in categorical)
        or tuple(item.column for item in categorical) != _CATEGORICAL_FEATURE_COLUMNS
    ):
        raise EDAError("invalid categorical feature ordering")
    registries = {"province": PROVINCES, "road_type": ROAD_TYPES}
    for categorical_summary in categorical:
        _validate_categorical_summary(
            categorical_summary, quality.row_count, registries[categorical_summary.column]
        )

    datetime_features = report.datetime_features
    if (
        type(datetime_features) is not tuple
        or any(type(item) is not DateSummary for item in datetime_features)
        or tuple(item.column for item in datetime_features) != _DATETIME_FEATURE_COLUMNS
    ):
        raise EDAError("invalid datetime feature ordering")
    for datetime_summary in datetime_features:
        _validate_date_summary(datetime_summary, quality.row_count)

    regression = report.regression_target
    if type(regression) is not NumericSummary or regression.column != _TARGET_VALUE_COLUMNS[0]:
        raise EDAError("invalid regression target summary")
    _validate_numeric_summary(regression, quality.row_count)

    classification = report.classification_target
    if (
        type(classification) is not ClassificationBalance
        or classification.column != _TARGET_VALUE_COLUMNS[1]
    ):
        raise EDAError("invalid classification target summary")
    if type(classification.positive_rate) is not float:
        raise EDAError("classification positive rate must be a finite float")
    if (
        classification.negative_count < 0
        or classification.positive_count < 0
        or classification.negative_count + classification.positive_count != quality.row_count
        or not _close(
            classification.positive_rate, classification.positive_count / quality.row_count
        )
    ):
        raise EDAError("invalid classification balance counts or rate")
    if not 0.0 <= classification.positive_rate <= 1.0:
        raise EDAError("classification positive rate out of range")

    correlations = report.target_correlations
    if (
        type(correlations) is not tuple
        or any(type(item) is not TargetCorrelation for item in correlations)
        or tuple((item.feature, item.target) for item in correlations) != _CORRELATION_PAIRS
    ):
        raise EDAError("invalid or incomplete target correlation set")
    for correlation in correlations:
        if correlation.pearson_r is not None and (
            type(correlation.pearson_r) is not float
            or not math.isfinite(correlation.pearson_r)
            or not -1.0 <= correlation.pearson_r <= 1.0
        ):
            raise EDAError("invalid pearson correlation value")


def _validate_numeric_summary(summary: NumericSummary, row_count: int) -> None:
    if type(summary) is not NumericSummary:
        raise EDAError("invalid numeric summary")
    if summary.count != row_count or summary.missing_count != 0:
        raise EDAError("numeric summary counts contradict the training row count")
    values = (
        summary.mean,
        summary.population_std,
        summary.minimum,
        summary.q1,
        summary.median,
        summary.q3,
        summary.maximum,
        summary.iqr_outlier_rate,
    )
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise EDAError("numeric summary contains non-finite or non-float values")
    if summary.population_std < 0:
        raise EDAError("numeric summary has a negative population std")
    if not summary.minimum <= summary.mean <= summary.maximum:
        raise EDAError("numeric summary mean is outside its minimum/maximum bounds")
    if not (summary.minimum <= summary.q1 <= summary.median <= summary.q3 <= summary.maximum):
        raise EDAError("numeric summary quartiles are not ordered")
    if not 0 <= summary.iqr_outlier_count <= summary.count:
        raise EDAError("numeric summary outlier count out of range")
    if not 0.0 <= summary.iqr_outlier_rate <= 1.0:
        raise EDAError("numeric summary outlier rate out of range")
    if not _close(summary.iqr_outlier_rate, summary.iqr_outlier_count / summary.count):
        raise EDAError("numeric summary outlier rate contradicts the outlier count")
    if type(summary.zero_variance) is not bool:
        raise EDAError("numeric summary zero_variance must be a bool")
    if summary.zero_variance:
        if summary.population_std != 0.0 or summary.minimum != summary.maximum:
            raise EDAError("zero-variance summary contradicts its statistics")
    elif summary.population_std == 0.0:
        raise EDAError("non-constant summary cannot have zero population std")


def _validate_categorical_summary(
    summary: CategoricalSummary, row_count: int, registry: tuple[str, ...]
) -> None:
    if type(summary) is not CategoricalSummary:
        raise EDAError("invalid categorical summary")
    if summary.count != row_count or summary.missing_count != 0:
        raise EDAError("categorical summary counts contradict the training row count")
    if type(summary.levels) is not tuple or not summary.levels:
        raise EDAError("categorical summary has no levels")
    if summary.cardinality != len(summary.levels):
        raise EDAError("categorical cardinality contradicts the level count")
    total = 0
    keys: list[tuple[int, str]] = []
    for level in summary.levels:
        if type(level) is not CategoricalLevel:
            raise EDAError("invalid categorical level")
        if type(level.value) is not str or level.value not in registry:
            raise EDAError("categorical level is outside the locked registry")
        if level.count <= 0:
            raise EDAError("categorical level has a non-positive count")
        total += level.count
        keys.append((-level.count, level.value))
        if not _close(level.proportion, level.count / row_count):
            raise EDAError("categorical proportion contradicts the level count")
    if total != row_count:
        raise EDAError("categorical level counts contradict the training row count")
    if keys != sorted(keys):
        raise EDAError("categorical levels are not ordered by descending count")


def _validate_date_summary(summary: DateSummary, row_count: int) -> None:
    if type(summary) is not DateSummary:
        raise EDAError("invalid datetime summary")
    if summary.count != row_count or summary.missing_count != 0:
        raise EDAError("datetime summary counts contradict the training row count")
    if type(summary.minimum) is not date or type(summary.maximum) is not date:
        raise EDAError("datetime summary boundaries must be dates")
    if not 1 <= summary.unique_count <= summary.count:
        raise EDAError("datetime summary unique count out of range")
    if summary.minimum > summary.maximum:
        raise EDAError("datetime summary boundaries are not ordered")


def render_data_card(report: EDAReport) -> str:
    """Render the deterministic in-memory Markdown data card."""
    _validate_report(report)

    fingerprint_bullet = f"- Training fingerprint: `{report.training_fingerprint}`"
    feature_bullet = "- Feature columns: " + ", ".join(
        f"`{column}`" for column in report.feature_columns
    )
    split_rows = [
        [
            item.name,
            _format_integer(item.row_count),
            _format_integer(item.date_count),
            _format_date(item.first_date),
            _format_date(item.last_date),
        ]
        for item in report.split_inventory
    ]
    quality_row = [
        _format_integer(report.data_quality.row_count),
        _format_integer(report.data_quality.segment_count),
        _format_integer(report.data_quality.date_count),
        _format_integer(report.data_quality.duplicate_key_count),
        _format_integer(report.data_quality.missing_cell_count),
        _format_integer(report.data_quality.non_finite_numeric_count),
    ]
    numeric_rows = [_numeric_table_row(summary) for summary in report.numeric_features]
    categorical_rows = [
        [
            summary.column,
            level.value,
            _format_integer(level.count),
            _format_float(level.proportion),
        ]
        for summary in report.categorical_features
        for level in summary.levels
    ]
    datetime_rows = [
        [
            summary.column,
            _format_integer(summary.count),
            _format_integer(summary.missing_count),
            _format_integer(summary.unique_count),
            _format_date(summary.minimum),
            _format_date(summary.maximum),
        ]
        for summary in report.datetime_features
    ]
    classification_row = [
        report.classification_target.column,
        _format_integer(report.classification_target.negative_count),
        _format_integer(report.classification_target.positive_count),
        _format_float(report.classification_target.positive_rate),
    ]
    correlation_rows = [
        [
            correlation.feature,
            correlation.target,
            (
                "not-defined"
                if correlation.pearson_r is None
                else _format_float(correlation.pearson_r)
            ),
        ]
        for correlation in report.target_correlations
    ]

    blocks = [
        "# RoadGuard AI - Phase 9 Train-Only Data Card",
        "## Scope and leakage guard\n\n"
        "- Statistics and correlations use only the canonical 34-date training partition.\n"
        "- Validation and test are represented only by row counts, date counts, and "
        "date boundaries.\n"
        "- No preprocessing was fit or applied, and no model was trained, selected, "
        "or evaluated.",
        "## Provenance\n\n"
        "- Contract: `roadguard.phase9.v1`\n"
        f"{fingerprint_bullet}\n"
        f"{feature_bullet}",
        "## Split inventory\n\n"
        + _table(("Partition", "Rows", "Dates", "First date", "Last date"), split_rows),
        "## Training data quality\n\n"
        + _table(
            (
                "Rows",
                "Segments",
                "Dates",
                "Duplicate keys",
                "Missing cells",
                "Non-finite numeric cells",
            ),
            [quality_row],
        ),
        "## Training feature summaries",
        "### Numeric features\n\n" + _table(_NUMERIC_HEADERS, numeric_rows),
        "### Categorical features\n\n"
        + _table(("Column", "Level", "Count", "Proportion"), categorical_rows),
        "### Datetime features\n\n"
        + _table(("Column", "Count", "Missing", "Unique", "Min", "Max"), datetime_rows),
        "## Training target summaries",
        "### Regression target\n\n"
        + _table(_NUMERIC_HEADERS, [_numeric_table_row(report.regression_target)]),
        "### Classification target\n\n"
        + _table(("Column", "Negative", "Positive", "Positive rate"), [classification_row]),
        "## Train-only target correlations\n\n"
        + _table(("Feature", "Target", "Pearson r"), correlation_rows),
        "## Limitations\n\n"
        "- This card is descriptive train-only evidence; it is not causal analysis "
        "or model-performance evidence.\n"
        "- Validation and test feature/target distributions were not summarized.\n"
        "- The SHA-256 fingerprint is an equality/integrity identifier, not "
        "anonymization, authentication, or a digital signature.",
    ]
    return "\n\n".join(blocks) + "\n"
