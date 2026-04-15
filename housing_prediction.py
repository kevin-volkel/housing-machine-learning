import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


MODEL_QUALITY_THRESHOLDS = {
    "mape": 0.10,
    "r2": 0.80,
}

DEFAULT_MODEL_OUTPUT_PATH = Path("redfin_median_sale_price_model.pkl")
DEFAULT_DATA_PATH = Path("redfin_data.xlsx")
DEFAULT_REGION = "National"
FORECAST_HOLDOUT_PERIODS = 12
FORECAST_FREQUENCY = "MS"
SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 0, 12)
SARIMAX_TREND = "c"
SARIMAX_MAXITER = 75
EXOGENOUS_FEATURE_CANDIDATES = [
    "Homes Sold",
    "New Listings",
    "Inventory",
    "Days on Market",
    "Average Sale To List",
]


@dataclass(frozen=True)
class TrainingResult:
    """Container for the trained SARIMAX models and evaluation outputs."""

    models: dict[str, SARIMAXResultsWrapper]
    target_column: str
    feature_columns: list[str]
    metrics: dict[str, float]
    row_count: int
    feature_count: int
    forecast_horizon: int
    evaluation_frame: pd.DataFrame
    default_exogenous_by_region: dict[str, dict[str, float]]
    training_start_by_region: dict[str, pd.Timestamp]
    training_end_by_region: dict[str, pd.Timestamp]
    regions: list[str]


def _read_redfin_data(data_path: Path) -> pd.DataFrame:
    """Read Redfin input data from CSV or Excel based on file extension."""
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    return pd.read_csv(data_path)


def _find_column(columns: list[str], preferred: str, contains_all: list[str]) -> str | None:
    """Find a column by exact preferred name, then by token match fallback."""
    lowered = {col.lower(): col for col in columns}
    if preferred.lower() in lowered:
        return lowered[preferred.lower()]

    for col in columns:
        col_lower = col.lower()
        if all(token in col_lower for token in contains_all):
            return col
    return None


def _parse_numeric_column(series: pd.Series) -> pd.Series:
    """Convert currency and percent-like text values into numeric values."""
    text_values = series.astype(str).str.strip()
    percent_mask = text_values.str.contains("%", na=False)

    cleaned = (
        text_values
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = cast(pd.Series, pd.to_numeric(cleaned, errors="coerce"))
    return cast(pd.Series, numeric.where(~percent_mask, numeric / 100.0))


def _parse_period_column(series: pd.Series) -> pd.Series:
    """Parse the Redfin month column into a normalized monthly period timestamp."""
    parsed = pd.Series(pd.to_datetime(series, format="%B %Y", errors="coerce"), index=series.index)
    if parsed.isna().all():
        parsed = pd.Series(pd.to_datetime(series, errors="coerce"), index=series.index)
    return cast(pd.Series, parsed.dt.to_period("M").dt.to_timestamp())


def _fill_sparse_region_labels(series: pd.Series) -> pd.Series:
    """Forward-fill sparse region labels that only appear at the start of each block."""
    filled = series.replace(r"^\s*$", pd.NA, regex=True).ffill()
    return cast(pd.Series, filled.astype(str).str.strip())


def _detect_columns(df: pd.DataFrame) -> tuple[str, str | None, str | None]:
    """Infer target, month, and region columns from the Redfin dataset."""
    df.columns = [str(col).strip() for col in df.columns]

    target_col = _find_column(
        columns=df.columns.tolist(),
        preferred="Median Sale Price",
        contains_all=["median", "sale", "price"],
    )
    if target_col is None and len(df.columns) >= 3:
        target_col = str(df.columns[2])

    if target_col is None:
        raise ValueError("Unable to identify target column for Median Sale Price.")

    month_col = _find_column(
        columns=df.columns.tolist(),
        preferred="Month of Period End",
        contains_all=["month"],
    )
    region_col = _find_column(
        columns=df.columns.tolist(),
        preferred="Region",
        contains_all=["region"],
    )
    return str(target_col), month_col, region_col


def _prepare_forecasting_panel(
    df: pd.DataFrame,
    target_col: str,
    month_col: str | None,
    region_col: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a clean region-by-month forecasting panel with selected exogenous features."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if month_col is None or month_col not in df.columns:
        raise ValueError("A month column is required for SARIMAX forecasting.")
    if region_col is None or region_col not in df.columns:
        raise ValueError("A region column is required for SARIMAX forecasting.")

    panel = pd.DataFrame(index=df.index)
    panel["period"] = _parse_period_column(df[month_col])
    panel["region"] = _fill_sparse_region_labels(df[region_col])
    panel["target"] = _parse_numeric_column(df[target_col])

    exogenous_features = [col for col in EXOGENOUS_FEATURE_CANDIDATES if col in df.columns]
    for col in exogenous_features:
        panel[col] = _parse_numeric_column(df[col])

    panel = panel.dropna(subset=["period", "region", "target"]).copy()
    panel = panel.sort_values(["region", "period"]).reset_index(drop=True)

    for col in exogenous_features:
        panel[col] = panel.groupby("region", sort=False)[col].transform(lambda values: values.ffill().bfill())
        panel[col] = panel[col].fillna(panel[col].median())

    return panel, exogenous_features


def _fit_region_sarimax(
    y: pd.Series,
    exog: pd.DataFrame | None,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    maxiter: int,
) -> SARIMAXResultsWrapper:
    """Fit a single region's SARIMAX model."""
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=SARIMAX_TREND,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = cast(SARIMAXResultsWrapper, model.fit(disp=False, maxiter=maxiter))
        if not bool(result.mle_retvals.get("converged", True)):
            result = cast(SARIMAXResultsWrapper, model.fit(disp=False, maxiter=maxiter * 2))
    return result


def _build_future_exogenous_frame(
    exogenous_features: list[str],
    baseline_values: dict[str, float],
    periods: pd.DatetimeIndex,
    overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Create a future exogenous design matrix from defaults plus any user overrides."""
    data = {
        feature_name: [float(baseline_values[feature_name])] * len(periods)
        for feature_name in exogenous_features
    }
    frame = pd.DataFrame(data, index=periods)

    for feature_name, value in (overrides or {}).items():
        if feature_name in frame.columns:
            frame[feature_name] = float(value)

    return frame


def _calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute the regression metrics tracked for model quality."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _format_metric_value(metric_name: str, value: float) -> str:
    """Render a metric value in a readable format."""
    if metric_name in {"mae", "rmse"}:
        return f"${value:,.2f}"
    if metric_name == "mape":
        return f"{value:.2%}"
    return f"{value:.4f}"


def format_metrics_report(
    metrics: dict[str, float],
    metric_thresholds: dict[str, float] | None = None,
) -> str:
    """Create a readable multi-line report for model metrics."""
    threshold_rules = {
        "mape": ("<", lambda actual, target: actual < target),
        "r2": (">", lambda actual, target: actual > target),
    }

    lines = [
        "Metrics",
        "-------",
    ]
    for metric_name in ("mae", "rmse", "mape", "r2"):
        value = metrics[metric_name]
        line = f"{metric_name.upper():<5}: {_format_metric_value(metric_name, value)}"
        if metric_thresholds and metric_name in metric_thresholds:
            target = metric_thresholds[metric_name]
            symbol, comparator = threshold_rules[metric_name]
            status = "PASS" if comparator(value, target) else "FAIL"
            line += f"  |  target {symbol} {_format_metric_value(metric_name, target)}  |  {status}"
        lines.append(line)
    return "\n".join(lines)


def format_training_summary(
    result: TrainingResult,
    metric_thresholds: dict[str, float] | None = None,
) -> str:
    """Create a readable summary of the forecasting training run."""
    return "\n".join(
        [
            "Training Summary",
            "----------------",
            f"Target column : {result.target_column}",
            f"Rows used     : {result.row_count:,}",
            f"Features used : {result.feature_count}",
            f"Regions       : {len(result.regions)}",
            f"Holdout       : {result.forecast_horizon} months per region",
            "",
            format_metrics_report(result.metrics, metric_thresholds),
        ]
    )


def _coerce_optional_float(value: Any) -> float | None:
    """Convert API feature inputs into a float when possible."""
    if value in {None, ""}:
        return None
    return float(value)


def _normalize_region_name(region_name: str) -> str:
    """Normalize region names for case-insensitive matching."""
    return " ".join(region_name.strip().lower().split())


def _resolve_region_name(requested_region: Any, known_regions: list[str]) -> str:
    """Resolve a user-supplied region name to the closest supported region."""
    if not known_regions:
        raise ValueError("No trained regions are available in the model artifact.")

    default_region = DEFAULT_REGION if DEFAULT_REGION in known_regions else known_regions[0]
    if requested_region in {None, ""}:
        return default_region

    normalized_lookup = {_normalize_region_name(region): region for region in known_regions}
    aliases = {
        "united states": DEFAULT_REGION,
        "us": DEFAULT_REGION,
        "u.s.": DEFAULT_REGION,
        "u.s.a.": DEFAULT_REGION,
    }

    normalized_requested = _normalize_region_name(str(requested_region))
    if normalized_requested in normalized_lookup:
        return normalized_lookup[normalized_requested]

    aliased_region = aliases.get(normalized_requested)
    if aliased_region in known_regions:
        return aliased_region

    return default_region


def _parse_requested_period(features: dict[str, float | str | None]) -> pd.Timestamp:
    """Build a monthly timestamp from the API's year/month fields."""
    try:
        year = int(float(features["year"]))
        month = int(float(features["month"]))
    except KeyError as exc:
        raise ValueError("Prediction requests must include both 'year' and 'month'.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction year and month must be numeric.") from exc

    if not 1 <= month <= 12:
        raise ValueError("Prediction month must be between 1 and 12.")

    return pd.Timestamp(year=year, month=month, day=1)


def load_model_artifact(model_path: Path = DEFAULT_MODEL_OUTPUT_PATH) -> dict[str, Any]:
    """Load a saved model artifact from disk."""
    with open(model_path, "rb") as file:
        artifact = pickle.load(file)
    return cast(dict[str, Any], artifact)


def predict_from_artifact(
    artifact: dict[str, Any],
    features: dict[str, float | str | None],
) -> dict[str, Any]:
    """Generate a single prediction from a saved SARIMAX artifact."""
    requested_period = _parse_requested_period(features)
    known_regions = cast(list[str], artifact["regions"])
    region_name = _resolve_region_name(features.get("region"), known_regions)
    exogenous_features = cast(list[str], artifact["exogenous_feature_columns"])
    region_model = cast(SARIMAXResultsWrapper, artifact["models"][region_name])

    training_start = pd.Timestamp(artifact["training_start_by_region"][region_name])
    training_end = pd.Timestamp(artifact["training_end_by_region"][region_name])
    if requested_period < training_start:
        raise ValueError(
            f"Predictions are only supported from {training_start.strftime('%Y-%m')} onward for {region_name}."
        )

    if requested_period <= training_end:
        predicted_mean = region_model.get_prediction(start=requested_period, end=requested_period).predicted_mean
        predicted_price = float(predicted_mean.iloc[0])
        prediction_mode = "in_sample"
    else:
        future_periods = pd.date_range(
            start=training_end + pd.offsets.MonthBegin(1),
            end=requested_period,
            freq=FORECAST_FREQUENCY,
        )
        overrides = {
            feature_name: value
            for feature_name in exogenous_features
            if (value := _coerce_optional_float(features.get(feature_name))) is not None
        }
        future_exog = _build_future_exogenous_frame(
            exogenous_features=exogenous_features,
            baseline_values=cast(dict[str, float], artifact["default_exogenous_by_region"][region_name]),
            periods=future_periods,
            overrides=overrides,
        )
        forecast = region_model.get_forecast(steps=len(future_periods), exog=future_exog).predicted_mean
        predicted_price = float(forecast.iloc[-1])
        prediction_mode = "forecast"

    return {
        "predicted_median_sale_price": round(predicted_price, 2),
        "formatted": f"${predicted_price:,.0f}",
        "region_used": region_name,
        "prediction_mode": prediction_mode,
    }


def train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    *,
    test_periods: int = FORECAST_HOLDOUT_PERIODS,
    order: tuple[int, int, int] = SARIMAX_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMAX_SEASONAL_ORDER,
    maxiter: int = SARIMAX_MAXITER,
) -> TrainingResult:
    """Train one SARIMAX model per region and evaluate on a chronological holdout."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw_df = _read_redfin_data(data_path)
    target_col, month_col, region_col = _detect_columns(raw_df)
    panel, exogenous_features = _prepare_forecasting_panel(
        raw_df,
        target_col=target_col,
        month_col=month_col,
        region_col=region_col,
    )

    if panel.empty:
        raise ValueError("No usable forecasting rows were found after preprocessing.")

    regions = panel["region"].dropna().astype(str).unique().tolist()
    feature_columns = ["year", "month", "region", *exogenous_features]

    models: dict[str, SARIMAXResultsWrapper] = {}
    default_exogenous_by_region: dict[str, dict[str, float]] = {}
    training_start_by_region: dict[str, pd.Timestamp] = {}
    training_end_by_region: dict[str, pd.Timestamp] = {}
    evaluation_frames: list[pd.DataFrame] = []

    for region_name in regions:
        region_frame = panel.loc[panel["region"] == region_name].copy()
        region_frame = region_frame.sort_values("period").set_index("period").asfreq(FORECAST_FREQUENCY)
        y = region_frame["target"].astype(float)
        exog = region_frame[exogenous_features].astype(float)

        if len(y) <= test_periods:
            raise ValueError(
                f"Region {region_name} only has {len(y)} rows; at least {test_periods + 1} are required."
            )

        split_point = len(region_frame) - test_periods
        train_y = y.iloc[:split_point]
        test_y = y.iloc[split_point:]
        train_exog = exog.iloc[:split_point]
        holdout_defaults = train_exog.iloc[-1].to_dict()
        holdout_future_exog = _build_future_exogenous_frame(
            exogenous_features=exogenous_features,
            baseline_values=cast(dict[str, float], holdout_defaults),
            periods=test_y.index,
        )

        holdout_model = _fit_region_sarimax(
            train_y,
            train_exog,
            order=order,
            seasonal_order=seasonal_order,
            maxiter=maxiter,
        )
        holdout_forecast = holdout_model.get_forecast(
            steps=len(test_y),
            exog=holdout_future_exog,
        ).predicted_mean

        evaluation_frames.append(
            pd.DataFrame(
                {
                    "region": region_name,
                    "period": test_y.index,
                    "actual": test_y.values,
                    "prediction": holdout_forecast.values,
                }
            )
        )

        full_model = _fit_region_sarimax(
            y,
            exog,
            order=order,
            seasonal_order=seasonal_order,
            maxiter=maxiter,
        )
        models[region_name] = full_model
        default_exogenous_by_region[region_name] = {key: float(value) for key, value in exog.iloc[-1].to_dict().items()}
        training_start_by_region[region_name] = pd.Timestamp(y.index[0])
        training_end_by_region[region_name] = pd.Timestamp(y.index[-1])

    evaluation_frame = pd.concat(evaluation_frames, ignore_index=True)
    metrics = _calculate_metrics(
        evaluation_frame["actual"],
        evaluation_frame["prediction"],
    )

    return TrainingResult(
        models=models,
        target_column=target_col,
        feature_columns=feature_columns,
        metrics=metrics,
        row_count=len(panel),
        feature_count=len(feature_columns),
        forecast_horizon=test_periods,
        evaluation_frame=evaluation_frame,
        default_exogenous_by_region=default_exogenous_by_region,
        training_start_by_region=training_start_by_region,
        training_end_by_region=training_end_by_region,
        regions=regions,
    )


def save_model_artifact(
    result: TrainingResult,
    model_output_path: Path = DEFAULT_MODEL_OUTPUT_PATH,
) -> None:
    """Persist the trained forecasting models and metadata for later inference use."""
    with open(model_output_path, "wb") as file:
        pickle.dump(
            {
                "model_type": "sarimax",
                "models": result.models,
                "target_column": result.target_column,
                "feature_columns": result.feature_columns,
                "exogenous_feature_columns": result.feature_columns[3:],
                "metrics": result.metrics,
                "regions": result.regions,
                "default_region": DEFAULT_REGION if DEFAULT_REGION in result.regions else result.regions[0],
                "forecast_horizon": result.forecast_horizon,
                "default_exogenous_by_region": result.default_exogenous_by_region,
                "training_start_by_region": {
                    region_name: timestamp.isoformat()
                    for region_name, timestamp in result.training_start_by_region.items()
                },
                "training_end_by_region": {
                    region_name: timestamp.isoformat()
                    for region_name, timestamp in result.training_end_by_region.items()
                },
            },
            file,
        )


def main() -> None:
    """Train and save SARIMAX forecasting models for median sale prices."""
    result = train_model()
    save_model_artifact(result, DEFAULT_MODEL_OUTPUT_PATH)

    print(format_training_summary(result, MODEL_QUALITY_THRESHOLDS))
    print(f"\nSaved forecasting artifact to: {DEFAULT_MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
