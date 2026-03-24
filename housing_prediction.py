import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_QUALITY_THRESHOLDS = {
    "mape": 0.10,
    "r2": 0.80,
}


@dataclass(frozen=True)
class TrainingResult:
    """Container for a trained model pipeline and evaluation outputs."""

    pipeline: Pipeline
    target_column: str
    feature_columns: list[str]
    metrics: dict[str, float]
    row_count: int
    feature_count: int
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: pd.Series


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


def _prepare_features(
    df: pd.DataFrame,
    target_col: str,
    month_col: str | None,
    region_col: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build model features and cleaned target vector from raw Redfin data."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    y = _parse_numeric_column(df[target_col])

    feature_df = pd.DataFrame(index=df.index)

    if month_col is not None and month_col in df.columns:
        month_series = pd.Series(pd.to_datetime(df[month_col], format="%B %Y", errors="coerce"), index=df.index)
        if month_series.isna().all():
            month_series = pd.Series(pd.to_datetime(df[month_col], errors="coerce"), index=df.index)
        feature_df["year"] = month_series.dt.year
        feature_df["month"] = month_series.dt.month

    if region_col is not None and region_col in df.columns:
        feature_df["region"] = df[region_col].astype(str).str.strip()

    for col in df.columns:
        col_lower = col.lower()
        if col == target_col:
            continue
        if month_col is not None and col == month_col:
            continue
        if region_col is not None and col == region_col:
            continue
        if "sale price" in col_lower:
            continue

        parsed = _parse_numeric_column(df[col])
        if parsed.notna().sum() > 0:
            feature_df[col] = parsed

    model_df = feature_df.copy()
    model_df["target"] = y
    model_df = model_df.dropna(subset=["target"])

    y_clean = model_df.pop("target")
    return model_df, y_clean


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


def _build_pipeline(X: pd.DataFrame, *, random_state: int = 42, n_estimators: int = 400) -> Pipeline:
    """Build the preprocessing and RandomForest training pipeline."""
    numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


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
    """Create a readable summary of the training run."""
    return "\n".join(
        [
            "Training Summary",
            "----------------",
            f"Target column : {result.target_column}",
            f"Rows used     : {result.row_count:,}",
            f"Features used : {result.feature_count}",
            "",
            format_metrics_report(result.metrics, metric_thresholds),
        ]
    )


def train_model(
    data_path: Path = Path("redfin_data.xlsx"),
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 400,
) -> TrainingResult:
    """Train the median sale price model and return evaluation details."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = _read_redfin_data(data_path)
    target_col, month_col, region_col = _detect_columns(df)
    X, y = _prepare_features(df, target_col=target_col, month_col=month_col, region_col=region_col)

    if X.empty:
        raise ValueError("No usable features were found after preprocessing.")

    pipeline = _build_pipeline(X, random_state=random_state, n_estimators=n_estimators)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)
    predictions = pd.Series(pipeline.predict(X_test), index=y_test.index, name="prediction")
    metrics = _calculate_metrics(y_test, predictions)

    return TrainingResult(
        pipeline=pipeline,
        target_column=target_col,
        feature_columns=X.columns.tolist(),
        metrics=metrics,
        row_count=len(X),
        feature_count=X.shape[1],
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
    )


def save_model_artifact(
    result: TrainingResult,
    model_output_path: Path = Path("redfin_median_sale_price_model.pkl"),
) -> None:
    """Persist the trained pipeline and metadata for later inference use."""
    with open(model_output_path, "wb") as file:
        pickle.dump(
            {
                "pipeline": result.pipeline,
                "target_column": result.target_column,
                "feature_columns": result.feature_columns,
                "metrics": result.metrics,
            },
            file,
        )


def main() -> None:
    """Train and save a RandomForest pipeline for median sale price prediction."""
    result = train_model()
    model_output_path = Path("redfin_median_sale_price_model.pkl")
    save_model_artifact(result, model_output_path)

    print(format_training_summary(result, MODEL_QUALITY_THRESHOLDS))
    print(f"\nSaved model pipeline to: {model_output_path}")


if __name__ == "__main__":
    main()
