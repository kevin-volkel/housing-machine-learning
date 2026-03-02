import pickle
from pathlib import Path
from typing import cast

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
    """Convert currency/percent-like text values into numeric values."""
    text_values = series.astype(str).str.strip()
    has_percent = text_values.str.contains("%", na=False).mean() > 0.2

    cleaned = (
        text_values
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = cast(pd.Series, pd.to_numeric(cleaned, errors="coerce"))

    if has_percent:
        numeric = numeric / 100.0
    return numeric


def _prepare_features(df: pd.DataFrame, target_col: str, month_col: str | None, region_col: str | None) -> tuple[pd.DataFrame, pd.Series]:
    """Build model features and cleaned target vector from raw Redfin data."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Parse target as numeric price values.
    y = _parse_numeric_column(df[target_col])

    feature_df = pd.DataFrame(index=df.index)

    # Derive temporal features from month/date values when available.
    if month_col is not None and month_col in df.columns:
        month_series = pd.Series(pd.to_datetime(df[month_col], format="%B %Y", errors="coerce"), index=df.index)
        if month_series.isna().all():
            month_series = pd.Series(pd.to_datetime(df[month_col], errors="coerce"), index=df.index)
        feature_df["year"] = month_series.dt.year
        feature_df["month"] = month_series.dt.month

    # Preserve region as a categorical feature.
    if region_col is not None and region_col in df.columns:
        feature_df["region"] = df[region_col].astype(str).str.strip()

    # Attempt to parse remaining usable columns as numeric predictors.
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


def main() -> None:
    """Train and save a RandomForest pipeline for median sale price prediction."""
    data_path = Path("redfin_data.xlsx")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = _read_redfin_data(data_path)
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

    X, y = _prepare_features(df, target_col=str(target_col), month_col=month_col, region_col=region_col)

    if X.empty:
        raise ValueError("No usable features were found after preprocessing.")

    # Split features into numeric and categorical groups for preprocessing.
    numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Build preprocessing pipeline: impute missing values, then one-hot encode categoricals.
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

    # Train a tree-based regressor on preprocessed feature matrix.
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Hold out a test split for basic evaluation metrics.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Report common regression metrics.
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Target column: {target_col}")
    print(f"Rows used: {len(X):,}")
    print(f"Features used: {X.shape[1]}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²: {r2:.4f}")

    # Persist trained pipeline and metadata for later inference use.
    model_output_path = Path("redfin_median_sale_price_model.pkl")
    with open(model_output_path, "wb") as file:
        pickle.dump(
            {
                "pipeline": pipeline,
                "target_column": target_col,
                "feature_columns": X.columns.tolist(),
            },
            file,
        )

    print(f"Saved model pipeline to: {model_output_path}")


if __name__ == "__main__":
    main()
