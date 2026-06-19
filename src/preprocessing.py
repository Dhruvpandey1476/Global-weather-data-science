"""
Data Preprocessing Module
Handles missing values, outliers, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

NUMERIC_COLS = [
    "temperature_celsius", "temperature_fahrenheit", "wind_mph", "wind_kph",
    "wind_degree", "pressure_mb", "pressure_in", "precip_mm", "precip_in",
    "humidity", "cloud", "feels_like_celsius", "feels_like_fahrenheit",
    "visibility_km", "visibility_miles", "uv_index", "gust_mph", "gust_kph",
    "air_quality_Carbon_Monoxide", "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide", "air_quality_PM2.5", "air_quality_PM10",
    "air_quality_us-epa-index", "air_quality_gb-defra-index", "moon_illumination",
]

AIR_QUALITY_COLS = [
    "air_quality_Carbon_Monoxide", "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide", "air_quality_PM2.5", "air_quality_PM10",
]


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse last_updated to datetime and extract time features."""
    df = df.copy()
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df["year"] = df["last_updated"].dt.year
    df["month"] = df["last_updated"].dt.month
    df["day"] = df["last_updated"].dt.day
    df["day_of_year"] = df["last_updated"].dt.day_of_year
    df["week"] = df["last_updated"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["last_updated"].dt.quarter
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    })
    logger.info("Datetime parsed and time features extracted.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using appropriate strategies."""
    df = df.copy()
    report = {}

    for col in df.columns:
        missing = df[col].isna().sum()
        if missing == 0:
            continue
        report[col] = missing

        if col in NUMERIC_COLS:
            # Use median for numeric columns
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == "object":
            # Use mode for categorical
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    if report:
        logger.info(f"Missing values handled: {report}")
    else:
        logger.info("No missing values found.")
    return df, report


def detect_outliers_iqr(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Detect outliers using IQR method and return outlier summary."""
    if cols is None:
        cols = [c for c in NUMERIC_COLS if c in df.columns]

    outlier_info = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_info[col] = {
            "count": int(mask.sum()),
            "pct": round(mask.mean() * 100, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
        }

    logger.info(f"Outlier detection complete for {len(cols)} columns.")
    return outlier_info


def cap_outliers_iqr(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Cap outliers at IQR bounds (Winsorization)."""
    df = df.copy()
    if cols is None:
        cols = [c for c in NUMERIC_COLS if c in df.columns]

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    logger.info("Outliers capped using IQR Winsorization.")
    return df


def normalize_features(df: pd.DataFrame, method: str = "standard", cols: list = None):
    """Normalize numerical features. Returns (df_scaled, scaler, scaled_cols)."""
    df = df.copy()
    if cols is None:
        cols = [c for c in NUMERIC_COLS if c in df.columns]

    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    logger.info(f"Features normalized using {method} scaler on {len(cols)} columns.")
    return df_scaled, scaler, cols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for analysis and modeling."""
    df = df.copy()

    # Temperature-humidity composite (Heat Index proxy)
    if "temperature_celsius" in df.columns and "humidity" in df.columns:
        df["heat_index"] = (
            df["temperature_celsius"] + 0.33 * (df["humidity"] / 100 * 6.105 *
            np.exp(17.27 * df["temperature_celsius"] / (237.7 + df["temperature_celsius"]))) - 4
        )

    # Dew point approximation
    if "temperature_celsius" in df.columns and "humidity" in df.columns:
        df["dew_point"] = df["temperature_celsius"] - ((100 - df["humidity"]) / 5)

    # Diurnal range (feels_like vs actual)
    if "temperature_celsius" in df.columns and "feels_like_celsius" in df.columns:
        df["temp_feels_diff"] = df["temperature_celsius"] - df["feels_like_celsius"]

    # Wind power index
    if "wind_kph" in df.columns:
        df["wind_power"] = 0.5 * 1.225 * (df["wind_kph"] / 3.6) ** 3

    # Air quality composite score (normalized average of pollutants)
        aq_cols = [c for c in AIR_QUALITY_COLS if c in df.columns]
    if aq_cols:
        # Pre-calculate mins and maxs once for all columns
        mins = df[aq_cols].min()
        maxs = df[aq_cols].max()
        
        # Perform vectorized min-max normalization and calculate the row-wise mean
        normalized_df = (df[aq_cols] - mins) / (maxs - mins + 1e-9)
        df["aq_composite"] = normalized_df.mean(axis=1)

    # Precipitation category
    if "precip_mm" in df.columns:
        df["precip_category"] = pd.cut(
            df["precip_mm"],
            bins=[-0.01, 0.1, 2, 10, 50, 9999],
            labels=["None", "Trace", "Light", "Moderate", "Heavy"],
        )

    logger.info("Feature engineering complete.")
    return df


def full_preprocess(df: pd.DataFrame):
    """Run the full preprocessing pipeline. Returns (df_clean, outlier_info, missing_report)."""
    df = parse_datetime(df)
    df, missing_report = handle_missing_values(df)
    outlier_info = detect_outliers_iqr(df)
    df = cap_outliers_iqr(df)
    df = engineer_features(df)
    logger.info("Full preprocessing pipeline complete.")
    return df, outlier_info, missing_report
