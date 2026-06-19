"""
Exploratory Data Analysis Module
Generates comprehensive visualizations saved to outputs/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "viridis"
STYLE = "seaborn-v0_8-whitegrid"
plt.style.use(STYLE)

NUMERIC_COLS = [
    "temperature_celsius", "wind_kph", "pressure_mb", "precip_mm",
    "humidity", "cloud", "visibility_km", "uv_index",
    "air_quality_PM2.5", "air_quality_PM10", "air_quality_Carbon_Monoxide",
    "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
]


def save_fig(name: str, dpi: int = 150):
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved figure: {path}")
    return str(path)


def plot_dataset_overview(df: pd.DataFrame) -> str:
    """Dataset shape, dtypes distribution, missing values overview."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

    # Row counts per continent
    continent_counts = df["continent"].value_counts() if "continent" in df.columns else df["country"].value_counts().head(10)
    axes[0].bar(continent_counts.index, continent_counts.values, color=sns.color_palette("tab10", len(continent_counts)))
    axes[0].set_title("Records by Continent")
    axes[0].set_xlabel("Continent")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    # Condition distribution
    if "condition_text" in df.columns:
        cond = df["condition_text"].value_counts().head(12)
        axes[1].barh(cond.index[::-1], cond.values[::-1], color=sns.color_palette("coolwarm", len(cond)))
        axes[1].set_title("Top Weather Conditions")
        axes[1].set_xlabel("Count")

    # Missing values heatmap-style bar
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]
    if len(missing) > 0:
        axes[2].barh(missing.index, missing.values, color="tomato")
        axes[2].set_title("Missing Values (%)")
        axes[2].set_xlabel("Missing %")
    else:
        axes[2].text(0.5, 0.5, "No Missing Values ✓", ha="center", va="center",
                     fontsize=14, color="green", transform=axes[2].transAxes)
        axes[2].set_title("Missing Values")

    return save_fig("01_dataset_overview")


def plot_temperature_analysis(df: pd.DataFrame) -> str:
    """Temperature distribution, box plot by continent/season."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle("Temperature Analysis", fontsize=16, fontweight="bold")

    col = "temperature_celsius"

    # Global distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df[col].dropna(), bins=60, color="#3498db", edgecolor="white", alpha=0.8)
    ax1.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.1f}°C")
    ax1.axvline(df[col].median(), color="orange", linestyle="--", label=f"Median: {df[col].median():.1f}°C")
    ax1.set_title("Global Temperature Distribution")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # KDE by continent
    ax2 = fig.add_subplot(gs[0, 1])
    if "continent" in df.columns:
        for continent, grp in df.groupby("continent"):
            grp[col].dropna().plot.kde(ax=ax2, label=continent)
    ax2.set_title("Temperature KDE by Continent")
    ax2.set_xlabel("Temperature (°C)")
    ax2.legend(fontsize=7)

    # Box plot by month
    ax3 = fig.add_subplot(gs[0, 2])
    if "month" in df.columns:
        month_data = [df[df["month"] == m][col].dropna() for m in range(1, 13)]
        ax3.boxplot(month_data, labels=[str(m) for m in range(1, 13)], patch_artist=True,
                    boxprops=dict(facecolor="#AED6F1"))
        ax3.set_title("Monthly Temperature Distribution")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Temperature (°C)")

    # Box plot by continent
    ax4 = fig.add_subplot(gs[1, 0])
    if "continent" in df.columns:
        df.boxplot(column=col, by="continent", ax=ax4, patch_artist=True)
        ax4.set_title("Temperature by Continent")
        ax4.set_xlabel("")
        ax4.tick_params(axis="x", rotation=20)
    plt.sca(ax4)
    plt.title("Temperature by Continent")

    # Seasonal average by continent
    ax5 = fig.add_subplot(gs[1, 1])
    if "season" in df.columns and "continent" in df.columns:
        pivot = df.groupby(["continent", "season"])[col].mean().unstack(fill_value=0)
        pivot.plot(kind="bar", ax=ax5, colormap="coolwarm")
        ax5.set_title("Avg Temperature: Continent × Season")
        ax5.set_xlabel("")
        ax5.tick_params(axis="x", rotation=30)
        ax5.legend(fontsize=7)

    # Top 10 hottest cities
    ax6 = fig.add_subplot(gs[1, 2])
    if "location_name" in df.columns:
        city_avg = df.groupby("location_name")[col].mean().sort_values(ascending=False).head(10)
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(city_avg)))
        ax6.barh(city_avg.index[::-1], city_avg.values[::-1], color=colors[::-1])
        ax6.set_title("Top 10 Hottest Cities (Avg)")
        ax6.set_xlabel("Avg Temp (°C)")

    plt.suptitle("Temperature Analysis", fontsize=16, fontweight="bold")
    return save_fig("02_temperature_analysis")


def plot_precipitation_analysis(df: pd.DataFrame) -> str:
    """Precipitation patterns and distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Precipitation Analysis", fontsize=16, fontweight="bold")

    col = "precip_mm"

    # Distribution (log scale)
    axes[0, 0].hist(df[col][df[col] > 0].dropna(), bins=60, color="#1abc9c", edgecolor="white", alpha=0.8, log=True)
    axes[0, 0].set_title("Precipitation Distribution (Log Scale)")
    axes[0, 0].set_xlabel("Precipitation (mm)")
    axes[0, 0].set_ylabel("Frequency (log)")

    # Monthly average precipitation
    if "month" in df.columns:
        monthly = df.groupby("month")[col].mean()
        axes[0, 1].bar(monthly.index, monthly.values, color=plt.cm.Blues(np.linspace(0.4, 1, 12)))
        axes[0, 1].set_title("Monthly Average Precipitation")
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Avg Precip (mm)")

    # Top 10 wettest cities
    if "location_name" in df.columns:
        city_precip = df.groupby("location_name")[col].mean().sort_values(ascending=False).head(10)
        axes[0, 2].barh(city_precip.index[::-1], city_precip.values[::-1], color="#2980b9")
        axes[0, 2].set_title("Top 10 Wettest Cities (Avg Daily)")
        axes[0, 2].set_xlabel("Avg Precip (mm/day)")

    # Precip vs Humidity scatter
    axes[1, 0].scatter(df["humidity"], df[col], alpha=0.2, s=5, c="#8e44ad")
    axes[1, 0].set_title("Precipitation vs Humidity")
    axes[1, 0].set_xlabel("Humidity (%)")
    axes[1, 0].set_ylabel("Precipitation (mm)")

    # Precip vs Temperature
    axes[1, 1].scatter(df["temperature_celsius"], df[col], alpha=0.2, s=5, c="#e74c3c")
    axes[1, 1].set_title("Precipitation vs Temperature")
    axes[1, 1].set_xlabel("Temperature (°C)")
    axes[1, 1].set_ylabel("Precipitation (mm)")

    # Precip by continent
    if "continent" in df.columns:
        cont_precip = df.groupby("continent")[col].mean().sort_values(ascending=False)
        axes[1, 2].bar(cont_precip.index, cont_precip.values,
                       color=sns.color_palette("Blues_d", len(cont_precip)))
        axes[1, 2].set_title("Avg Precipitation by Continent")
        axes[1, 2].set_xlabel("Continent")
        axes[1, 2].tick_params(axis="x", rotation=30)

    return save_fig("03_precipitation_analysis")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Correlation heatmap for numerical features."""
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    return save_fig("04_correlation_heatmap")


def plot_time_series(df: pd.DataFrame) -> str:
    """Global average temperature time series with rolling mean."""
    if "last_updated" not in df.columns:
        return ""

    ts = df.groupby("last_updated")["temperature_celsius"].mean().reset_index()
    ts = ts.sort_values("last_updated")
    ts["rolling_7"] = ts["temperature_celsius"].rolling(7, center=True).mean()
    ts["rolling_30"] = ts["temperature_celsius"].rolling(30, center=True).mean()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Time Series Analysis", fontsize=16, fontweight="bold")

    # Global average temperature
    axes[0].plot(ts["last_updated"], ts["temperature_celsius"], alpha=0.3, color="#3498db", linewidth=0.8, label="Daily")
    axes[0].plot(ts["last_updated"], ts["rolling_7"], color="#e74c3c", linewidth=1.5, label="7-day Rolling")
    axes[0].plot(ts["last_updated"], ts["rolling_30"], color="#2ecc71", linewidth=2, label="30-day Rolling")
    axes[0].set_title("Global Daily Average Temperature")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-continent time series
    if "continent" in df.columns and "month" in df.columns:
        cont_monthly = df.groupby(["continent", "month"])["temperature_celsius"].mean().unstack(level=0)
        cont_monthly.plot(ax=axes[1], colormap="tab10", linewidth=2)
        axes[1].set_title("Monthly Average Temperature by Continent")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Temperature (°C)")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    return save_fig("05_time_series")


def plot_wind_analysis(df: pd.DataFrame) -> str:
    """Wind speed and direction analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Wind Analysis", fontsize=16, fontweight="bold")

    # Wind speed distribution
    if "wind_kph" in df.columns:
        axes[0].hist(df["wind_kph"].dropna(), bins=50, color="#f39c12", edgecolor="white", alpha=0.8)
        axes[0].set_title("Wind Speed Distribution")
        axes[0].set_xlabel("Wind Speed (km/h)")
        axes[0].set_ylabel("Frequency")

    # Wind direction rose (simplified)
    if "wind_direction" in df.columns:
        wd_counts = df["wind_direction"].value_counts()
        axes[1].bar(wd_counts.index, wd_counts.values, color=plt.cm.hsv(np.linspace(0, 1, len(wd_counts))))
        axes[1].set_title("Wind Direction Distribution")
        axes[1].set_xlabel("Direction")
        axes[1].tick_params(axis="x", rotation=45)

    # Wind vs Temperature
    if "wind_kph" in df.columns and "temperature_celsius" in df.columns:
        scatter = axes[2].scatter(df["wind_kph"], df["temperature_celsius"],
                                   c=df["humidity"], cmap="RdYlGn", alpha=0.3, s=5)
        plt.colorbar(scatter, ax=axes[2], label="Humidity (%)")
        axes[2].set_title("Wind Speed vs Temperature (colored by Humidity)")
        axes[2].set_xlabel("Wind Speed (km/h)")
        axes[2].set_ylabel("Temperature (°C)")

    return save_fig("06_wind_analysis")


def plot_humidity_pressure(df: pd.DataFrame) -> str:
    """Humidity and pressure analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Humidity & Pressure Analysis", fontsize=16, fontweight="bold")

    if "humidity" in df.columns:
        axes[0, 0].hist(df["humidity"].dropna(), bins=40, color="#9b59b6", edgecolor="white", alpha=0.8)
        axes[0, 0].set_title("Humidity Distribution")
        axes[0, 0].set_xlabel("Humidity (%)")
        axes[0, 0].set_ylabel("Frequency")

    if "pressure_mb" in df.columns:
        axes[0, 1].hist(df["pressure_mb"].dropna(), bins=40, color="#1abc9c", edgecolor="white", alpha=0.8)
        axes[0, 1].set_title("Pressure Distribution")
        axes[0, 1].set_xlabel("Pressure (mb)")
        axes[0, 1].set_ylabel("Frequency")

    # Humidity vs Precipitation
    if "humidity" in df.columns and "precip_mm" in df.columns:
        axes[1, 0].scatter(df["humidity"], df["precip_mm"], alpha=0.2, s=5, color="#e74c3c")
        axes[1, 0].set_title("Humidity vs Precipitation")
        axes[1, 0].set_xlabel("Humidity (%)")
        axes[1, 0].set_ylabel("Precipitation (mm)")

    # Pressure vs Temperature
    if "pressure_mb" in df.columns and "temperature_celsius" in df.columns:
        axes[1, 1].scatter(df["pressure_mb"], df["temperature_celsius"], alpha=0.2, s=5, color="#3498db")
        axes[1, 1].set_title("Pressure vs Temperature")
        axes[1, 1].set_xlabel("Pressure (mb)")
        axes[1, 1].set_ylabel("Temperature (°C)")

    return save_fig("07_humidity_pressure")


def run_eda(df: pd.DataFrame) -> dict:
    """Run all EDA and return dict of figure paths."""
    figures = {}
    figures["overview"] = plot_dataset_overview(df)
    figures["temperature"] = plot_temperature_analysis(df)
    figures["precipitation"] = plot_precipitation_analysis(df)
    figures["correlation"] = plot_correlation_heatmap(df)
    figures["time_series"] = plot_time_series(df)
    figures["wind"] = plot_wind_analysis(df)
    figures["humidity_pressure"] = plot_humidity_pressure(df)
    logger.info(f"EDA complete: {len(figures)} figures generated.")
    return figures
