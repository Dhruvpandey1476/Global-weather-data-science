"""
Advanced Analysis Module
Anomaly Detection, Climate Analysis, Air Quality, Feature Importance, Spatial Analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from scipy import stats
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str, dpi: int = 150) -> str:
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved figure: {path}")
    return str(path)


# ═══════════════════════ 1. ANOMALY DETECTION ═══════════════════════

def detect_anomalies(df: pd.DataFrame) -> dict:
    """Isolation Forest + Z-score anomaly detection on key weather features."""
    numeric_cols = ["temperature_celsius", "precip_mm", "wind_kph",
                    "pressure_mb", "humidity", "visibility_km"]
    cols = [c for c in numeric_cols if c in df.columns]
    subset = df[cols].dropna()

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_labels = iso.fit_predict(subset)          # -1 = anomaly
    iso_scores = iso.decision_function(subset)    # lower = more anomalous

    # Z-score method (|z| > 3)
    z_scores  = np.abs(stats.zscore(subset))
    z_anomaly = (z_scores > 3).any(axis=1)

    df_anomaly = df.loc[subset.index].copy()
    df_anomaly["iso_anomaly"]   = (iso_labels == -1)
    df_anomaly["iso_score"]     = iso_scores
    df_anomaly["zscore_anomaly"] = np.array(z_anomaly)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Anomaly Detection Analysis", fontsize=15, fontweight="bold")

    # PCA 2-D scatter colored by Isolation Forest
    pca  = PCA(n_components=2)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(subset)
    pcs  = pca.fit_transform(X_sc)
    colors = ["#e74c3c" if a else "#3498db" for a in (iso_labels == -1)]
    axes[0, 0].scatter(pcs[:, 0], pcs[:, 1], c=colors, alpha=0.4, s=8)
    axes[0, 0].set_title(f"Isolation Forest Anomalies (PCA 2-D)\n"
                          f"Anomalies: {(iso_labels==-1).sum()} / {len(iso_labels)}")
    axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(color="#e74c3c", label="Anomaly"),
                                Patch(color="#3498db", label="Normal")])

    # Anomaly scores distribution
    axes[0, 1].hist(iso_scores, bins=60, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[0, 1].axvline(iso.offset_, color="red", linestyle="--", label="Decision Boundary")
    axes[0, 1].set_title("Isolation Forest Score Distribution")
    axes[0, 1].set_xlabel("Anomaly Score (lower = more anomalous)")
    axes[0, 1].legend()

    # Anomaly count by feature (Z-score)
    z_col_counts = pd.Series((z_scores > 3).sum(axis=0), index=cols)
    axes[1, 0].bar(z_col_counts.index, z_col_counts.values, color="#e67e22", edgecolor="white")
    axes[1, 0].set_title("Z-Score Anomalies per Feature (|z|>3)")
    axes[1, 0].set_xlabel("Feature")
    axes[1, 0].set_ylabel("Anomaly Count")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # Anomaly temperature distribution
    ax = axes[1, 1]
    iso_anom  = df_anomaly[df_anomaly["iso_anomaly"]]["temperature_celsius"].dropna()
    iso_norm  = df_anomaly[~df_anomaly["iso_anomaly"]]["temperature_celsius"].dropna()
    ax.hist(iso_norm,  bins=60, alpha=0.6, label="Normal",  color="#3498db", density=True)
    ax.hist(iso_anom,  bins=30, alpha=0.6, label="Anomaly", color="#e74c3c", density=True)
    ax.set_title("Temperature: Normal vs Anomaly")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.legend()

    fig_path = save_fig("11_anomaly_detection")

    summary = {
        "iso_anomaly_count": int((iso_labels == -1).sum()),
        "iso_anomaly_pct": round((iso_labels == -1).mean() * 100, 2),
        "zscore_anomaly_count": int(z_anomaly.sum()),
        "zscore_anomaly_pct": round(z_anomaly.mean() * 100, 2),
        "figure": fig_path,
        "df_anomaly": df_anomaly,
    }
    logger.info(f"Anomaly detection done: IF={summary['iso_anomaly_count']}, Z={summary['zscore_anomaly_count']}")
    return summary


# ═══════════════════════ 2. CLIMATE ANALYSIS ════════════════════════

def climate_analysis(df: pd.DataFrame) -> dict:
    """Long-term climate patterns and regional variations."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Climate Analysis — Regional & Long-Term Patterns", fontsize=15, fontweight="bold")

    # Annual temperature trend by continent
    if "continent" in df.columns and "year" in df.columns:
        trend = df.groupby(["year", "continent"])["temperature_celsius"].mean().unstack()
        trend.plot(ax=axes[0, 0], marker="o", markersize=4, linewidth=2)
        axes[0, 0].set_title("Annual Avg Temperature by Continent")
        axes[0, 0].set_xlabel("Year"); axes[0, 0].set_ylabel("Temp (°C)")
        axes[0, 0].legend(fontsize=7)

    # Monthly climatology (all years)
    if "month" in df.columns and "continent" in df.columns:
        monthly_clim = df.groupby(["month", "continent"])["temperature_celsius"].mean().unstack()
        monthly_clim.plot(ax=axes[0, 1], linewidth=2)
        axes[0, 1].set_title("Monthly Temperature Climatology by Continent")
        axes[0, 1].set_xlabel("Month"); axes[0, 1].set_ylabel("Temp (°C)")
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
        axes[0, 1].legend(fontsize=7)

    # Climate variability (std deviation of monthly temp)
    if "location_name" in df.columns and "month" in df.columns:
        variability = df.groupby("location_name")["temperature_celsius"].std().sort_values(ascending=False).head(15)
        axes[0, 2].barh(variability.index[::-1], variability.values[::-1], color="#e74c3c")
        axes[0, 2].set_title("Top 15 Cities by Temperature Variability (σ)")
        axes[0, 2].set_xlabel("Std Dev (°C)")

    # Precipitation climatology by continent
    if "month" in df.columns and "continent" in df.columns:
        rain_clim = df.groupby(["month", "continent"])["precip_mm"].mean().unstack()
        rain_clim.plot(kind="bar", ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title("Monthly Precipitation Climatology by Continent")
        axes[1, 0].set_xlabel("Month"); axes[1, 0].set_ylabel("Avg Precip (mm)")
        axes[1, 0].tick_params(axis="x", rotation=0)
        axes[1, 0].legend(fontsize=7)

    # Climate cluster heatmap (city × monthly temp)
    if "location_name" in df.columns and "month" in df.columns:
        pivot = df.groupby(["location_name", "month"])["temperature_celsius"].mean().unstack()
        pivot = pivot.dropna()
        if len(pivot) > 2:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist
            top = pivot.head(20)
            sns.heatmap(top, cmap="RdYlBu_r", ax=axes[1, 1], cbar_kws={"label": "°C"}, linewidths=0.3)
            axes[1, 1].set_title("Monthly Temperature Heatmap (Top 20 Cities)")
            axes[1, 1].set_xlabel("Month")
            axes[1, 1].tick_params(axis="y", labelsize=7)

    # Boxplot of annual range per continent
    if "continent" in df.columns:
        annual_range = df.groupby(["continent", "location_name"])["temperature_celsius"].agg(
            lambda x: x.max() - x.min()
        ).reset_index()
        annual_range.columns = ["continent", "city", "temp_range"]
        sns.boxplot(data=annual_range, x="continent", y="temp_range",
                    palette="Set2", ax=axes[1, 2])
        axes[1, 2].set_title("Annual Temperature Range by Continent")
        axes[1, 2].set_xlabel(""); axes[1, 2].set_ylabel("Annual Temp Range (°C)")
        axes[1, 2].tick_params(axis="x", rotation=20)

    fig_path = save_fig("12_climate_analysis")
    logger.info("Climate analysis complete.")
    return {"figure": fig_path}


# ═══════════════════════ 3. AIR QUALITY ANALYSIS ════════════════════

def air_quality_analysis(df: pd.DataFrame) -> dict:
    """Air quality correlations with weather parameters."""
    aq_cols = [c for c in ["air_quality_PM2.5", "air_quality_PM10",
                            "air_quality_Carbon_Monoxide", "air_quality_Ozone",
                            "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide"] if c in df.columns]
    weather_cols = [c for c in ["temperature_celsius", "humidity", "wind_kph",
                                 "precip_mm", "pressure_mb", "visibility_km"] if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Air Quality Analysis", fontsize=15, fontweight="bold")

    # AQ pollutant distributions
    if "air_quality_PM2.5" in df.columns and "air_quality_PM10" in df.columns:
        axes[0, 0].hist(df["air_quality_PM2.5"].dropna(), bins=50, alpha=0.7,
                        label="PM2.5", color="#e74c3c", density=True)
        axes[0, 0].hist(df["air_quality_PM10"].dropna(), bins=50, alpha=0.7,
                        label="PM10", color="#3498db", density=True)
        axes[0, 0].set_title("PM2.5 vs PM10 Distribution")
        axes[0, 0].set_xlabel("μg/m³"); axes[0, 0].legend()

    # Top 10 most polluted cities (PM2.5)
    if "location_name" in df.columns and "air_quality_PM2.5" in df.columns:
        city_aq = df.groupby("location_name")["air_quality_PM2.5"].mean().sort_values(ascending=False).head(10)
        colors = ["#e74c3c" if v > 25 else "#f39c12" if v > 12 else "#2ecc71" for v in city_aq.values]
        axes[0, 1].barh(city_aq.index[::-1], city_aq.values[::-1], color=colors[::-1])
        axes[0, 1].axvline(12, color="orange", linestyle="--", linewidth=1, label="WHO 12 μg/m³")
        axes[0, 1].axvline(25, color="red",    linestyle="--", linewidth=1, label="WHO 25 μg/m³")
        axes[0, 1].set_title("Top 10 Most Polluted Cities (PM2.5)")
        axes[0, 1].set_xlabel("Avg PM2.5 (μg/m³)")
        axes[0, 1].legend(fontsize=8)

    # AQ correlation with weather
    if aq_cols and weather_cols:
        corr_data = df[aq_cols + weather_cols].corr().loc[aq_cols, weather_cols]
        sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=axes[0, 2], cbar_kws={"label": "Correlation"})
        axes[0, 2].set_title("AQ Pollutants vs Weather Correlations")
        axes[0, 2].tick_params(axis="both", labelsize=7)

    # PM2.5 vs Wind speed
    if "air_quality_PM2.5" in df.columns and "wind_kph" in df.columns:
        sc = axes[1, 0].scatter(df["wind_kph"], df["air_quality_PM2.5"],
                                 c=df["humidity"] if "humidity" in df.columns else "blue",
                                 cmap="YlOrRd", alpha=0.3, s=5)
        plt.colorbar(sc, ax=axes[1, 0], label="Humidity")
        axes[1, 0].set_title("PM2.5 vs Wind Speed")
        axes[1, 0].set_xlabel("Wind Speed (km/h)"); axes[1, 0].set_ylabel("PM2.5 (μg/m³)")

    # Seasonal AQ variation
    if "season" in df.columns and "air_quality_PM2.5" in df.columns:
        season_order = ["Spring", "Summer", "Autumn", "Winter"]
        sns.boxplot(data=df[df["season"].isin(season_order)], x="season", y="air_quality_PM2.5",
                    order=season_order, palette="YlOrRd", ax=axes[1, 1])
        axes[1, 1].set_title("PM2.5 by Season")
        axes[1, 1].set_xlabel(""); axes[1, 1].set_ylabel("PM2.5 (μg/m³)")

    # AQ EPA Index breakdown
    if "air_quality_us-epa-index" in df.columns:
        epa_labels = {1: "Good", 2: "Moderate", 3: "Unhealthy for Sensitive",
                      4: "Unhealthy", 5: "Very Unhealthy", 6: "Hazardous"}
        epa_counts = df["air_quality_us-epa-index"].value_counts().sort_index()
        epa_colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
        bars = axes[1, 2].bar(
            [epa_labels.get(i, str(i)) for i in epa_counts.index],
            epa_counts.values,
            color=epa_colors[:len(epa_counts)]
        )
        axes[1, 2].set_title("US EPA Air Quality Index Distribution")
        axes[1, 2].set_xlabel("AQI Category"); axes[1, 2].set_ylabel("Count")
        axes[1, 2].tick_params(axis="x", rotation=30)

    fig_path = save_fig("13_air_quality")
    logger.info("Air quality analysis complete.")
    return {"figure": fig_path}


# ═══════════════════════ 4. FEATURE IMPORTANCE ══════════════════════

def feature_importance_analysis(df: pd.DataFrame) -> dict:
    """RF + XGBoost feature importance with SHAP-style bar chart."""
    target = "temperature_celsius"
    candidate_features = [
        "humidity", "wind_kph", "pressure_mb", "precip_mm", "cloud",
        "visibility_km", "uv_index", "gust_kph", "feels_like_celsius",
        "air_quality_PM2.5", "air_quality_PM10", "air_quality_Ozone",
        "month", "day_of_year", "heat_index", "dew_point", "aq_composite",
    ]
    features = [f for f in candidate_features if f in df.columns]
    use_df   = df[[target] + features].dropna()
    X = use_df[features].values
    y = use_df[target].values

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                                   random_state=42, verbosity=0)
    xgb_model.fit(X, y)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Feature Importance Analysis", fontsize=15, fontweight="bold")

    # RF importance
    top_rf = rf_imp.head(12)
    axes[0].barh(top_rf.index[::-1], top_rf.values[::-1], color="#3498db", edgecolor="white")
    axes[0].set_title("Random Forest — Feature Importance")
    axes[0].set_xlabel("Importance Score")

    # XGBoost importance
    top_xgb = xgb_imp.head(12)
    axes[1].barh(top_xgb.index[::-1], top_xgb.values[::-1], color="#e74c3c", edgecolor="white")
    axes[1].set_title("XGBoost — Feature Importance")
    axes[1].set_xlabel("Importance Score")

    # Combined comparison
    combined = pd.DataFrame({"Random Forest": rf_imp, "XGBoost": xgb_imp}).dropna().head(10)
    combined.sort_values("Random Forest", inplace=True)
    combined.plot(kind="barh", ax=axes[2], colormap="Paired", edgecolor="white")
    axes[2].set_title("Feature Importance Comparison (Top 10)")
    axes[2].set_xlabel("Importance Score")
    axes[2].legend()

    fig_path = save_fig("14_feature_importance")
    logger.info("Feature importance analysis complete.")
    return {"rf_importance": rf_imp, "xgb_importance": xgb_imp, "figure": fig_path}


# ═══════════════════════ 5. SPATIAL ANALYSIS ════════════════════════

def spatial_analysis(df: pd.DataFrame) -> dict:
    """Geographical patterns using scatter maps (matplotlib)."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Spatial / Geographical Analysis", fontsize=15, fontweight="bold")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("No lat/lon columns found. Skipping spatial analysis.")
        return {}

    city_stats = df.groupby("location_name").agg(
        lat=("latitude", "first"),
        lon=("longitude", "first"),
        avg_temp=("temperature_celsius", "mean"),
        avg_precip=("precip_mm", "mean"),
        avg_pm25=("air_quality_PM2.5", "mean"),
        avg_humidity=("humidity", "mean"),
        continent=("continent", "first") if "continent" in df.columns else ("location_name", "first"),
    ).reset_index()

    # Temperature map
    sc1 = axes[0, 0].scatter(city_stats["lon"], city_stats["lat"],
                              c=city_stats["avg_temp"], cmap="RdYlBu_r",
                              s=200, edgecolors="white", linewidths=0.5, zorder=2)
    plt.colorbar(sc1, ax=axes[0, 0], label="Avg Temp (°C)", shrink=0.7)
    axes[0, 0].set_title("Global Average Temperature by City")
    axes[0, 0].set_xlabel("Longitude"); axes[0, 0].set_ylabel("Latitude")
    axes[0, 0].axhline(0, color="gray", linestyle="--", alpha=0.4)
    axes[0, 0].grid(True, alpha=0.3)
    for _, row in city_stats.iterrows():
        axes[0, 0].annotate(row["location_name"], (row["lon"], row["lat"]),
                             fontsize=5, alpha=0.7)

    # Precipitation map
    sc2 = axes[0, 1].scatter(city_stats["lon"], city_stats["lat"],
                              c=city_stats["avg_precip"], cmap="Blues",
                              s=200, edgecolors="white", linewidths=0.5, zorder=2)
    plt.colorbar(sc2, ax=axes[0, 1], label="Avg Precip (mm/day)", shrink=0.7)
    axes[0, 1].set_title("Global Average Precipitation by City")
    axes[0, 1].set_xlabel("Longitude"); axes[0, 1].set_ylabel("Latitude")
    axes[0, 1].grid(True, alpha=0.3)

    # PM2.5 map
    sc3 = axes[1, 0].scatter(city_stats["lon"], city_stats["lat"],
                              c=city_stats["avg_pm25"], cmap="YlOrRd",
                              s=200, edgecolors="white", linewidths=0.5, zorder=2)
    plt.colorbar(sc3, ax=axes[1, 0], label="Avg PM2.5 (μg/m³)", shrink=0.7)
    axes[1, 0].set_title("Global Air Quality (PM2.5) by City")
    axes[1, 0].set_xlabel("Longitude"); axes[1, 0].set_ylabel("Latitude")
    axes[1, 0].grid(True, alpha=0.3)

    # Temperature vs Latitude
    axes[1, 1].scatter(city_stats["lat"], city_stats["avg_temp"],
                        c=city_stats["avg_humidity"], cmap="YlGnBu",
                        s=120, edgecolors="white", linewidths=0.5)
    z = np.polyfit(city_stats["lat"], city_stats["avg_temp"], 2)
    p = np.poly1d(z)
    lat_range = np.linspace(city_stats["lat"].min(), city_stats["lat"].max(), 100)
    axes[1, 1].plot(lat_range, p(lat_range), color="red", linewidth=2, linestyle="--", label="Trend")
    axes[1, 1].set_title("Temperature vs Latitude (colored by Humidity)")
    axes[1, 1].set_xlabel("Latitude"); axes[1, 1].set_ylabel("Avg Temp (°C)")
    axes[1, 1].legend()

    fig_path = save_fig("15_spatial_analysis")
    logger.info("Spatial analysis complete.")

    # Folium interactive map (saved separately)
    folium_path = _make_folium_map(city_stats)
    return {"figure": fig_path, "folium_map": folium_path}


from pathlib import Path
import os
import pandas as pd

def _make_folium_map(city_stats: pd.DataFrame) -> str:
    """Generate interactive folium choropleth map safely writing all bytes to disk."""
    try:
        import folium
        from folium.plugins import HeatMap

        # Drop any records missing coordinate data upfront to prevent broken JS injection
        clean_stats = city_stats.dropna(subset=["lat", "lon"])
        if clean_stats.empty:
            logger.warning("No valid lat/lon coordinates available for Folium map.")
            return ""

        # Sanitize location names: backticks break folium's JS template literals
        clean_stats = clean_stats.copy()
        clean_stats["location_name"] = clean_stats["location_name"].str.replace("`", "'")

        world_map = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

        # Temperature markers
        for _, row in clean_stats.iterrows():
            temp = row["avg_temp"]
            if pd.isna(temp):
                color = "gray"
                temp_str = "N/A"
            else:
                color = "red" if temp > 25 else ("orange" if temp > 15 else ("blue" if temp < 5 else "green"))
                temp_str = f"{temp:.1f}°C"

            precip = f"{row['avg_precip']:.2f} mm" if pd.notna(row['avg_precip']) else "N/A"
            pm25 = f"{row['avg_pm25']:.1f} μg/m³" if pd.notna(row['avg_pm25']) else "N/A"

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{row['location_name']}</b><br>"
                    f"Avg Temp: {temp_str}<br>"
                    f"Avg Precip: {precip}<br>"
                    f"PM2.5: {pm25}",
                    max_width=200,
                ),
                tooltip=str(row["location_name"]),
            ).add_to(world_map)

        # Heat map layer for PM2.5 (filter out NaNs explicitly)
        heat_data = [
            [row["lat"], row["lon"], row["avg_pm25"]]
            for _, row in clean_stats.dropna(subset=["avg_pm25"]).iterrows()
        ]
        
        if heat_data:
            HeatMap(heat_data, name="PM2.5 Heatmap", min_opacity=0.3, radius=40).add_to(world_map)
            
        folium.LayerControl().add_to(world_map)

        # Setup paths
        map_dir = Path("outputs/report")
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / "interactive_map.html"
        
        # --- THE FIX: Render map directly into string representation ---
        world_map.render()
        html_content = world_map.get_root().render()
        
        # Explicit block write ensuring flush and hard drive sync sync
        with open(map_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            f.flush()               # Flush Python code memory buffers
            os.fsync(f.fileno())    # Force OS kernel to write the actual bytes to physical disk

        logger.info(f"Interactive map successfully finalized: {map_path} ({os.path.getsize(map_path)} bytes)")
        return str(map_path)
        
    except Exception as e:
        logger.warning(f"Folium map skipped: {e}", exc_info=True)
        return ""
# ═══════════════════════ 6. GEOGRAPHICAL PATTERNS ═══════════════════

def geographical_patterns(df: pd.DataFrame) -> dict:
    """Weather differences across countries and continents."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Geographical Weather Patterns", fontsize=15, fontweight="bold")

    # Mean temperature by country (top 20)
    if "country" in df.columns:
        country_temp = df.groupby("country")["temperature_celsius"].mean().sort_values(ascending=False)
        top20 = country_temp.head(20)
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(top20)))
        axes[0, 0].barh(top20.index[::-1], top20.values[::-1], color=colors)
        axes[0, 0].set_title("Avg Temperature by Country (Top 20)")
        axes[0, 0].set_xlabel("Avg Temp (°C)")
        axes[0, 0].tick_params(axis="y", labelsize=7)

    # Radar chart of continental weather averages
    features = ["temperature_celsius", "humidity", "wind_kph", "precip_mm", "uv_index"]
    features = [f for f in features if f in df.columns]
    if "continent" in df.columns and features:
        cont_means = df.groupby("continent")[features].mean()
        # Normalize to 0-1
        norm = (cont_means - cont_means.min()) / (cont_means.max() - cont_means.min() + 1e-9)

        categories = features
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        ax_radar = fig.add_subplot(2, 3, 2, polar=True)
        for i, (cont, row) in enumerate(norm.iterrows()):
            values = row.tolist() + [row.tolist()[0]]
            ax_radar.plot(angles, values, linewidth=2, label=cont)
            ax_radar.fill(angles, values, alpha=0.05)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, size=7)
        ax_radar.set_title("Normalised Weather by Continent\n(Radar)", pad=15)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    # Humidity by continent violin
    if "continent" in df.columns and "humidity" in df.columns:
        continents = df["continent"].unique()
        data_violin = [df[df["continent"] == c]["humidity"].dropna().values for c in continents]
        vp = axes[0, 2].violinplot(data_violin, showmedians=True)
        for i, pc in enumerate(vp["bodies"]):
            pc.set_facecolor(plt.cm.tab10(i / len(continents)))
            pc.set_alpha(0.7)
        axes[0, 2].set_xticks(range(1, len(continents) + 1))
        axes[0, 2].set_xticklabels(continents, rotation=20, fontsize=8)
        axes[0, 2].set_title("Humidity Distribution by Continent")
        axes[0, 2].set_ylabel("Humidity (%)")

    # UV index by continent
    if "continent" in df.columns and "uv_index" in df.columns:
        uv_cont = df.groupby("continent")["uv_index"].mean().sort_values(ascending=False)
        axes[1, 0].bar(uv_cont.index, uv_cont.values,
                        color=plt.cm.YlOrRd(np.linspace(0.3, 1, len(uv_cont))))
        axes[1, 0].set_title("Average UV Index by Continent")
        axes[1, 0].set_ylabel("UV Index")
        axes[1, 0].tick_params(axis="x", rotation=20)

    # Country-level precipitation heatmap (pivot: continent × season)
    if "continent" in df.columns and "season" in df.columns:
        pivot = df.groupby(["continent", "season"])["precip_mm"].mean().unstack(fill_value=0)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues",
                    ax=axes[1, 1], cbar_kws={"label": "Avg Precip (mm)"})
        axes[1, 1].set_title("Avg Precipitation: Continent × Season")
        axes[1, 1].set_xlabel(""); axes[1, 1].tick_params(axis="y", rotation=0)

    # Cloud cover by continent
    if "continent" in df.columns and "cloud" in df.columns:
        cloud_cont = df.groupby(["continent", "month"])["cloud"].mean().unstack()
        cloud_cont.plot(ax=axes[1, 2], colormap="Blues", linewidth=2)
        axes[1, 2].set_title("Monthly Cloud Cover by Continent")
        axes[1, 2].set_xlabel("Month"); axes[1, 2].set_ylabel("Avg Cloud (%)")
        axes[1, 2].set_xticks(range(1, 13))
        axes[1, 2].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
        axes[1, 2].legend(fontsize=7)

    fig_path = save_fig("16_geographical_patterns")
    logger.info("Geographical patterns analysis complete.")
    return {"figure": fig_path}


# ═══════════════════════ RUNNER ═════════════════════════════════════

def run_advanced(df: pd.DataFrame) -> dict:
    """Run all advanced analyses, dropping missing data to prevent JS crashes."""
    
    df_clean = df.copy()
    
    # 1. Drop rows where critical mapping coordinates are missing or NaN
    coord_cols = ['latitude', 'longitude']  # Double-check if your dataset uses 'lat', 'lon'
    coord_cols = [c for c in coord_cols if c in df_clean.columns]
    if coord_cols:
        df_clean = df_clean.dropna(subset=coord_cols)
        # Also remove rows where latitude/longitude are exactly 0.0 if they shouldn't be
        df_clean = df_clean[(df_clean[coord_cols] != 0).all(axis=1)]

    # 2. Clean text columns just in case
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace("'", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.replace('\\', "", regex=False)
            )

    # 3. Process clean data
    results = {}
    results["anomaly"]     = detect_anomalies(df_clean)
    results["climate"]     = climate_analysis(df_clean)
    results["air_quality"] = air_quality_analysis(df_clean)
    results["feature_imp"] = feature_importance_analysis(df_clean)
    results["spatial"]     = spatial_analysis(df_clean)
    results["geo_patterns"]= geographical_patterns(df_clean)
    
    logger.info("All advanced analyses complete.")
    return results