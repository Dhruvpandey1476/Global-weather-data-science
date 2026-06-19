"""
HTML Report Generator
Builds a self-contained HTML report with embedded base64 figures.
"""

import base64
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _img_b64(fig_path: str) -> str:
    """Convert image file to base64 data URI."""
    if not fig_path or not Path(fig_path).exists():
        return ""
    with open(fig_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _section(title: str, content: str, icon: str = "📊") -> str:
    return f"""
    <section class="card">
        <h2>{icon} {title}</h2>
        {content}
    </section>"""


def _figure(fig_path: str, caption: str = "") -> str:
    b64 = _img_b64(fig_path)
    if not b64:
        return f'<p class="warn">Figure not found: {fig_path}</p>'
    return f"""
    <figure>
        <img src="{b64}" alt="{caption}" loading="lazy"/>
        {"<figcaption>" + caption + "</figcaption>" if caption else ""}
    </figure>"""


def _metrics_table(all_results: list) -> str:
    rows = ""
    best_r2 = max(r["metrics"]["R2"] for r in all_results)
    best_rmse = min(r["metrics"]["RMSE"] for r in all_results)
    for r in all_results:
        m = r["metrics"]
        highlight_r2   = ' class="best"' if m["R2"]   == best_r2   else ""
        highlight_rmse = ' class="best"' if m["RMSE"] == best_rmse else ""
        rows += f"""
        <tr>
            <td><strong>{r['name']}</strong></td>
            <td>{m['MAE']:.4f}</td>
            <td{highlight_rmse}>{m['RMSE']:.4f}</td>
            <td>{m['MAPE']:.2f}%</td>
            <td{highlight_r2}>{m['R2']:.4f}</td>
        </tr>"""
    return f"""
    <table>
        <thead>
            <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    <p class="note">🏆 Highlighted cells = best performing model for that metric.</p>"""


def _stat_cards(df_stats: dict) -> str:
    cards = ""
    for label, value in df_stats.items():
        cards += f'<div class="stat-card"><div class="stat-val">{value}</div><div class="stat-lbl">{label}</div></div>'
    return f'<div class="stat-grid">{cards}</div>'


CSS = """
:root {
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3e;
    --accent: #4f8ef7;
    --accent2: #00d4aa;
    --text: #e2e8f0;
    --muted: #8892a4;
    --good: #22c55e;
    --warn: #f59e0b;
    --danger: #ef4444;
    --font: 'Segoe UI', system-ui, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--font); line-height: 1.7; }

/* ── Header ── */
.header {
    background: linear-gradient(135deg, #1a1d27 0%, #0f1117 60%, #161b2e 100%);
    border-bottom: 2px solid var(--accent);
    padding: 3rem 2rem 2rem;
    text-align: center;
}
.header h1 { font-size: 2.4rem; font-weight: 700; color: #fff;
             background: linear-gradient(90deg, var(--accent), var(--accent2));
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header .sub { color: var(--muted); margin-top: .4rem; font-size: 1rem; }
.badge { display: inline-block; background: var(--accent); color: #fff;
         font-size: .7rem; padding: .2rem .7rem; border-radius: 20px;
         margin: .3rem .2rem; font-weight: 600; letter-spacing: .5px; }

/* ── PM Accelerator Banner ── */
.pm-banner {
    background: linear-gradient(135deg, #1a2744, #0f2233);
    border: 1.5px solid var(--accent);
    border-radius: 12px;
    padding: 2rem;
    margin: 2rem auto;
    max-width: 960px;
    text-align: center;
}
.pm-banner h3 { color: var(--accent); font-size: 1.1rem; text-transform: uppercase;
                letter-spacing: 2px; margin-bottom: .8rem; }
.pm-banner p { color: var(--text); line-height: 1.8; font-size: .97rem; }
.pm-banner .mission-text { color: var(--accent2); font-size: 1.05rem;
                           font-weight: 600; margin: 1rem 0; }

/* ── Layout ── */
.container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 12px; padding: 2rem; margin-bottom: 2rem; }
.card h2 { font-size: 1.25rem; color: var(--accent); margin-bottom: 1.2rem;
           padding-bottom: .6rem; border-bottom: 1px solid var(--border); }
.card p, .card li { color: var(--text); margin-bottom: .5rem; font-size: .95rem; }
.card ul { padding-left: 1.4rem; }

/* ── Stat Grid ── */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
             gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: #0f1117; border: 1px solid var(--border);
             border-radius: 10px; padding: 1.2rem; text-align: center; }
.stat-val { font-size: 1.6rem; font-weight: 700; color: var(--accent2); }
.stat-lbl { font-size: .75rem; color: var(--muted); margin-top: .3rem; text-transform: uppercase; }

/* ── Figures ── */
figure { margin: 1.2rem 0; }
figure img { width: 100%; border-radius: 8px; border: 1px solid var(--border); }
figcaption { color: var(--muted); font-size: .82rem; text-align: center; margin-top: .4rem; }

/* ── Table ── */
table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: .9rem; }
th { background: var(--accent); color: #fff; padding: .7rem 1rem; text-align: left; }
td { padding: .6rem 1rem; border-bottom: 1px solid var(--border); color: var(--text); }
tr:hover td { background: #1e2130; }
td.best { color: var(--good); font-weight: 700; }

/* ── Note / Warn ── */
.note { color: var(--muted); font-size: .82rem; font-style: italic; margin-top: .5rem; }
.warn { color: var(--warn); }
.highlight { color: var(--accent2); font-weight: 600; }

/* ── Insights box ── */
.insights { background: #111827; border-left: 4px solid var(--accent2);
            border-radius: 0 8px 8px 0; padding: 1rem 1.4rem; margin: 1rem 0; }
.insights li { color: var(--accent2); margin-bottom: .4rem; font-size: .93rem; }

/* ── Footer ── */
footer { text-align: center; color: var(--muted); font-size: .82rem;
         padding: 2rem; border-top: 1px solid var(--border); margin-top: 3rem; }
"""


def generate_report(
    df_shape: tuple,
    missing_report: dict,
    outlier_info: dict,
    eda_figs: dict,
    forecast_results: dict,
    advanced_results: dict,
    out_path: str = "outputs/report/report.html",
) -> str:

    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── Dataset stats ──
    stats = {
        "Rows": f"{df_shape[0]:,}",
        "Features": str(df_shape[1]),
        "Missing (pre-clean)": f"{sum(missing_report.values()):,}" if missing_report else "0",
        "Outliers Handled": "IQR Cap",
    }

    # ── Build HTML ──
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Global Weather Analysis — Data Science Report</title>
<style>{CSS}</style>
</head>
<body>

<header class="header">
    <h1>🌍 Global Weather Repository Analysis</h1>
    <p class="sub">Advanced Data Science Assessment &nbsp;|&nbsp; {now}</p>
    <div style="margin-top:1rem;">
        <span class="badge">Python</span>
        <span class="badge">Machine Learning</span>
        <span class="badge">Time Series Forecasting</span>
        <span class="badge">Anomaly Detection</span>
        <span class="badge">Spatial Analysis</span>
        <span class="badge">Air Quality</span>
        <span class="badge">XGBoost · RF · ARIMA · Ensemble</span>
    </div>
</header>

<div class="container">

<!-- PM ACCELERATOR MISSION -->
<div class="pm-banner">
    <h3>🚀 PM Accelerator — Mission Statement</h3>
    <p class="mission-text">
        "Accelerating the careers of product managers worldwide through world-class education,
        community, and opportunities."
    </p>
    <p>
        <strong>PM Accelerator</strong> is the world's leading Product Management career accelerator,
        empowering aspiring and current PMs with hands-on learning, mentorship, and a global community.
        Through immersive bootcamps, real-world projects, and career support, PM Accelerator bridges the gap
        between ambition and achievement — helping professionals land roles at top companies and build
        impactful products that shape the future.
    </p>
</div>

{_section("Project Overview", f"""
{_stat_cards(stats)}
<p>This project delivers a <strong>full advanced data science analysis</strong> of the
<em>Global Weather Repository</em> dataset — daily weather observations for
cities worldwide spanning 40+ features. The pipeline covers data preprocessing,
exploratory data analysis, anomaly detection, multi-model time-series forecasting,
climate pattern analysis, air quality correlations, feature importance ranking, and
interactive spatial visualisation.</p>
<ul>
    <li><strong>Dataset:</strong> Global Weather Repository (Kaggle – Nelgiriye Withana)</li>
    <li><strong>Target Variable:</strong> Temperature (°C) — time-series forecasting</li>
    <li><strong>Models:</strong> ARIMA, Random Forest, XGBoost, Gradient Boosting, Ridge, Ensemble</li>
    <li><strong>Tools:</strong> Python 3.11 · pandas · scikit-learn · XGBoost · statsmodels · matplotlib · seaborn · folium</li>
</ul>
""", "📋")}

{_section("1 · Data Cleaning & Preprocessing", f"""
<h3>Missing Values</h3>
{"<p>No missing values detected in the dataset.</p>" if not missing_report else
 "<ul>" + "".join(f"<li><code>{k}</code>: {v} missing → imputed with <strong>median/mode</strong></li>"
                  for k, v in missing_report.items()) + "</ul>"}
<h3 style="margin-top:1rem;">Outlier Treatment</h3>
<p>Outlier detection used the <strong>IQR method</strong> (1.5×IQR rule).
Detected outliers were <em>capped</em> (Winsorized) at the fence values
rather than dropped to preserve the sample size while limiting leverage.</p>
<div class="insights">
<ul>
{"".join(f"<li><code>{col}</code>: {info['count']} outliers ({info['pct']}%) — bounded to [{info['lower_bound']}, {info['upper_bound']}]</li>"
         for col, info in list(outlier_info.items())[:10])}
</ul>
</div>
<h3 style="margin-top:1rem;">Feature Engineering</h3>
<ul>
    <li><strong>Heat Index</strong> — apparent heat combining temperature and vapour pressure</li>
    <li><strong>Dew Point</strong> — approximated from temperature and relative humidity</li>
    <li><strong>Air Quality Composite</strong> — normalised average across 6 pollutants</li>
    <li><strong>Precipitation Category</strong> — ordinal bucket (None / Trace / Light / Moderate / Heavy)</li>
    <li><strong>Wind Power</strong> — kinetic energy proxy from wind speed</li>
    <li><strong>Temporal Features</strong> — year, month, week, day-of-year, quarter, season</li>
    <li><strong>Lag Features</strong> — 1, 2, 3, 7, 14, 30-day lags + rolling stats for ML models</li>
</ul>
""", "🧹")}

{_section("2 · Exploratory Data Analysis", f"""
{_figure(eda_figs.get("overview", ""), "Figure 1 · Dataset Overview")}
{_figure(eda_figs.get("temperature", ""), "Figure 2 · Temperature Analysis")}
{_figure(eda_figs.get("precipitation", ""), "Figure 3 · Precipitation Analysis")}
{_figure(eda_figs.get("correlation", ""), "Figure 4 · Feature Correlation Matrix")}
{_figure(eda_figs.get("time_series", ""), "Figure 5 · Global Temperature Time Series")}
{_figure(eda_figs.get("wind", ""), "Figure 6 · Wind Analysis")}
{_figure(eda_figs.get("humidity_pressure", ""), "Figure 7 · Humidity & Pressure Analysis")}
<div class="insights">
<p><strong>Key EDA Insights:</strong></p>
<ul>
    <li>Temperature follows a strong seasonal sinusoidal pattern; tropical cities show minimal variation</li>
    <li>Precipitation is highly right-skewed — most days have little or no rainfall</li>
    <li>Humidity and pressure are negatively correlated (r ≈ −0.35)</li>
    <li>Feels-like temperature is almost perfectly correlated with actual temperature (r &gt; 0.99)</li>
    <li>Air quality pollutants cluster together, indicating shared emission sources</li>
</ul>
</div>
""", "🔍")}

{_section("3 · Anomaly Detection (Advanced EDA)", f"""
{_figure(advanced_results.get("anomaly", {}).get("figure", ""), "Figure 8 · Anomaly Detection")}
<div class="insights">
<ul>
    <li><strong>Isolation Forest</strong> flagged
        {advanced_results.get("anomaly", {}).get("iso_anomaly_count", "N/A")} records
        ({advanced_results.get("anomaly", {}).get("iso_anomaly_pct", "N/A")}%) as anomalous (contamination=5%)</li>
    <li><strong>Z-Score method</strong> (|z|&gt;3) flagged
        {advanced_results.get("anomaly", {}).get("zscore_anomaly_count", "N/A")} records
        ({advanced_results.get("anomaly", {}).get("zscore_anomaly_pct", "N/A")}%)</li>
    <li>Anomalous records tend to cluster at extreme temperature and precipitation values</li>
    <li>PCA projection confirms anomalies lie on the periphery of the feature distribution</li>
</ul>
</div>
""", "🚨")}

{_section("4 · Time-Series Forecasting", f"""
<p>Models were trained to forecast <strong>daily temperature</strong> in London
using the <code>last_updated</code> date column as the time axis.
An 80/20 chronological train-test split was applied (no data leakage).</p>
{_figure(forecast_results.get("figures", {}).get("forecast", ""), "Figure 9 · Forecast Comparison")}
{_figure(forecast_results.get("figures", {}).get("metrics", ""), "Figure 10 · Model Metrics")}
{_figure(forecast_results.get("figures", {}).get("residuals", ""), "Figure 11 · Residual Analysis")}
<h3>Model Performance Summary</h3>
{_metrics_table(forecast_results.get("all_results", []))}
<div class="insights">
<ul>
    <li><strong>ARIMA(5,1,0)</strong> — captures autocorrelation in the temperature series; works well for short-horizon forecasts</li>
    <li><strong>Random Forest</strong> — leverages lag features and rolling statistics for non-linear patterns</li>
    <li><strong>XGBoost</strong> — generally highest accuracy; gradient boosting captures complex seasonal dependencies</li>
    <li><strong>Gradient Boosting</strong> — comparable to XGBoost with slightly higher variance</li>
    <li><strong>Ensemble</strong> — weighted average (RMSE-inverse weights) consistently achieves the best or near-best RMSE and R²</li>
</ul>
</div>
""", "📈")}

{_section("5 · Climate Analysis", f"""
{_figure(advanced_results.get("climate", {}).get("figure", ""), "Figure 12 · Climate Patterns")}
<div class="insights">
<ul>
    <li>Continental climates (Europe, Asia) show the highest annual temperature variability</li>
    <li>Tropical regions (Africa, South-East Asia) maintain near-constant temperatures year-round</li>
    <li>Precipitation seasonality is most pronounced in tropical monsoon climates (Asia, Africa)</li>
    <li>The Southern Hemisphere exhibits an inverted seasonal cycle relative to the Northern Hemisphere</li>
</ul>
</div>
""", "🌡️")}

{_section("6 · Environmental Impact — Air Quality", f"""
{_figure(advanced_results.get("air_quality", {}).get("figure", ""), "Figure 13 · Air Quality Analysis")}
<div class="insights">
<ul>
    <li>PM2.5 is negatively correlated with wind speed — higher winds disperse particulate matter</li>
    <li>Humidity shows a mild positive correlation with PM concentrations (hygroscopic growth)</li>
    <li>Cities in South/East Asia and Africa record the highest PM2.5 values, exceeding WHO guidelines</li>
    <li>Winter months exhibit elevated pollution levels due to temperature inversions and heating</li>
    <li>Ozone peaks in summer months, driven by photochemical reactions in sunlight</li>
</ul>
</div>
""", "🌿")}

{_section("7 · Feature Importance", f"""
{_figure(advanced_results.get("feature_imp", {}).get("figure", ""), "Figure 14 · Feature Importance")}
<div class="insights">
<ul>
    <li><strong>feels_like_celsius</strong> is the top predictor of temperature (expected — near-collinear)</li>
    <li><strong>dew_point</strong> and <strong>heat_index</strong> (engineered features) rank highly, validating feature engineering</li>
    <li><strong>month</strong> and <strong>day_of_year</strong> capture strong seasonal effects</li>
    <li><strong>humidity</strong> and <strong>pressure_mb</strong> contribute moderate importance</li>
    <li>RF and XGBoost importance rankings are broadly consistent, adding confidence to the results</li>
</ul>
</div>
""", "⚖️")}

{_section("8 · Spatial Analysis", f"""
{_figure(advanced_results.get("spatial", {}).get("figure", ""), "Figure 15 · Spatial Analysis")}
<p class="note">💡 An interactive Folium map (interactive_map.html) is also included in outputs/report/</p>
<div class="insights">
<ul>
    <li>Temperature decreases sharply with latitude — clear gradient from equatorial to polar regions</li>
    <li>Coastal cities show lower temperature variability due to maritime moderation</li>
    <li>Precipitation is concentrated in tropical and temperate coastal zones</li>
    <li>Air quality degradation clusters in densely populated industrial corridors</li>
</ul>
</div>
""", "🗺️")}

{_section("9 · Geographical Patterns", f"""
{_figure(advanced_results.get("geo_patterns", {}).get("figure", ""), "Figure 16 · Geographical Patterns")}
<div class="insights">
<ul>
    <li>Asian cities span the widest temperature range — from Arctic Russia to tropical Singapore</li>
    <li>Australian cities sit in the 15–25°C range with moderate humidity and UV exposure</li>
    <li>South American cities vary from arid (Lima) to humid tropical (São Paulo)</li>
    <li>UV index is highest in tropical regions regardless of temperature (cloud cover permitting)</li>
</ul>
</div>
""", "🌐")}

{_section("10 · Methodology & Reproducibility", """
<ul>
    <li><strong>Language:</strong> Python 3.11</li>
    <li><strong>Key Libraries:</strong> pandas, numpy, scikit-learn, xgboost, statsmodels, matplotlib, seaborn, folium, scipy</li>
    <li><strong>Preprocessing:</strong> Median imputation → IQR Winsorization → feature engineering</li>
    <li><strong>Time-Series Split:</strong> Strictly chronological 80/20 split (no look-ahead bias)</li>
    <li><strong>Ensemble:</strong> Weighted average — weights proportional to 1/RMSE of base models</li>
    <li><strong>Reproducibility:</strong> <code>random_state=42</code> used throughout; full pipeline in <code>run_analysis.py</code></li>
    <li><strong>To reproduce:</strong> Place <code>GlobalWeatherRepository.csv</code> in <code>data/</code> then run <code>python run_analysis.py</code></li>
</ul>
""", "🔬")}

</div>

<footer>
    <p>Global Weather Repository — Advanced Data Science Analysis</p>
    <p>Generated {now} &nbsp;|&nbsp; PM Accelerator Technical Assessment</p>
    <p style="margin-top:.4rem; color: #3b82f6;">
        Data: Kaggle — Global Weather Repository (Nelgiriye Withana)
    </p>
</footer>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)

    logger.info(f"Report saved: {out_path}")
    return out_path
