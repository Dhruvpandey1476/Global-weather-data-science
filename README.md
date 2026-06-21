# 🌍 Global Weather Repository — Advanced Data Science Analysis

> **PM Accelerator Technical Assessment** — Data Scientist Internship

---

## 📌 PM Accelerator Mission

> *"Accelerating the careers of product managers worldwide through world-class education, community, and opportunities."*

PM Accelerator is the world's leading Product Management career accelerator, empowering aspiring and current PMs with hands-on learning, mentorship, and a global community. Through immersive bootcamps, real-world projects, and career support, PM Accelerator bridges the gap between ambition and achievement.

---

## 🎯 Project Overview

This project performs a **full advanced data science analysis** of the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) dataset — daily weather observations for cities worldwide with 40+ features.

### What's Covered

| Area | Techniques |
|------|-----------|
| **Data Cleaning** | Missing value imputation, IQR Winsorization, outlier detection |
| **Feature Engineering** | Heat index, dew point, AQ composite, lag features, temporal features |
| **EDA** | Temperature, precipitation, wind, humidity, pressure, correlations, time-series |
| **Anomaly Detection** | Isolation Forest + Z-score (|z|>3) |
| **Forecasting** | ARIMA, Random Forest, XGBoost, Gradient Boosting, Ridge, **Ensemble** |
| **Climate Analysis** | Long-term regional patterns, seasonal climatology, variability |
| **Air Quality** | PM2.5/PM10, pollutant-weather correlations, WHO benchmark comparison |
| **Feature Importance** | RF + XGBoost importance, cross-model comparison |
| **Spatial Analysis** | Coordinate-based global maps, temperature-latitude regression |
| **Geo Patterns** | Country/continent differences, radar chart, violin plots, UV analysis |

---

## 🗂️ Project Structure

```
global_weather_ds/
├── run_analysis.py          # Main pipeline entry point
├── requirements.txt         # Python dependencies
├── README.md
│
├── src/
│   ├── data_loader.py       # Load CSV or generate synthetic data
│   ├── preprocessing.py     # Cleaning, outliers, normalization, feature engineering
│   ├── eda.py               # All EDA visualizations (7 figures)
│   ├── models.py            # ARIMA, RF, XGBoost, GB, Ridge, Ensemble (3 figures)
│   ├── advanced.py          # Anomaly detection, climate, AQ, importance, spatial (6 figures)
│   └── report.py            # Self-contained HTML report generator
│
├── data/
│   └── GlobalWeatherRepository.csv   ← place dataset here
│
└── outputs/
    ├── figures/             # All 16 PNG figures
    ├── models/              # Saved model artifacts
    └── report/
        ├── report.html      # Final self-contained HTML report (embedded images)
        └── interactive_map.html   # Interactive Folium map
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/global-weather-ds.git
cd global-weather-ds
pip install -r requirements.txt
```

### 2. Get the Dataset

Download **GlobalWeatherRepository.csv** from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) and place it in the `data/` folder.

> **No dataset? No problem.** If the CSV is not found, the pipeline automatically generates a realistic synthetic dataset (30 cities × 730 days) matching the exact schema.

### 3. Run the Pipeline

```bash
# Default run (London forecasting)
python run_analysis.py

# Specify dataset path
python run_analysis.py --data path/to/GlobalWeatherRepository.csv

# Forecast a different city
python run_analysis.py --city "New York"
python run_analysis.py --city "Tokyo"
python run_analysis.py --city "Mumbai"
```

### 4. View Results

```
outputs/report/report.html      ← Open in any browser (self-contained, no server needed)
outputs/report/interactive_map.html  ← Interactive global map
outputs/figures/*.png           ← All 16 analysis figures
```

---

## 📊 Models & Evaluation

All models forecast **daily temperature (°C)** with an **80/20 chronological split**:

| Model | Description |
|-------|-------------|
| `ARIMA(5,1,0)` | Classical time-series with auto-correlation structure |
| `Random Forest` | 200 trees, lag + rolling features |
| `XGBoost` | Gradient boosting, 300 estimators, learning rate 0.05 |
| `Gradient Boosting` | sklearn GBR, 200 estimators |
| `Ridge Regression` | L2-regularised linear baseline |
| `Ensemble` | RMSE-weighted average of all ML models |

**Metrics reported:** MAE · RMSE · MAPE · R²

---

## 📈 Key Findings

- **Temperature** follows a strong sinusoidal seasonal cycle; tropical cities show <3°C annual variation
- **Precipitation** is highly right-skewed — ~70% of days have no measurable rainfall
- **XGBoost + Ensemble** consistently outperform ARIMA on RMSE and R² for daily temperature forecasting
- **PM2.5** is strongly negatively correlated with wind speed (r ≈ −0.55) — wind disperses pollutants
- **Isolation Forest** identified ~5% anomalous records; many correspond to extreme weather events
- **Latitude** explains ~60% of variance in average annual temperature (polynomial fit R² ≈ 0.82)
- **Heat index** and **dew point** (engineered features) rank among the top 5 most important predictors

---

## 🛠️ Tech Stack

```
Python 3.11
├── pandas · numpy                  Data manipulation
├── matplotlib · seaborn            Static visualizations
├── scikit-learn                    RF, GBR, Ridge, preprocessing, anomaly detection
├── xgboost                         Gradient boosting
├── statsmodels                     ARIMA time-series model
├── scipy                           Statistical tests, Z-scores, Q-Q plots
└── folium                          Interactive geographic map
```

## 📬 Submission

Submitted via the [PM Accelerator Google Form](https://forms.gle/XfM3Xrzpo9sbHr4g8).

Repository is set to **Public** for evaluation.

---

## 📄 License

MIT License — free to use and adapt with attribution.
