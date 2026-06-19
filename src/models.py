"""
Forecasting Models Module
ARIMA, Random Forest, XGBoost, and Ensemble forecasting with evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pathlib import Path
import warnings
import logging
import joblib
import os

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── helpers ────────────────────────────

def save_fig(name: str, dpi: int = 150):
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved figure: {path}")
    return str(path)


def evaluate(y_true, y_pred, model_name: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)
    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
               "MAPE": round(mape, 4), "R2": round(r2, 4)}
    logger.info(f"{model_name} — MAE:{mae:.3f}  RMSE:{rmse:.3f}  MAPE:{mape:.2f}%  R²:{r2:.4f}")
    return metrics


# ─────────────────────────── data prep ──────────────────────────

def build_time_series(df: pd.DataFrame, city: str = "London", target: str = "temperature_celsius"):
    """Aggregate daily average temperature for one city, sorted by date."""
    sub = df[df["location_name"] == city].copy()
    sub = sub.sort_values("last_updated")
    ts = sub.groupby("last_updated")[target].mean().reset_index()
    ts.columns = ["date", "value"]
    ts = ts.dropna().set_index("date")
    return ts


def create_lag_features(series: pd.Series, lags: list = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
    """Create lag features + rolling statistics for supervised learning."""
    df = pd.DataFrame({"value": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    df["rolling_mean_7"]  = series.shift(1).rolling(7).mean()
    df["rolling_std_7"]   = series.shift(1).rolling(7).std()
    df["rolling_mean_14"] = series.shift(1).rolling(14).mean()
    df["rolling_mean_30"] = series.shift(1).rolling(30).mean()
    df["day_of_year"]     = pd.DatetimeIndex(series.index).day_of_year
    df["month"]           = pd.DatetimeIndex(series.index).month
    df["week"]            = pd.DatetimeIndex(series.index).isocalendar().week.astype(int)
    df = df.dropna()
    return df


def train_test_split_ts(df_feat: pd.DataFrame, test_ratio: float = 0.20):
    split = int(len(df_feat) * (1 - test_ratio))
    train = df_feat.iloc[:split]
    test  = df_feat.iloc[split:]
    feature_cols = [c for c in df_feat.columns if c != "value"]
    X_train, y_train = train[feature_cols].values, train["value"].values
    X_test,  y_test  = test[feature_cols].values,  test["value"].values
    return X_train, y_train, X_test, y_test, test.index


# ─────────────────────────── ARIMA ──────────────────────────────

def train_arima(series: pd.Series, test_steps: int = 60):
    """Fit ARIMA(5,1,0) on training portion and forecast test_steps ahead."""
    try:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        train = series.iloc[:-test_steps]
        test  = series.iloc[-test_steps:]

        model = _ARIMA(train, order=(5, 1, 0))
        result = model.fit()
        forecast = result.forecast(steps=test_steps)
        metrics = evaluate(test.values, forecast.values, "ARIMA")
        return {"name": "ARIMA(5,1,0)", "forecast": forecast, "actual": test,
                "metrics": metrics, "test_index": test.index}
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")
        return None


# ─────────────────────────── ML models ──────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test, test_index):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    metrics = evaluate(y_test, preds, "RandomForest")
    return {"name": "Random Forest", "model": model, "scaler": scaler,
            "preds": preds, "actual": y_test, "test_index": test_index, "metrics": metrics}


def train_xgboost(X_train, y_train, X_test, y_test, test_index):
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, eval_metric="rmse",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds, "XGBoost")
    return {"name": "XGBoost", "model": model,
            "preds": preds, "actual": y_test, "test_index": test_index, "metrics": metrics}


def train_gradient_boosting(X_train, y_train, X_test, y_test, test_index):
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds, "GradientBoosting")
    return {"name": "Gradient Boosting", "model": model,
            "preds": preds, "actual": y_test, "test_index": test_index, "metrics": metrics}


def train_ridge(X_train, y_train, X_test, y_test, test_index):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    metrics = evaluate(y_test, preds, "Ridge")
    return {"name": "Ridge Regression", "model": model, "scaler": scaler,
            "preds": preds, "actual": y_test, "test_index": test_index, "metrics": metrics}


# ─────────────────────────── Ensemble ───────────────────────────

def ensemble_predict(results: list, weights: list = None) -> dict:
    """Weighted average ensemble of ML model predictions."""
    if weights is None:
        # Weight inversely proportional to RMSE
        rmses = np.array([r["metrics"]["RMSE"] for r in results])
        inv = 1.0 / (rmses + 1e-9)
        weights = inv / inv.sum()

    preds_stack = np.column_stack([r["preds"] for r in results])
    ensemble_preds = preds_stack @ weights
    y_true = results[0]["actual"]
    metrics = evaluate(y_true, ensemble_preds, "Ensemble")
    return {"name": "Ensemble (Weighted Avg)", "preds": ensemble_preds,
            "actual": y_true, "test_index": results[0]["test_index"],
            "metrics": metrics, "weights": weights}


# ─────────────────────────── Visualization ──────────────────────

def plot_forecast_comparison(ml_results: list, arima_result: dict = None, city: str = "London") -> str:
    """Plot actual vs predicted for all models."""
    n_models = len(ml_results) + (1 if arima_result else 0)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Forecast Comparison — {city}", fontsize=15, fontweight="bold")

    idx = 0
    if arima_result:
        ax = axes[idx]
        ax.plot(arima_result["actual"].index, arima_result["actual"].values,
                color="black", label="Actual", linewidth=1.5)
        ax.plot(arima_result["forecast"].index, arima_result["forecast"].values,
                color="#e74c3c", linestyle="--", label="ARIMA Forecast", linewidth=1.5)
        m = arima_result["metrics"]
        ax.set_title(f"ARIMA(5,1,0) — MAE:{m['MAE']:.2f} | RMSE:{m['RMSE']:.2f} | R²:{m['R2']:.3f}")
        ax.legend(); ax.set_ylabel("Temp (°C)"); ax.grid(True, alpha=0.3)
        idx += 1

    colors = ["#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    for i, res in enumerate(ml_results):
        ax = axes[idx]
        ax.plot(res["test_index"], res["actual"], color="black", label="Actual", linewidth=1.5)
        ax.plot(res["test_index"], res["preds"],
                color=colors[i % len(colors)], linestyle="--",
                label=f"{res['name']} Forecast", linewidth=1.5)
        m = res["metrics"]
        ax.set_title(f"{res['name']} — MAE:{m['MAE']:.2f} | RMSE:{m['RMSE']:.2f} | R²:{m['R2']:.3f}")
        ax.legend(); ax.set_ylabel("Temp (°C)"); ax.grid(True, alpha=0.3)
        idx += 1

    return save_fig("08_forecast_comparison")


def plot_model_metrics_comparison(all_results: list) -> str:
    """Bar chart comparison of all model metrics."""
    names   = [r["name"] for r in all_results]
    mae     = [r["metrics"]["MAE"]  for r in all_results]
    rmse    = [r["metrics"]["RMSE"] for r in all_results]
    mape    = [r["metrics"]["MAPE"] for r in all_results]
    r2      = [r["metrics"]["R2"]   for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

    palette = sns_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for ax, metric, values, lower_better in [
        (axes[0, 0], "MAE",  mae,  True),
        (axes[0, 1], "RMSE", rmse, True),
        (axes[1, 0], "MAPE (%)", mape, True),
        (axes[1, 1], "R²",  r2,  False),
    ]:
        bars = ax.bar(names, values, color=palette[:len(names)], edgecolor="white")
        ax.set_title(f"{metric} {'(lower=better)' if lower_better else '(higher=better)'}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    return save_fig("09_model_metrics_comparison")


def plot_residuals(result: dict) -> str:
    """Residual analysis for best model."""
    residuals = result["actual"] - result["preds"] if isinstance(result["actual"], np.ndarray) \
        else result["actual"].values - result["preds"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Residual Analysis — {result['name']}", fontsize=14, fontweight="bold")

    # Residual plot
    axes[0].scatter(result["preds"], residuals, alpha=0.4, s=15, color="#3498db")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residuals")

    # Distribution of residuals
    axes[1].hist(residuals, bins=40, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")

    # Q-Q plot
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[2].scatter(osm, osr, s=10, alpha=0.5, color="#e74c3c")
    axes[2].plot(osm, slope * np.array(osm) + intercept, color="black", linewidth=1.5)
    axes[2].set_title("Q-Q Plot")
    axes[2].set_xlabel("Theoretical Quantiles"); axes[2].set_ylabel("Sample Quantiles")

    return save_fig("10_residuals")

def run_forecasting(df: pd.DataFrame, city: str = "London") -> dict:
    """Run full forecasting pipeline and return all results."""
    logger.info(f"Starting forecasting pipeline for city: {city}")

    ts = build_time_series(df, city=city)
    series = ts["value"]

    # ML features
    df_feat = create_lag_features(series)
    X_tr, y_tr, X_te, y_te, test_idx = train_test_split_ts(df_feat)

    # Train models
    rf_res  = train_random_forest(X_tr, y_tr, X_te, y_te, test_idx)
    xgb_res = train_xgboost(X_tr, y_tr, X_te, y_te, test_idx)
    gb_res  = train_gradient_boosting(X_tr, y_tr, X_te, y_te, test_idx)
    ridge_res = train_ridge(X_tr, y_tr, X_te, y_te, test_idx)
    ml_results = [rf_res, xgb_res, gb_res, ridge_res]

    # ──────────────────────────────────────────────────────────────────────────
    # 💾 NEW: SAVE INDIVIDUAL TRAINED MODELS & SCALERS
    # ──────────────────────────────────────────────────────────────────────────
    clean_city_name = city.lower().replace(" ", "_")
    for res in ml_results:
        # Generate a clean filename based on the model name
        model_filename = res["name"].lower().replace(" ", "_")
        
        # Save the machine learning model
        model_path = MODEL_DIR / f"{model_filename}_{clean_city_name}.pkl"
        joblib.dump(res["model"], model_path)
        logger.info(f"Saved trained model binary to: {model_path}")
        
        # If the model used a standard scaler (like RF or Ridge), save it too
        if "scaler" in res:
            scaler_path = MODEL_DIR / f"{model_filename}_scaler_{clean_city_name}.pkl"
            joblib.dump(res["scaler"], scaler_path)
    # ──────────────────────────────────────────────────────────────────────────

    # Ensemble
    ensemble_res = ensemble_predict(ml_results)
    ml_results.append(ensemble_res)

    # ARIMA
    arima_res = train_arima(series)

    # Visualize
    fig_forecast = plot_forecast_comparison(ml_results, arima_res, city)

    all_results = ml_results.copy()
    if arima_res:
        all_results.append(arima_res)
    fig_metrics  = plot_model_metrics_comparison(all_results)

    # Best ML model by RMSE
    best_ml = min(ml_results, key=lambda r: r["metrics"]["RMSE"])
    fig_resid = plot_residuals(best_ml)

    return {
        "ml_results": ml_results,
        "arima_result": arima_res,
        "all_results": all_results,
        "best_model": best_ml,
        "figures": {"forecast": fig_forecast, "metrics": fig_metrics, "residuals": fig_resid},
        "city": city,
    }