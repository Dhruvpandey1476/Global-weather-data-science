"""
run_analysis.py
───────────────
Main entry point for the Global Weather Repository Data Science Analysis.
Runs the full pipeline: load → preprocess → EDA → models → advanced → report.

Usage:
    python run_analysis.py                      # uses synthetic data if CSV not found
    python run_analysis.py --city "New York"    # forecast a different city
    python run_analysis.py --data path/to/data.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import joblib
import os

# ── ensure src/ is importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader    import load_data
from src.preprocessing  import full_preprocess
from src.eda            import run_eda
from src.models         import run_forecasting
from src.advanced       import run_advanced
from src.report         import generate_report

# ── logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── make sure output dirs exist ─────────────────────────────────────────────────
for d in ["outputs/figures", "outputs/report", "outputs/models", "data"]:
    Path(d).mkdir(parents=True, exist_ok=True)


def banner(msg: str):
    line = "═" * 60
    logger.info(f"\n{line}\n  {msg}\n{line}")


def main(data_path: str = "data/GlobalWeatherRepository.csv", city: str = "London"):
    t0 = time.time()

    # ── 1. LOAD ────────────────────────────────────────────────────────────────
    banner("STEP 1 · Loading Data")
    df_raw = load_data(data_path)
    logger.info(f"Raw shape: {df_raw.shape}")

    # ── 2. PREPROCESS ──────────────────────────────────────────────────────────
    banner("STEP 2 · Preprocessing")
    df, outlier_info, missing_report = full_preprocess(df_raw)
    logger.info(f"Clean shape: {df.shape}")

    # ── 3. EDA ─────────────────────────────────────────────────────────────────
    banner("STEP 3 · Exploratory Data Analysis")
    eda_figs = run_eda(df)
    logger.info(f"EDA figures: {list(eda_figs.keys())}")

    # ── 4. FORECASTING ─────────────────────────────────────────────────────────
    banner(f"STEP 4 · Forecasting (city = {city})")
    # Verify city exists, fall back to first available
    available_cities = df["location_name"].unique().tolist()
    if city not in available_cities:
        city = available_cities[0]
        logger.warning(f"City not found — falling back to '{city}'")
    forecast_results = run_forecasting(df, city=city)

    # ── 5. ADVANCED ANALYSES ───────────────────────────────────────────────────
    banner("STEP 5 · Advanced Analyses")
    advanced_results = run_advanced(df)

    # ── 6. REPORT ──────────────────────────────────────────────────────────────
    banner("STEP 6 · Generating HTML Report")
    report_path = generate_report(
        df_shape        = df.shape,
        missing_report  = missing_report,
        outlier_info    = outlier_info,
        eda_figs        = eda_figs,
        forecast_results= forecast_results,
        advanced_results= advanced_results,
        out_path        = "outputs/report/report.html",
    )

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"📄 Report  → {report_path}")
    logger.info(f"📊 Figures → outputs/figures/ ({len(list(Path('outputs/figures').glob('*.png')))} files)")
    logger.info("Done. ✅")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Weather Analysis Pipeline")
    parser.add_argument("--data", default="data/GlobalWeatherRepository.csv",
                        help="Path to GlobalWeatherRepository.csv")
    parser.add_argument("--city", default="London",
                        help="City to use for time-series forecasting")
    args = parser.parse_args()
    main(data_path=args.data, city=args.city)
