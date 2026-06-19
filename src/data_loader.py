"""
Data Loader for Global Weather Repository
Loads real CSV if present, else generates realistic synthetic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CITIES = [
    {"name": "London",        "country": "United Kingdom", "lat": 51.51, "lon": -0.13,  "continent": "Europe",        "base_temp": 11,  "temp_range": 8,  "base_precip": 1.8},
    {"name": "Paris",         "country": "France",         "lat": 48.85, "lon": 2.35,   "continent": "Europe",        "base_temp": 12,  "temp_range": 9,  "base_precip": 1.5},
    {"name": "Berlin",        "country": "Germany",        "lat": 52.52, "lon": 13.40,  "continent": "Europe",        "base_temp": 10,  "temp_range": 10, "base_precip": 1.4},
    {"name": "Madrid",        "country": "Spain",          "lat": 40.42, "lon": -3.70,  "continent": "Europe",        "base_temp": 15,  "temp_range": 12, "base_precip": 0.8},
    {"name": "Rome",          "country": "Italy",          "lat": 41.90, "lon": 12.50,  "continent": "Europe",        "base_temp": 16,  "temp_range": 11, "base_precip": 1.0},
    {"name": "New York",      "country": "United States",  "lat": 40.71, "lon": -74.01, "continent": "North America",  "base_temp": 13,  "temp_range": 14, "base_precip": 1.2},
    {"name": "Los Angeles",   "country": "United States",  "lat": 34.05, "lon": -118.24,"continent": "North America",  "base_temp": 19,  "temp_range": 6,  "base_precip": 0.4},
    {"name": "Chicago",       "country": "United States",  "lat": 41.88, "lon": -87.63, "continent": "North America",  "base_temp": 11,  "temp_range": 16, "base_precip": 1.0},
    {"name": "Toronto",       "country": "Canada",         "lat": 43.65, "lon": -79.38, "continent": "North America",  "base_temp": 9,   "temp_range": 16, "base_precip": 1.1},
    {"name": "Mexico City",   "country": "Mexico",         "lat": 19.43, "lon": -99.13, "continent": "North America",  "base_temp": 16,  "temp_range": 5,  "base_precip": 1.5},
    {"name": "Tokyo",         "country": "Japan",          "lat": 35.68, "lon": 139.69, "continent": "Asia",           "base_temp": 15,  "temp_range": 13, "base_precip": 2.0},
    {"name": "Beijing",       "country": "China",          "lat": 39.91, "lon": 116.39, "continent": "Asia",           "base_temp": 13,  "temp_range": 18, "base_precip": 0.9},
    {"name": "Mumbai",        "country": "India",          "lat": 19.08, "lon": 72.88,  "continent": "Asia",           "base_temp": 27,  "temp_range": 4,  "base_precip": 5.0},
    {"name": "Delhi",         "country": "India",          "lat": 28.61, "lon": 77.21,  "continent": "Asia",           "base_temp": 25,  "temp_range": 15, "base_precip": 1.2},
    {"name": "Singapore",     "country": "Singapore",      "lat": 1.29,  "lon": 103.85, "continent": "Asia",           "base_temp": 27,  "temp_range": 2,  "base_precip": 5.5},
    {"name": "Dubai",         "country": "UAE",            "lat": 25.20, "lon": 55.27,  "continent": "Asia",           "base_temp": 28,  "temp_range": 12, "base_precip": 0.1},
    {"name": "Sydney",        "country": "Australia",      "lat": -33.87,"lon": 151.21, "continent": "Oceania",        "base_temp": 17,  "temp_range": 7,  "base_precip": 1.5},
    {"name": "Melbourne",     "country": "Australia",      "lat": -37.81,"lon": 144.96, "continent": "Oceania",        "base_temp": 15,  "temp_range": 8,  "base_precip": 1.5},
    {"name": "Cairo",         "country": "Egypt",          "lat": 30.06, "lon": 31.25,  "continent": "Africa",         "base_temp": 22,  "temp_range": 11, "base_precip": 0.05},
    {"name": "Lagos",         "country": "Nigeria",        "lat": 6.52,  "lon": 3.38,   "continent": "Africa",         "base_temp": 28,  "temp_range": 3,  "base_precip": 4.5},
    {"name": "Johannesburg",  "country": "South Africa",   "lat": -26.20,"lon": 28.04,  "continent": "Africa",         "base_temp": 16,  "temp_range": 8,  "base_precip": 2.0},
    {"name": "Nairobi",       "country": "Kenya",          "lat": -1.29, "lon": 36.82,  "continent": "Africa",         "base_temp": 18,  "temp_range": 3,  "base_precip": 3.0},
    {"name": "São Paulo",     "country": "Brazil",         "lat": -23.55,"lon": -46.63, "continent": "South America",  "base_temp": 20,  "temp_range": 5,  "base_precip": 4.0},
    {"name": "Buenos Aires",  "country": "Argentina",      "lat": -34.60,"lon": -58.38, "continent": "South America",  "base_temp": 17,  "temp_range": 9,  "base_precip": 2.5},
    {"name": "Lima",          "country": "Peru",           "lat": -12.05,"lon": -77.04, "continent": "South America",  "base_temp": 19,  "temp_range": 4,  "base_precip": 0.1},
    {"name": "Moscow",        "country": "Russia",         "lat": 55.75, "lon": 37.62,  "continent": "Europe",         "base_temp": 5,   "temp_range": 20, "base_precip": 1.2},
    {"name": "Seoul",         "country": "South Korea",    "lat": 37.57, "lon": 126.98, "continent": "Asia",           "base_temp": 12,  "temp_range": 16, "base_precip": 1.5},
    {"name": "Bangkok",       "country": "Thailand",       "lat": 13.75, "lon": 100.52, "continent": "Asia",           "base_temp": 29,  "temp_range": 3,  "base_precip": 4.5},
    {"name": "Istanbul",      "country": "Turkey",         "lat": 41.01, "lon": 28.96,  "continent": "Europe",         "base_temp": 14,  "temp_range": 11, "base_precip": 1.8},
    {"name": "Riyadh",        "country": "Saudi Arabia",   "lat": 24.69, "lon": 46.72,  "continent": "Asia",           "base_temp": 27,  "temp_range": 14, "base_precip": 0.05},
]

MOON_PHASES = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
               "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]

CONDITIONS = [
    "Sunny", "Partly cloudy", "Cloudy", "Overcast", "Mist",
    "Patchy rain possible", "Light rain", "Moderate rain", "Heavy rain",
    "Light snow", "Moderate snow", "Blizzard", "Fog", "Thundery outbreaks possible",
    "Blowing snow", "Clear", "Light drizzle"
]


def generate_synthetic_data(n_days: int = 730) -> pd.DataFrame:
    """Generate realistic synthetic weather data for 30 world cities."""
    np.random.seed(42)
    records = []

    start_date = pd.Timestamp("2022-01-01")

    for city in CITIES:
        dates = pd.date_range(start_date, periods=n_days, freq="D")
        n = len(dates)

        # Seasonal temperature variation (sine wave)
        day_of_year = np.array([d.day_of_year for d in dates])
        seasonal = city["temp_range"] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        # Southern hemisphere flip
        if city["lat"] < 0:
            seasonal = -seasonal

        temp = city["base_temp"] + seasonal + np.random.normal(0, 2, n)
        temp_f = temp * 9 / 5 + 32
        feels_like = temp - np.random.uniform(0, 3, n)

        # Humidity inversely correlated with temp in dry climates
        humidity = np.clip(65 - (temp - city["base_temp"]) * 0.5 + np.random.normal(0, 10, n), 20, 100)

        # Precipitation (higher in rainy season for tropical cities)
        rain_season = city["base_precip"] * (1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 180) / 365))
        precip_mm = np.maximum(0, rain_season + np.random.exponential(0.5, n) * np.random.binomial(1, 0.3, n))
        precip_in = precip_mm / 25.4

        # Wind
        wind_kph = np.abs(np.random.normal(18, 8, n))
        wind_mph = wind_kph * 0.621371
        gust_kph = wind_kph * np.random.uniform(1.2, 1.8, n)
        gust_mph = gust_kph * 0.621371
        wind_degree = np.random.randint(0, 360, n)

        # Pressure (inversely related to precipitation)
        pressure_mb = 1013 - precip_mm * 2 + np.random.normal(0, 5, n)
        pressure_in = pressure_mb * 0.02953

        # Cloud cover (correlated with precipitation)
        cloud = np.clip(precip_mm * 10 + np.random.normal(40, 20, n), 0, 100).astype(int)

        # Visibility (lower in high humidity/rain)
        visibility_km = np.clip(20 - precip_mm * 3 - (humidity - 50) * 0.1 + np.random.normal(0, 2, n), 1, 30)
        visibility_miles = visibility_km * 0.621371

        uv_index = np.clip(8 - abs(city["lat"]) / 15 + seasonal * 0.15 + np.random.uniform(0, 2, n), 0, 11)

        # Air quality (worse in industrial cities)
        aq_base = 1.5 if city["name"] in ["Delhi", "Beijing", "Cairo", "Mumbai", "Lagos"] else 0.8
        co = np.abs(np.random.normal(300 * aq_base, 80, n))
        ozone = np.abs(np.random.normal(30, 10, n))
        no2 = np.abs(np.random.normal(20 * aq_base, 8, n))
        so2 = np.abs(np.random.normal(8 * aq_base, 4, n))
        pm25 = np.abs(np.random.normal(18 * aq_base, 10, n))
        pm10 = pm25 * np.random.uniform(1.4, 2.0, n)
        epa_index = np.clip((pm25 / 12).astype(int) + 1, 1, 6)
        defra_index = np.clip((pm10 / 20).astype(int) + 1, 1, 10)

        # Moon
        moon_illum = np.random.randint(0, 101, n)
        moon_phase_idx = (moon_illum // 13) % len(MOON_PHASES)

        # Conditions based on precip + cloud
        conditions = []
        for p, c in zip(precip_mm, cloud):
            if p > 5:
                cond = "Heavy rain"
            elif p > 2:
                cond = "Moderate rain"
            elif p > 0.5:
                cond = "Light rain"
            elif c > 80:
                cond = "Overcast"
            elif c > 50:
                cond = "Cloudy"
            elif c > 20:
                cond = "Partly cloudy"
            else:
                cond = "Sunny" if city["lat"] > 0 else "Clear"
            conditions.append(cond)

        # Sunrise/sunset (approximate)
        sunrise_hour = 6 + abs(city["lat"]) / 90 * 2 * np.sin(2 * np.pi * day_of_year / 365)
        sunset_hour = 18 + abs(city["lat"]) / 90 * 2 * np.sin(2 * np.pi * day_of_year / 365)

        for i, dt in enumerate(dates):
            records.append({
                "country": city["country"],
                "location_name": city["name"],
                "latitude": city["lat"],
                "longitude": city["lon"],
                "continent": city["continent"],
                "timezone": "UTC",
                "last_updated_epoch": int(dt.timestamp()),
                "last_updated": dt.strftime("%Y-%m-%d %H:%M"),
                "temperature_celsius": round(temp[i], 1),
                "temperature_fahrenheit": round(temp_f[i], 1),
                "wind_mph": round(wind_mph[i], 1),
                "wind_kph": round(wind_kph[i], 1),
                "wind_degree": int(wind_degree[i]),
                "wind_direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(wind_degree[i] / 45) % 8],
                "pressure_mb": round(pressure_mb[i], 1),
                "pressure_in": round(pressure_in[i], 2),
                "precip_mm": round(precip_mm[i], 2),
                "precip_in": round(precip_in[i], 3),
                "humidity": int(humidity[i]),
                "cloud": int(cloud[i]),
                "feels_like_celsius": round(feels_like[i], 1),
                "feels_like_fahrenheit": round(feels_like[i] * 9 / 5 + 32, 1),
                "visibility_km": round(visibility_km[i], 1),
                "visibility_miles": round(visibility_miles[i], 1),
                "uv_index": round(uv_index[i], 1),
                "gust_mph": round(gust_mph[i], 1),
                "gust_kph": round(gust_kph[i], 1),
                "air_quality_Carbon_Monoxide": round(co[i], 1),
                "air_quality_Ozone": round(ozone[i], 1),
                "air_quality_Nitrogen_dioxide": round(no2[i], 1),
                "air_quality_Sulphur_dioxide": round(so2[i], 1),
                "air_quality_PM2.5": round(pm25[i], 1),
                "air_quality_PM10": round(pm10[i], 1),
                "air_quality_us-epa-index": int(epa_index[i]),
                "air_quality_gb-defra-index": int(defra_index[i]),
                "sunrise": f"{int(sunrise_hour[i]):02d}:{int((sunrise_hour[i] % 1) * 60):02d} AM",
                "sunset": f"{int(sunset_hour[i]):02d}:{int((sunset_hour[i] % 1) * 60):02d} PM",
                "moonrise": "10:30 PM",
                "moonset": "09:15 AM",
                "moon_phase": MOON_PHASES[int(moon_phase_idx[i])],
                "moon_illumination": int(moon_illum[i]),
                "condition_text": conditions[i],
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated synthetic dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_data(data_path: str = "data/GlobalWeatherRepository.csv") -> pd.DataFrame:
    """Load dataset from CSV or generate synthetic data."""
    path = Path(data_path)
    if path.exists():
        logger.info(f"Loading real dataset from {data_path}")
        df = pd.read_csv(data_path)
        if "continent" not in df.columns:
            # Add continent based on country (simplified)
            continent_map = {
                "United Kingdom": "Europe", "France": "Europe", "Germany": "Europe",
                "United States": "North America", "Canada": "North America",
                "Japan": "Asia", "China": "Asia", "India": "Asia",
                "Australia": "Oceania", "Brazil": "South America",
                "Egypt": "Africa", "Nigeria": "Africa", "South Africa": "Africa",
            }
            df["continent"] = df["country"].map(continent_map).fillna("Unknown")
        logger.info(f"Loaded {df.shape[0]} rows from CSV")
    else:
        logger.info("Dataset not found, generating synthetic data...")
        df = generate_synthetic_data()
        df.to_csv("data/GlobalWeatherRepository_synthetic.csv", index=False)
        logger.info("Synthetic data saved to data/GlobalWeatherRepository_synthetic.csv")
    return df
