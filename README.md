---
title: Disaster Data Prep & Flood Risk
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
#  Disaster Data Prep & Flood Risk (India)

**An OpenEnv environment that simulates cleaning messy rainfall and river datasets inspired by IMD and CWC.**  
Built for the OpenEnv Hackathon — deterministic, lightweight, and ready to deploy on Hugging Face Spaces with Docker.

---

##  What is this?

If you've ever worked with real‑world government data in India, you know it rarely arrives clean. This environment recreates that exact chaos in a tiny, self‑contained dataset (40 rows) and asks an AI agent to fix it.

The dataset contains:

- **`date`** – observation date  
- **`state`** – Indian state name (inconsistent case, e.g., `rajasthan`, `Rajasthan`, `RAJASTHAN`)  
- **`rainfall_mm`** – rainfall in mm, but often stored as `"120mm"` (string with unit) or simply missing  
- **`river_level_m`** – river water level in metres, with missing values  
- **`flood_risk`** – `High`, `Medium`, or `Low` (sometimes wrong because it hasn't been updated after data changes)

All the messiness is **intentional** — mixed units, missing values, duplicates, casing inconsistencies, and even incorrect flood risk labels.

---

##  Flood Risk Logic (Fixed)

The environment uses a **deterministic rule** to compute flood risk:


# Disaster Data Preparation & Flood Risk Analysis Environment (India)

This OpenEnv environment simulates the cleaning and preparation of rainfall and flood datasets inspired by the Indian Meteorological Department (IMD) and Central Water Commission (CWC). The agent must clean messy data, normalize values, and recompute flood risk indicators.

## Problem Description

Disaster management agencies often receive raw data with missing values, inconsistent units, duplicate entries, and incorrect labels. Before feeding data into prediction models, it must be standardized. This environment provides a deterministic, lightweight dataset (40 rows) with intentionally introduced issues. The agent's goal is to apply a sequence of cleaning actions to maximize a reward based on cleanliness, consistency, and flood risk accuracy.

## Dataset

The dataset (`data/rainfall_flood_dataset.csv`) contains columns:
- `date`: observation date
- `state`: Indian state name
- `rainfall_mm`: rainfall in millimeters (with mixed units and missing values)
- `river_level_m`: river water level in meters
- `flood_risk`: categorical label (High/Medium/Low)

Flood risk logic:
- High: rainfall > 100 mm AND river level > 6 m
- Medium: rainfall > 50 mm (and not High)
- Low: otherwise

## Action Space

Seven granular actions:
1. `fix_rainfall_units` – convert strings like "120mm" to numeric 120.
2. `fill_missing_rainfall(method)` – impute missing rainfall using 'mean' or 'median'.
3. `fill_missing_river_level(method)` – impute missing river levels.
4. `normalize_state_names` – convert state names to title case, standardize abbreviations.
5. `remove_duplicates` – drop duplicate rows.
6. `detect_outliers(column)` – mark outliers as NaN using IQR method.
7. `recompute_flood_risk` – recalculate `flood_risk` column using the defined logic.

## Tasks

| Task   | Max Steps | Description |
|--------|-----------|-------------|
| easy   | 5         | Fix basic issues: missing values and inconsistent casing. |
| medium | 8         | Clean mixed units, fill missing, normalize states, remove duplicates. |
| hard   | 12        | Full pipeline: clean all issues, detect outliers, recompute flood risk accurately. |

## Setup and Deployment

### Local Development
```bash
pip install -r requirements.txt
uvicorn server.main:app --reload --port 7860
