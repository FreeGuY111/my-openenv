import pandas as pd
import numpy as np
import pickle

def impute_missing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values: mode for categorical, median for numeric."""
    if column not in df.columns:
        return df
    if df[column].dtype in ['object', 'category']:
        mode_val = df[column].mode()[0] if not df[column].mode().empty else "unknown"
        df[column].fillna(mode_val, inplace=True)
    else:
        median_val = df[column].median()
        df[column].fillna(median_val, inplace=True)
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if "magnitude" in df.columns:
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    for col in ["affected_population", "damage_cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def validate_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clamp out-of-range coordinates to valid ranges. Return (df, fixed_count)."""
    fixed = 0
    if "latitude" in df.columns:
        invalid = ~df["latitude"].between(-90, 90)
        df.loc[invalid, "latitude"] = df.loc[invalid, "latitude"].clip(-90, 90)
        fixed += invalid.sum()
    if "longitude" in df.columns:
        invalid = ~df["longitude"].between(-180, 180)
        df.loc[invalid, "longitude"] = df.loc[invalid, "longitude"].clip(-180, 180)
        fixed += invalid.sum()
    return df, fixed

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def predict_severity(df: pd.DataFrame, model) -> np.ndarray:
    """Predict severity class (Low/Medium/High) using trained model."""
    features = []
    for _, row in df.iterrows():
        mag = row.get("magnitude", 0)
        lat = row.get("latitude", 0)
        lon = row.get("longitude", 0)
        pop = row.get("affected_population", 0)
        cost = row.get("damage_cost", 0)
        features.append([mag, lat, lon, pop, cost])
    X = np.array(features, dtype=float)
    # Impute NaNs with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    preds = model.predict(X)
    return preds

def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv("data/clean.csv")
