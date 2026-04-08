from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum

class ActionType(str, Enum):
    FIX_RAINFALL_UNITS = "fix_rainfall_units"
    FILL_MISSING_RAINFALL = "fill_missing_rainfall"
    FILL_MISSING_RIVER_LEVEL = "fill_missing_river_level"
    NORMALIZE_STATE_NAMES = "normalize_state_names"
    REMOVE_DUPLICATES = "remove_duplicates"
    DETECT_OUTLIERS = "detect_outliers"
    RECOMPUTE_FLOOD_RISK = "recompute_flood_risk"

class Action(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = {}

class DatasetSummary(BaseModel):
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_summary: Dict[str, int]

class Observation(BaseModel):
    dataset_summary: DatasetSummary
    missing_value_count: int
    duplicate_count: int
    schema_validity_score: float
    anomaly_score: float
    steps_remaining: int

class Reward(BaseModel):
    value: float
    components: Dict[str, float]
    penalty: float = 0.0

class State(BaseModel):
    data: List[Dict[str, Any]]
    metrics: Dict[str, Any]
