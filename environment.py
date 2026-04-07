import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from pydantic import Field
from openenv import Env, Observation, Action, Reward, Task, Grader
import pickle
import os

from utils import (
    impute_missing, remove_duplicates, correct_dtypes,
    validate_coordinates, normalize_dates, predict_severity,
    load_ground_truth
)

# ---------------------- Typed Models ----------------------
class DisasterObservation(Observation):
    step: int = 0
    done: bool = False
    missing_counts: Dict[str, int] = Field(default_factory=dict)
    duplicate_count: int = 0
    geo_invalid_count: int = 0
    date_invalid_count: int = 0
    prediction_made: bool = False
    cleaning_steps_completed: List[str] = Field(default_factory=list)

class DisasterAction(Action):
    action_string: str

class DisasterReward(Reward):
    value: float = 0.0

# ---------------------- Environment ----------------------
class DisasterCleanerEnv(Env):
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.data = None
        self.clean_gt = None
        self.step_count = 0
        self.completed_steps = set()
        self.prediction_result = None
        self.prediction_correct = False
        self.max_steps = 20

        # Load ground truth (for grading only, not used during step)
        self.clean_gt = load_ground_truth()

        # Load trained model
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)

    async def reset(self) -> DisasterObservation:
        """Reset environment: load messy.csv, reset counters."""
        self.step_count = 0
        self.completed_steps.clear()
        self.prediction_result = None
        self.prediction_correct = False

        # Load the messy dataset (deterministic)
        self.data = pd.read_csv("data/messy.csv")
        return self._get_observation()

    async def step(self, action: DisasterAction) -> Tuple[DisasterObservation, DisasterReward, bool, Dict]:
        action_str = action.action_string.strip().lower()
        reward_val = 0.0
        done = False
        info = {}

        # --- Action handling with partial rewards ---
        if action_str.startswith("impute_missing:"):
            column = action_str.split(":", 1)[1]
            if column in self.data.columns and f"impute_{column}" not in self.completed_steps:
                self.data = impute_missing(self.data, column)
                self.completed_steps.add(f"impute_{column}")
                reward_val += 0.3
            else:
                reward_val -= 0.3

        elif action_str == "remove_duplicates":
            if "remove_duplicates" not in self.completed_steps:
                before = len(self.data)
                self.data = remove_duplicates(self.data)
                after = len(self.data)
                if after < before:
                    reward_val += 0.3
                    self.completed_steps.add("remove_duplicates")
                else:
                    reward_val -= 0.3
            else:
                reward_val -= 0.3

        elif action_str == "correct_dtypes":
            if "correct_dtypes" not in self.completed_steps:
                self.data = correct_dtypes(self.data)
                self.completed_steps.add("correct_dtypes")
                reward_val += 0.3
            else:
                reward_val -= 0.3

        elif action_str == "validate_coordinates":
            if "validate_coordinates" not in self.completed_steps:
                self.data, fixed = validate_coordinates(self.data)
                if fixed > 0:
                    reward_val += 0.3
                else:
                    reward_val += 0.1  # still valid action
                self.completed_steps.add("validate_coordinates")
            else:
                reward_val -= 0.3

        elif action_str == "normalize_dates":
            if "normalize_dates" not in self.completed_steps:
                self.data = normalize_dates(self.data)
                self.completed_steps.add("normalize_dates")
                reward_val += 0.3
            else:
                reward_val -= 0.3

        elif action_str == "predict_severity":
            if "predict_severity" not in self.completed_steps:
                # Predict on current cleaned data
                pred = predict_severity(self.data, self.model)
                self.prediction_result = pred
                # Compare with ground truth severity_class
                merged = self.data[["event_id"]].merge(self.clean_gt, on="event_id", how="left")
                actual = merged["severity_class"].values
                if len(pred) == len(actual):
                    from sklearn.metrics import accuracy_score
                    acc = accuracy_score(pred, actual)
                    reward_val += 0.4 * acc
                    self.prediction_correct = acc > 0.8
                else:
                    reward_val -= 0.3
                self.completed_steps.add("predict_severity")
            else:
                reward_val -= 0.3

        else:
            reward_val -= 0.3  # invalid action

        self.step_count += 1

        # Check if all required steps for this task are done
        required_steps = self._get_required_steps()
        all_cleaning_done = required_steps.issubset(self.completed_steps)
        if all_cleaning_done and "predict_severity" in self.completed_steps:
            done = True
            reward_val += 1.0  # bonus for full completion

        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_observation()
        reward = DisasterReward(value=reward_val)
        return obs, reward, done, info

    async def state(self) -> Dict[str, Any]:
        return {
            "task": self.task_name,
            "step": self.step_count,
            "completed_steps": list(self.completed_steps),
            "data_shape": self.data.shape if self.data is not None else None,
            "prediction_correct": self.prediction_correct,
        }

    def _get_observation(self) -> DisasterObservation:
        if self.data is None:
            return DisasterObservation(step=self.step_count, done=False)
        missing = self.data.isnull().sum().to_dict()
        return DisasterObservation(
            step=self.step_count,
            done=False,
            missing_counts=missing,
            duplicate_count=self.data.duplicated().sum(),
            geo_invalid_count=self._count_geo_invalid(),
            date_invalid_count=self._count_date_invalid(),
            prediction_made="predict_severity" in self.completed_steps,
            cleaning_steps_completed=list(self.completed_steps),
        )

    def _count_geo_invalid(self) -> int:
        if "latitude" not in self.data or "longitude" not in self.data:
            return 0
        lat_invalid = (~self.data["latitude"].between(-90, 90)) | self.data["latitude"].isnull()
        lon_invalid = (~self.data["longitude"].between(-180, 180)) | self.data["longitude"].isnull()
        return (lat_invalid | lon_invalid).sum()

    def _count_date_invalid(self) -> int:
        if "date" not in self.data:
            return 0
        return self.data["date"].isnull().sum()

    def _get_required_steps(self) -> set:
        if self.task_name == "easy":
            return {"impute_disaster_type", "impute_magnitude", "correct_dtypes"}
        elif self.task_name == "medium":
            return {"impute_disaster_type", "impute_magnitude", "correct_dtypes",
                    "validate_coordinates", "normalize_dates"}
        else:  # hard
            return {"impute_disaster_type", "impute_magnitude", "correct_dtypes",
                    "validate_coordinates", "normalize_dates", "remove_duplicates",
                    "predict_severity"}

# ---------------------- Graders (0.0–1.0, deterministic) ----------------------
class EasyGrader(Grader):
    async def score(self, env: DisasterCleanerEnv, trajectory: List) -> float:
        df = env.data
        if df is None:
            return 0.0
        # Must have imputed disaster_type and magnitude, and correct dtypes
        required_cols = ["disaster_type", "magnitude"]
        missing_imputed = all(df[col].notnull().all() for col in required_cols)
        dtypes_correct = df["magnitude"].dtype in [float, int] and df["disaster_type"].dtype == object
        return 1.0 if (missing_imputed and dtypes_correct) else 0.0

class MediumGrader(Grader):
    async def score(self, env: DisasterCleanerEnv, trajectory: List) -> float:
        df = env.data
        if df is None:
            return 0.0
        easy_score = await EasyGrader().score(env, trajectory)
        if easy_score < 1.0:
            return 0.0
        geo_valid = (df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)).all()
        dates_valid = df["date"].notnull().all()
        return 1.0 if (geo_valid and dates_valid) else 0.0

class HardGrader(Grader):
    async def score(self, env: DisasterCleanerEnv, trajectory: List) -> float:
        df = env.data
        if df is None:
            return 0.0
        medium_score = await MediumGrader().score(env, trajectory)
        if medium_score < 1.0:
            return 0.0
        no_duplicates = df.duplicated().sum() == 0
        prediction_correct = env.prediction_correct
        return 1.0 if (no_duplicates and prediction_correct) else 0.0
