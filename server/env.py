import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from copy import deepcopy
import os

from .models import Observation, Action, Reward, State, DatasetSummary

class FloodRiskEnv:
    def __init__(self, data_path: str = "data/rainfall_flood_dataset.csv", max_steps: int = 20):
        self.data_path = data_path
        self.max_steps = max_steps
        self.original_df: pd.DataFrame = None
        self.df: pd.DataFrame = None
        self.steps_taken = 0
        self.task_corruption = "hard"  # default, can be overridden in reset
        self.action_history = []
        self._load_original_data()

    def _load_original_data(self):
        self.original_df = pd.read_csv(self.data_path)
        # Ensure proper dtypes for original clean reference
        self.original_df['date'] = pd.to_datetime(self.original_df['date'])
        self.original_df['state'] = self.original_df['state'].str.title().str.strip()
        self.original_df['rainfall_mm'] = pd.to_numeric(self.original_df['rainfall_mm'], errors='coerce')
        self.original_df['river_level_m'] = pd.to_numeric(self.original_df['river_level_m'], errors='coerce')
        # Recompute flood_risk based on logic
        self.original_df = self._compute_flood_risk(self.original_df)

    def _compute_flood_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        def risk(row):
            if pd.isna(row['rainfall_mm']) or pd.isna(row['river_level_m']):
                return np.nan
            if row['rainfall_mm'] > 100 and row['river_level_m'] > 6:
                return 'High'
            elif row['rainfall_mm'] > 50:
                return 'Medium'
            else:
                return 'Low'
        df['flood_risk'] = df.apply(risk, axis=1)
        return df

    def _apply_corruption(self, df: pd.DataFrame, level: str) -> pd.DataFrame:
        df = df.copy()
        if level == "easy":
            # Only basic issues: a few missing values, inconsistent case
            df.loc[2, 'rainfall_mm'] = np.nan
            df.loc[7, 'river_level_m'] = np.nan
            df['state'] = df['state'].apply(lambda x: x.lower() if np.random.rand() > 0.8 else x)
        elif level == "medium":
            # Include mixed units, more missing, duplicates
            df['rainfall_mm'] = df['rainfall_mm'].astype(str)
            df.loc[df.sample(frac=0.2).index, 'rainfall_mm'] = df.loc[df.sample(frac=0.2).index, 'rainfall_mm'] + 'mm'
            df.loc[df.sample(frac=0.15).index, 'rainfall_mm'] = np.nan
            df.loc[df.sample(frac=0.1).index, 'river_level_m'] = np.nan
            df = pd.concat([df, df.sample(frac=0.1)], ignore_index=True)
            df['state'] = df['state'].apply(lambda x: x.lower() if np.random.rand() > 0.5 else x)
        else:  # hard
            # All issues plus some wrong flood_risk labels
            df['rainfall_mm'] = df['rainfall_mm'].astype(str)
            mask = df.sample(frac=0.4).index
            df.loc[mask, 'rainfall_mm'] = df.loc[mask, 'rainfall_mm'] + 'mm'
            df.loc[df.sample(frac=0.3).index, 'rainfall_mm'] = np.nan
            df.loc[df.sample(frac=0.2).index, 'river_level_m'] = np.nan
            df = pd.concat([df, df.sample(frac=0.2)], ignore_index=True)
            df['state'] = df['state'].apply(lambda x: x.lower() if np.random.rand() > 0.7 else x)
            # Introduce wrong flood_risk
            df['flood_risk'] = np.where(df['flood_risk'] == 'High', 'Medium', df['flood_risk'])
        return df

    def reset(self, task: str = "hard") -> Observation:
        self.task_corruption = task
        base_df = self.original_df.copy()
        self.df = self._apply_corruption(base_df, task)
        self.steps_taken = 0
        self.action_history = []
        return self._get_observation()

    def _get_observation(self) -> Observation:
        summary = self._get_dataset_summary()
        missing = self.df.isna().sum().sum()
        dup = self.df.duplicated().sum()
        schema_score = self._compute_schema_validity_score()
        anomaly_score = self._compute_anomaly_score()
        steps_left = self.max_steps - self.steps_taken
        return Observation(
            dataset_summary=summary,
            missing_value_count=int(missing),
            duplicate_count=int(dup),
            schema_validity_score=schema_score,
            anomaly_score=anomaly_score,
            steps_remaining=steps_left
        )

    def _get_dataset_summary(self) -> DatasetSummary:
        return DatasetSummary(
            row_count=len(self.df),
            column_count=len(self.df.columns),
            columns=list(self.df.columns),
            dtypes={col: str(dtype) for col, dtype in self.df.dtypes.items()},
            missing_summary=self.df.isna().sum().to_dict()
        )

    def _compute_schema_validity_score(self) -> float:
        # Checks: rainfall and river_level should be numeric, no missing in date/state,
        # flood_risk in {High,Medium,Low}
        score = 1.0
        if not pd.api.types.is_numeric_dtype(self.df['rainfall_mm']):
            score -= 0.3
        if not pd.api.types.is_numeric_dtype(self.df['river_level_m']):
            score -= 0.3
        if self.df['date'].isna().any() or self.df['state'].isna().any():
            score -= 0.2
        valid_risks = {'High', 'Medium', 'Low'}
        if not set(self.df['flood_risk'].dropna().unique()).issubset(valid_risks):
            score -= 0.2
        return max(0.0, score)

    def _compute_anomaly_score(self) -> float:
        # Simple outlier detection score based on IQR for numeric columns
        score = 0.0
        if pd.api.types.is_numeric_dtype(self.df['rainfall_mm']):
            q1 = self.df['rainfall_mm'].quantile(0.25)
            q3 = self.df['rainfall_mm'].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.df['rainfall_mm'] < q1 - 1.5*iqr) | (self.df['rainfall_mm'] > q3 + 1.5*iqr)).sum()
            score += outliers / len(self.df) if len(self.df) > 0 else 0
        if pd.api.types.is_numeric_dtype(self.df['river_level_m']):
            q1 = self.df['river_level_m'].quantile(0.25)
            q3 = self.df['river_level_m'].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.df['river_level_m'] < q1 - 1.5*iqr) | (self.df['river_level_m'] > q3 + 1.5*iqr)).sum()
            score += outliers / len(self.df) if len(self.df) > 0 else 0
        return min(1.0, score / 2)  # average anomaly proportion

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.steps_taken += 1
        prev_df = self.df.copy()
        action_success = True
        penalty = 0.0
        try:
            if action.action_type == Action.FIX_RAINFALL_UNITS:
                self._fix_rainfall_units()
            elif action.action_type == Action.FILL_MISSING_RAINFALL:
                method = action.params.get('method', 'mean')
                self._fill_missing_rainfall(method)
            elif action.action_type == Action.FILL_MISSING_RIVER_LEVEL:
                method = action.params.get('method', 'mean')
                self._fill_missing_river_level(method)
            elif action.action_type == Action.NORMALIZE_STATE_NAMES:
                self._normalize_state_names()
            elif action.action_type == Action.REMOVE_DUPLICATES:
                self._remove_duplicates()
            elif action.action_type == Action.DETECT_OUTLIERS:
                column = action.params.get('column', 'rainfall_mm')
                self._detect_outliers(column)
            elif action.action_type == Action.RECOMPUTE_FLOOD_RISK:
                self._recompute_flood_risk()
            else:
                action_success = False
                penalty = 0.2
        except Exception as e:
            action_success = False
            penalty = 0.3

        # Compute reward
        cleanliness = self._cleanliness_score()
        consistency = self._consistency_score()
        flood_accuracy = self._flood_risk_accuracy()
        reward_val = 0.4 * cleanliness + 0.3 * consistency + 0.3 * flood_accuracy
        if not action_success:
            reward_val -= penalty
        # Penalize redundant actions that changed nothing
        if prev_df.equals(self.df) and action_success:
            reward_val -= 0.1

        # Termination conditions
        done = (self.steps_taken >= self.max_steps) or (self._is_perfect())
        info = {"cleanliness": cleanliness, "consistency": consistency, "flood_accuracy": flood_accuracy}

        obs = self._get_observation()
        reward = Reward(value=reward_val, components={
            "cleanliness": cleanliness,
            "consistency": consistency,
            "flood_accuracy": flood_accuracy
        }, penalty=penalty if not action_success else 0.0)

        return obs, reward, done, info

    # Action implementations
    def _fix_rainfall_units(self):
        self.df['rainfall_mm'] = self.df['rainfall_mm'].astype(str).str.replace('mm', '', regex=False)
        self.df['rainfall_mm'] = pd.to_numeric(self.df['rainfall_mm'], errors='coerce')

    def _fill_missing_rainfall(self, method: str):
        if method == 'mean':
            val = self.df['rainfall_mm'].mean()
        elif method == 'median':
            val = self.df['rainfall_mm'].median()
        else:
            val = 0
        self.df['rainfall_mm'].fillna(val, inplace=True)

    def _fill_missing_river_level(self, method: str):
        if method == 'mean':
            val = self.df['river_level_m'].mean()
        elif method == 'median':
            val = self.df['river_level_m'].median()
        else:
            val = 0
        self.df['river_level_m'].fillna(val, inplace=True)

    def _normalize_state_names(self):
        self.df['state'] = self.df['state'].str.title().str.strip()
        # Standardize common abbreviations
        self.df['state'] = self.df['state'].replace({'Up': 'Uttar Pradesh', 'Uttar Pradesh': 'Uttar Pradesh'})

    def _remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def _detect_outliers(self, column: str):
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        # Mark outliers as NaN (simulate removal)
        self.df.loc[(self.df[column] < lower) | (self.df[column] > upper), column] = np.nan

    def _recompute_flood_risk(self):
        self.df = self._compute_flood_risk(self.df)

    def _cleanliness_score(self) -> float:
        # Proportion of non-missing values in key numeric columns
        total_cells = len(self.df) * 2  # rainfall and river_level
        missing = self.df['rainfall_mm'].isna().sum() + self.df['river_level_m'].isna().sum()
        return 1.0 - (missing / total_cells) if total_cells > 0 else 1.0

    def _consistency_score(self) -> float:
        # Check if rainfall and river_level are numeric, state names consistent, no duplicates
        score = 1.0
        if not pd.api.types.is_numeric_dtype(self.df['rainfall_mm']):
            score -= 0.3
        if not pd.api.types.is_numeric_dtype(self.df['river_level_m']):
            score -= 0.3
        if self.df.duplicated().sum() > 0:
            score -= 0.2
        # State names should be title case and not contain weird chars
        if not self.df['state'].str.istitle().all():
            score -= 0.2
        return max(0.0, score)

    def _flood_risk_accuracy(self) -> float:
        # Compare current flood_risk with that computed from ground truth logic
        if self.df['flood_risk'].isna().any():
            return 0.0
        temp_df = self.df.copy()
        temp_df = self._compute_flood_risk(temp_df)
        matches = (self.df['flood_risk'] == temp_df['flood_risk']).sum()
        return matches / len(self.df) if len(self.df) > 0 else 1.0

    def _is_perfect(self) -> bool:
        return (self._cleanliness_score() == 1.0 and 
                self._consistency_score() == 1.0 and 
                self._flood_risk_accuracy() == 1.0)

    def state(self) -> State:
        return State(
            data=self.df.to_dict(orient='records'),
            metrics={
                "steps_taken": self.steps_taken,
                "cleanliness": self._cleanliness_score(),
                "consistency": self._consistency_score(),
                "flood_accuracy": self._flood_risk_accuracy()
            }
        )
