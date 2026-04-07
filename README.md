# Disaster Data Cleaning and Severity Prediction Environment

An OpenEnv-compliant environment for training and evaluating language model agents on a realistic disaster response task: cleaning inconsistent field data and predicting disaster severity. The environment uses a dataset of 100 disaster events from India and South Asia, with patterns derived from EM-DAT (CRED) and Indian disaster management reports.

---

## Real-World Utility

Disaster management agencies receive data from multiple sources that often contain missing values, duplicate records, invalid GPS coordinates, inconsistent date formats, and type errors. Manual cleaning is slow and error-prone. This environment simulates that challenge: an agent must clean a structured dataset of disaster events and then predict severity (Low, Medium, High) using a pre-trained Random Forest model. Successful agents can help accelerate situational awareness and resource allocation.

---

## Tasks and Difficulty

The environment defines three tasks with increasing difficulty. Each task requires the agent to complete a minimum set of cleaning steps. Graders are deterministic and return a score in the range [0.0, 1.0].

- **Easy**: Impute missing values in `disaster_type` and `magnitude`; correct data types (convert numeric columns).
- **Medium**: All easy steps plus validate and fix geographic coordinates (latitude between -90 and 90, longitude between -180 and 180); normalise dates to ISO format YYYY-MM-DD.
- **Hard**: All medium steps plus remove duplicate records; predict severity using the trained model.

The graders (`EasyGrader`, `MediumGrader`, `HardGrader`) evaluate the final dataset state after the episode ends. Only the reward function provides dense feedback during the episode.

---

## Action Space

The agent outputs exact strings (case-insensitive, format-sensitive). The following actions are available:

| Action string | Effect |
|---------------|--------|
| `impute_missing:<column>` | Impute missing values (mode for categorical, median for numeric). Example: `impute_missing:disaster_type` |
| `remove_duplicates` | Drop duplicate rows, keeping the first occurrence. |
| `correct_dtypes` | Convert magnitude, affected_population, damage_cost to numeric types. |
| `validate_coordinates` | Clamp latitude to [-90,90] and longitude to [-180,180]. |
| `normalize_dates` | Parse various date formats into ISO YYYY-MM-DD. |
| `predict_severity` | Run the Random Forest model on the current data and compute prediction accuracy against ground truth. |

Repeated or invalid actions incur a penalty.

---

## Observation Space

After each step, the environment returns a `DisasterObservation` object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current step count (1-based). |
| `done` | bool | Whether the episode has ended. |
| `missing_counts` | Dict[str, int] | Number of missing values per column. |
| `duplicate_count` | int | Number of duplicate rows. |
| `geo_invalid_count` | int | Rows with latitude or longitude out of range. |
| `date_invalid_count` | int | Rows where the date could not be parsed. |
| `prediction_made` | bool | Whether `predict_severity` has been called. |
| `cleaning_steps_completed` | List[str] | Names of completed cleaning actions. |

This dense observation allows the agent to track progress and decide which action to take next.

---

## Reward Function

Rewards are given per step and accumulated over the episode.

| Event | Reward |
|-------|--------|
| Correct cleaning step (first time) | +0.3 |
| Fixing at least one invalid coordinate | +0.3 |
| Prediction accuracy (0.0–1.0) scaled by 0.4 | +0.4 * accuracy |
| Episode completes with all required steps done | +1.0 (bonus) |
| Invalid action (unknown string or repeat of completed step) | -0.3 |

For example, if the agent imputes `disaster_type` correctly, it receives +0.3. Repeating the same imputation later yields -0.3. Prediction reward is proportional to accuracy (e.g., 85% accuracy gives +0.34).

---

## Dataset

The dataset contains 100 disaster events from India and South Asia (India, Nepal, Bangladesh, Sri Lanka, Pakistan, Myanmar) for the period 2020–2023. It is synthetic but statistically consistent with real-world patterns from EM-DAT.

### Columns

- `event_id`: Unique identifier (IND001 to IND100)
- `disaster_type`: flood, cyclone, earthquake, landslide, drought, heatwave
- `latitude`: -90 to 90
- `longitude`: -180 to 180
- `magnitude`: 1.0–9.0 (Richter scale for earthquakes, otherwise intensity)
- `date`: Event date
- `affected_population`: Number of people affected (integer)
- `damage_cost`: Estimated cost in USD (integer)
- `severity_class`: Low, Medium, High (only in `clean.csv`, used for grading)

### Messy Dataset (`data/messy.csv`)

The messy dataset includes intentional real-world errors:
- Duplicate records (e.g., IND001 appears twice)
- Missing values in magnitude, affected_population, and disaster_type
- Invalid coordinates (latitude = 999, longitude = 180.95, latitude = -200)
- Mixed date formats: `2023-07-15`, `15/01/2023`, `May 22 2023`, `invalid-date`
- Type errors (some magnitude values stored as strings)

### Ground Truth (`data/clean.csv`)

The fully cleaned and validated version with the `severity_class` column. Used only by the graders and not exposed to the agent.

---

## Machine Learning Component

- **Model**: Random Forest Classifier from scikit-learn.
- **Input features**: magnitude, latitude, longitude, affected_population, damage_cost.
- **Output**: Low, Medium, or High.
- **Training**: The notebook `notebooks/training.ipynb` generates the dataset and trains the model, saving it as `model.pkl`.
- **Integration**: The environment loads `model.pkl` at initialisation. When the agent calls `predict_severity`, the model predicts on the current cleaned data, and the reward is computed by comparing predictions against the ground truth.

---

## Setup and Usage

### 1. for Cloning the repository...
```bash
git clone https://github.com/your-username/disaster-cleaner-env
cd disaster-cleaner-env
