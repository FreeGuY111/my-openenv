# Data Cleaning & Severity Prediction (India/South Asia disaster)

OpenEnv environment for cleaning real‑world disaster datasets (missing values, invalid coordinates, duplicates) and predicting severity (Low/Medium/High) using a Random Forest model.

## Real‑World Utilitys
Can be used by disaster management agencies (NDMA, Red Cross) to clean incoming field data and prioritise response.

## Tasks here.. 
| Difficulty | Required Steps |
|------------|----------------|
| Easy       | Impute missing (disaster_type, magnitude), correct data types |
| Medium     | Easy + validate coordinates, normalize dates |
| Hard       | Medium + remove duplicates, predict severity |

## Action Space
- `impute_missing:<column>`
- `remove_duplicates`
- `correct_dtypes`
- `validate_coordinates`
- `normalize_dates`
- `predict_severity`

## Observation Space
Returns missing_counts, duplicate_count, geo_invalid_count, date_invalid_count, prediction_made, cleaning_steps_completed.

## to Setup this.. do
```bash
pip install -r requirements.txt
python inference.py
