TASKS = {
    "easy": {
        "max_steps": 5,
        "description": "Fix basic issues: missing values and inconsistent state casing.",
        "initial_corruption": "easy"
    },
    "medium": {
        "max_steps": 8,
        "description": "Clean mixed units, fill missing values, normalize state names, remove duplicates.",
        "initial_corruption": "medium"
    },
    "hard": {
        "max_steps": 12,
        "description": "Full pipeline: clean all issues, detect outliers, recompute flood risk accurately.",
        "initial_corruption": "hard"
    }
}
