from .env import FloodRiskEnv
from .models import Action
from typing import List, Dict, Any

def grade_solution(env: FloodRiskEnv, actions: List[Action]) -> Dict[str, Any]:
    """Simulate a sequence of actions and return final score."""
    obs = env.reset()
    total_reward = 0.0
    for action in actions:
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        if done:
            break
    final_state = env.state()
    cleanliness = final_state.metrics["cleanliness"]
    consistency = final_state.metrics["consistency"]
    flood_accuracy = final_state.metrics["flood_accuracy"]
    final_score = (cleanliness + consistency + flood_accuracy) / 3.0
    return {
        "score": final_score,
        "total_reward": total_reward,
        "steps": env.steps_taken,
        "cleanliness": cleanliness,
        "consistency": consistency,
        "flood_accuracy": flood_accuracy,
        "success": final_score >= 0.95
    }
