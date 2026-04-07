import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from environment import DisasterCleanerEnv, DisasterAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TASK_NAME = os.getenv("DISASTER_TASK", "easy")   # easy, medium, or hard
MAX_STEPS = 20

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent cleaning a disaster dataset for India/South Asia.
Dataset columns: event_id, disaster_type, latitude, longitude, magnitude, date, affected_population, damage_cost.

Allowed actions (exact strings):
- impute_missing:<column>   (e.g., impute_missing:disaster_type)
- remove_duplicates
- correct_dtypes
- validate_coordinates
- normalize_dates
- predict_severity

Reward: +0.3 per correct step, -0.3 for wrong/repeated step, +0.4 * accuracy for prediction, +1.0 bonus for full completion.
Output ONLY the action string, nothing else.
""").strip()

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(obs, step: int, history: List[str]) -> str:
    return textwrap.dedent(f"""
Step {step}
Missing counts: {obs.missing_counts}
Duplicates: {obs.duplicate_count}
Invalid coordinates: {obs.geo_invalid_count}
Invalid dates: {obs.date_invalid_count}
Completed steps: {obs.cleaning_steps_completed}
Prediction made: {obs.prediction_made}
Recent history: {history[-4:]}
Choose next action.
""").strip()

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DisasterCleanerEnv(task_name=TASK_NAME)
    rewards = []
    steps_taken = 0
    success = False
    final_score = 0.0

    log_start(task=TASK_NAME, env="disaster-cleaner", model=MODEL_NAME)

    try:
        obs = await env.reset()
        history = []
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            user_prompt = build_user_prompt(obs, step, history)
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=50,
                )
                action_str = (completion.choices[0].message.content or "").strip()
                # Basic validation: if not a valid action, fallback to safe action
                valid_prefixes = ["impute_missing:", "remove_duplicates", "correct_dtypes",
                                  "validate_coordinates", "normalize_dates", "predict_severity"]
                if not any(action_str.startswith(p) for p in valid_prefixes):
                    action_str = "correct_dtypes"
            except Exception as e:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                action_str = "correct_dtypes"

            result = await env.step(DisasterAction(action_string=action_str))
            obs, reward_obj, done, _ = result
            reward = reward_obj.value

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")
            if done:
                break

        # Compute final score using the appropriate grader
        from environment import EasyGrader, MediumGrader, HardGrader
        if TASK_NAME == "easy":
            grader = EasyGrader()
        elif TASK_NAME == "medium":
            grader = MediumGrader()
        else:
            grader = HardGrader()
        final_score = await grader.score(env, [])
        success = final_score >= 0.8

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
