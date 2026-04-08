from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

from .env import FloodRiskEnv
from .models import Observation, Action, State, Reward
from .tasks import TASKS
from .grader import grade_solution



app = FastAPI()

# Global environment instance
env: Optional[FloodRiskEnv] = None
current_task: str = "hard"

class ResetRequest(BaseModel):
    task: str = "hard"

class StepRequest(BaseModel):
    action: Action

class GradeRequest(BaseModel):
    actions: list

@app.post("/reset")
async def reset(request: ResetRequest):
    global env, current_task
    task = request.task
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}. Choose from {list(TASKS.keys())}")
    current_task = task
    max_steps = TASKS[task]["max_steps"]
    env = FloodRiskEnv(max_steps=max_steps)
    obs = env.reset(task=TASKS[task]["initial_corruption"])
    return {"observation": obs.dict(), "task": task}

@app.post("/step")
async def step(request: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    obs, reward, done, info = env.step(request.action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
async def get_state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return env.state().dict()

@app.post("/grade")
async def grade(request: GradeRequest):
    global env, current_task
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    result = grade_solution(env, request.actions)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}