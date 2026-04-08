from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

from .env import FloodRiskEnv
from .models import Observation, Action, State, Reward
from .tasks import TASKS
from .grader import grade_solution

app = FastAPI(title="Disaster Data Prep & Flood Risk", version="1.0.0")

# Global environment instance
env: Optional[FloodRiskEnv] = None
current_task: str = "hard"

class ResetRequest(BaseModel):
    task: str = "hard"   # default to hard if not provided

class StepRequest(BaseModel):
    action: Action

class GradeRequest(BaseModel):
    actions: list

@app.get("/")
async def root():
    """Redirect to the interactive API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "disaster-flood-risk"}

@app.post("/reset")
async def reset(request: ResetRequest = None):
    """Reset the environment with a task (easy, medium, hard)."""
    global env, current_task
    
    # If no body is sent, default to hard
    if request is None:
        task = "hard"
    else:
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
    """Apply an action and get observation, reward, and done flag."""
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
    """Get the full current dataset and metrics."""
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return env.state().dict()

@app.post("/grade")
async def grade(request: GradeRequest):
    """Evaluate a sequence of actions and return a final score."""
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    result = grade_solution(env, request.actions)
    return result

# Exception handler for 422 to provide clearer error message
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request body. Check /docs for expected format."},
    )
