"""
api_server.py — FastAPI REST wrapper around SupplyChainEnv.

Endpoints:
  POST /reset          body: {"task_id": "easy_restock", "seed": 42}
  POST /step           body: {"session_id": "...", "action": {...}}
  GET  /state/{sid}
  POST /grade          body: {"session_id": "..."}
  GET  /tasks          — list available tasks
  GET  /health

Run locally:
    uvicorn api_server:app --reload --port 7860
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supply_chain_env.env.environment import SupplyChainEnv
from supply_chain_env.graders.graders import EpisodeLog, grade
from supply_chain_env.tasks.registry import TASK_REGISTRY

app = FastAPI(
    title="SupplyChainEnv API",
    description="OpenEnv-compatible Supply Chain Inventory environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (replace with Redis for production) ────────────────
_sessions: Dict[str, Dict] = {}


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_restock"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class GradeRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "description": cfg["task_description"],
            "hint":        cfg["task_hint"],
            "warehouses":  cfg["warehouses"],
            "sku_count":   len(cfg["skus"]),
        }
        for task_id, cfg in TASK_REGISTRY.items()
    }


@app.post("/reset")
def reset_env(req: ResetRequest):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'")

    env   = SupplyChainEnv(task_id=req.task_id, seed=req.seed)
    state = env.reset()

    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "env":            env,
        "task_id":        req.task_id,
        "initial_budget": state.budget_remaining,
        "log":            EpisodeLog(task_id=req.task_id, initial_budget=state.budget_remaining),
    }
    return {"session_id": sid, "state": state.model_dump()}


@app.post("/step")
def step_env(req: StepRequest):
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "Session not found. Call /reset first.")

    env: SupplyChainEnv = sess["env"]
    log: EpisodeLog     = sess["log"]

    try:
        result = env.step(req.action)
    except Exception as e:
        raise HTTPException(400, str(e))

    # Update log
    log.actions_taken.append({**req.action, "day": result.state.day})
    log.daily_rewards.append(result.reward)
    log.total_reward += result.reward

    if result.done:
        s = result.state
        log.stockout_events      = s.stockout_events
        log.revenue_total        = s.revenue_total
        log.overstock_cost_total = s.overstock_cost_total
        log.budget_spent         = sess["initial_budget"] - s.budget_remaining
        log.days_completed       = s.day

    return {
        "state":  result.state.model_dump(),
        "reward": result.reward,
        "done":   result.done,
        "info":   result.info,
    }


@app.get("/state/{session_id}")
def get_state(session_id: str):
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(404, "Session not found.")
    return sess["env"].state().model_dump()


@app.post("/grade")
def grade_session(req: GradeRequest):
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "Session not found.")
    log: EpisodeLog = sess["log"]
    if log.days_completed == 0:
        # Force-populate from current state
        s = sess["env"].state()
        log.stockout_events      = s.stockout_events
        log.revenue_total        = s.revenue_total
        log.overstock_cost_total = s.overstock_cost_total
        log.budget_spent         = sess["initial_budget"] - s.budget_remaining
        log.days_completed       = s.day
    return grade(log)
