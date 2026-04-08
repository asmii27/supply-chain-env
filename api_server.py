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
from fastapi.responses import HTMLResponse
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
    task_id: Optional[str] = "easy_restock"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class GradeRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SupplyChainEnv</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0f1117; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; min-height: 100vh; }
    .hero { background: linear-gradient(135deg, #1a1f35 0%, #0f1117 100%); padding: 60px 40px; text-align: center; border-bottom: 1px solid #2a2f45; }
    .hero h1 { font-size: 2.8rem; font-weight: 700; color: #fff; margin-bottom: 12px; }
    .hero h1 span { color: #4f8ef7; }
    .hero p { font-size: 1.1rem; color: #9ca3af; max-width: 600px; margin: 0 auto 30px; }
    .badges { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
    .badge { background: #1e2538; border: 1px solid #2a3550; padding: 8px 18px; border-radius: 20px; font-size: 0.85rem; color: #4f8ef7; text-decoration: none; }
    .badge:hover { background: #4f8ef7; color: #fff; transition: 0.2s; }
    .container { max-width: 1000px; margin: 0 auto; padding: 50px 40px; }
    .section-title { font-size: 1.4rem; font-weight: 600; color: #fff; margin-bottom: 20px; padding-bottom: 8px; border-bottom: 1px solid #2a2f45; }
    .tasks { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 50px; }
    .task-card { background: #1a1f2e; border: 1px solid #2a3550; border-radius: 12px; padding: 24px; }
    .task-card .difficulty { font-size: 0.8rem; font-weight: 600; padding: 3px 10px; border-radius: 10px; display: inline-block; margin-bottom: 12px; }
    .easy { background: #0d3320; color: #34d399; }
    .medium { background: #3a2800; color: #fbbf24; }
    .hard { background: #3a0d0d; color: #f87171; }
    .task-card h3 { font-size: 1rem; color: #fff; margin-bottom: 8px; }
    .task-card p { font-size: 0.875rem; color: #9ca3af; line-height: 1.5; }
    .task-card .score { margin-top: 12px; font-size: 0.8rem; color: #4f8ef7; }
    .endpoints { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 50px; }
    .endpoint { background: #1a1f2e; border: 1px solid #2a3550; border-radius: 10px; padding: 18px; display: flex; align-items: center; gap: 14px; text-decoration: none; color: inherit; }
    .endpoint:hover { border-color: #4f8ef7; transition: 0.2s; }
    .method { font-size: 0.75rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; min-width: 45px; text-align: center; }
    .get { background: #0d3320; color: #34d399; }
    .post { background: #1a2d5a; color: #60a5fa; }
    .endpoint-info h4 { font-size: 0.95rem; color: #fff; font-family: monospace; }
    .endpoint-info p { font-size: 0.8rem; color: #9ca3af; }
    .footer { text-align: center; padding: 30px; border-top: 1px solid #2a2f45; color: #6b7280; font-size: 0.875rem; }
    .footer a { color: #4f8ef7; text-decoration: none; }
  </style>
</head>
<body>
  <div class="hero">
    <h1>🏭 <span>SupplyChain</span>Env</h1>
    <p>A real-world OpenEnv environment where AI agents learn to manage warehouse inventory, prevent stockouts, and maximise profit.</p>
    <div class="badges">
      <a class="badge" href="/docs">📖 API Docs</a>
      <a class="badge" href="/health">❤️ Health</a>
      <a class="badge" href="/tasks">📋 Tasks</a>
      <a class="badge" href="https://github.com/asmii27/supply-chain-env" target="_blank">⭐ GitHub</a>
    </div>
  </div>

  <div class="container">
    <h2 class="section-title">🎯 Tasks</h2>
    <div class="tasks">
      <div class="task-card">
        <span class="difficulty easy">🟢 Easy</span>
        <h3>easy_restock</h3>
        <p>1 warehouse, 1 SKU critically low at 8 units. Daily demand ~6. Prevent stockouts for 30 days.</p>
        <div class="score">Budget: $2,000 · Baseline: 0.61</div>
      </div>
      <div class="task-card">
        <span class="difficulty medium">🟡 Medium</span>
        <h3>medium_balance</h3>
        <p>2 warehouses, 2 SKUs. WH-NORTH overstocked, WH-SOUTH empty. 5-day lead times. Balance inventory.</p>
        <div class="score">Budget: $4,000 · Baseline: 0.48</div>
      </div>
      <div class="task-card">
        <span class="difficulty hard">🔴 Hard</span>
        <h3>hard_optimize</h3>
        <p>3 warehouses, 4 SKUs, tight budget, +40% demand spike mid-month. SKU-D has 7-day lead time.</p>
        <div class="score">Budget: $6,000 · Baseline: 0.37</div>
      </div>
    </div>

    <h2 class="section-title">🔌 API Endpoints</h2>
    <div class="endpoints">
      <a class="endpoint" href="/health">
        <span class="method get">GET</span>
        <div class="endpoint-info"><h4>/health</h4><p>Health check</p></div>
      </a>
      <a class="endpoint" href="/tasks">
        <span class="method get">GET</span>
        <div class="endpoint-info"><h4>/tasks</h4><p>List all tasks</p></div>
      </a>
      <a class="endpoint" href="/docs">
        <span class="method post">POST</span>
        <div class="endpoint-info"><h4>/reset</h4><p>Start new episode</p></div>
      </a>
      <a class="endpoint" href="/docs">
        <span class="method post">POST</span>
        <div class="endpoint-info"><h4>/step</h4><p>Take an action</p></div>
      </a>
      <a class="endpoint" href="/docs">
        <span class="method get">GET</span>
        <div class="endpoint-info"><h4>/state/{session_id}</h4><p>Get current state</p></div>
      </a>
      <a class="endpoint" href="/docs">
        <span class="method post">POST</span>
        <div class="endpoint-info"><h4>/grade</h4><p>Score the episode (0.0–1.0)</p></div>
      </a>
    </div>
  </div>

  <div class="footer">
    Built for <strong>OpenEnv Hackathon</strong> by Meta × Hugging Face × Scaler &nbsp;·&nbsp;
    <a href="https://github.com/asmii27/supply-chain-env">GitHub</a> &nbsp;·&nbsp;
    <a href="/docs">API Docs</a>
  </div>
</body>
</html>
"""


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
def reset_env(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    task_id = req.task_id or "easy_restock"
    seed    = req.seed or 42
    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    env   = SupplyChainEnv(task_id=task_id, seed=seed)
    state = env.reset()
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "env":            env,
        "task_id":        task_id,
        "initial_budget": state.budget_remaining,
        "log":            EpisodeLog(task_id=task_id, initial_budget=state.budget_remaining),
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
