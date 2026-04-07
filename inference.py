"""
Inference Script — SupplyChainEnv
===================================
MANDATORY env vars:
  API_BASE_URL   The API endpoint for the LLM
  MODEL_NAME     The model identifier
  HF_TOKEN       Your Hugging Face API key

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Mandatory env vars ────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

if not API_KEY:
    print("ERROR: HF_TOKEN or API_KEY environment variable must be set.", flush=True)
    sys.exit(1)

from supply_chain_env.env.environment import SupplyChainEnv
from supply_chain_env.graders.graders import EpisodeLog, grade

# ── Config ────────────────────────────────────────────────────────────────────
MAX_STEPS   = 30
TEMPERATURE = 0.2
MAX_TOKENS  = 300
ALL_TASKS   = ["easy_restock", "medium_balance", "hard_optimize"]
SEED        = 42
BENCHMARK   = "supply-chain-env"
SUCCESS_SCORE_THRESHOLD = 0.3

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Structured logging ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt helpers ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert supply chain manager AI agent.
Each day you observe the warehouse state and choose exactly ONE action.

AVAILABLE ACTIONS (respond with valid JSON only, no markdown):
1. Restock a SKU:
   {"action_type": "restock", "sku_id": "SKU-X", "quantity": <int>}

2. Transfer between warehouses:
   {"action_type": "transfer", "sku_id": "SKU-X", "from_warehouse": "WH-A", "to_warehouse": "WH-B", "quantity": <int>}

3. Apply a price markdown:
   {"action_type": "markdown", "sku_id": "SKU-X", "discount_pct": <0.0-0.9>}

4. Do nothing:
   {"action_type": "noop"}

RULES:
- Never spend more than budget_remaining
- Reorder before stock hits zero — account for lead_time_days
- Transfer costs $0.50/unit
- Markdown boosts demand by 1.5x the discount fraction
- Output ONLY a single JSON object. No explanation, no markdown fences.
"""


def build_user_prompt(state: dict, day: int) -> str:
    stock_lines = []
    for sw in state["stock"]:
        sku_id = sw["sku_id"]
        sku    = state["skus"].get(sku_id, {})
        fc     = next(
            (f for f in state["forecasts"]
             if f["sku_id"] == sku_id and f["warehouse_id"] == sw["warehouse_id"]),
            None
        )
        fc_str = f"forecast_7d={fc['forecast_7d']}" if fc else "no forecast"
        stock_lines.append(
            f"  {sku_id}@{sw['warehouse_id']}: on_hand={sw['on_hand']} "
            f"reorder_point={sku.get('reorder_point','?')} "
            f"lead_time={sku.get('lead_time_days','?')}d "
            f"cost=${sku.get('unit_cost','?')} {fc_str}"
        )

    pending_lines = [
        f"  {p['sku_id']}->{p['warehouse_id']}: qty={p['quantity']} arrives_day={p['arrive_on_day']}"
        for p in state["pending_orders"]
    ]

    return (
        f"DAY {day}/30 | Budget: ${state['budget_remaining']:.2f} | "
        f"Stockouts: {state['stockout_events']} | Revenue: ${state['revenue_total']:.2f}\n"
        f"TASK: {state['task_description']}\n"
        f"HINT: {state['task_hint']}\n"
        f"STOCK:\n" + "\n".join(stock_lines) +
        ("\nPENDING:\n" + "\n".join(pending_lines) if pending_lines else "") +
        "\nChoose your action:"
    )


# ── Action parser ─────────────────────────────────────────────────────────────

def parse_action(raw: str) -> Dict[str, Any]:
    raw   = raw.strip()
    raw   = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if "action_type" in action:
                return action
        except json.JSONDecodeError:
            pass
    return {"action_type": "noop"}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = SEED) -> None:
    env   = SupplyChainEnv(task_id=task_id, seed=seed)
    state = env.reset()

    log = EpisodeLog(task_id=task_id, initial_budget=state.budget_remaining)
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards:      List[float] = []
    conversation: List[Dict]  = []
    steps_taken   = 0
    score         = 0.0
    success       = False

    try:
        for day in range(1, MAX_STEPS + 1):
            state_dict = json.loads(state.model_dump_json())
            user_msg   = build_user_prompt(state_dict, day)

            conversation.append({"role": "user", "content": user_msg})
            if len(conversation) > 8:
                conversation = conversation[-8:]

            # LLM call
            error_msg = None
            try:
                response   = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw_action = response.choices[0].message.content or ""
            except Exception as e:
                raw_action = '{"action_type": "noop"}'
                error_msg  = str(e)[:80]

            action = parse_action(raw_action)
            conversation.append({"role": "assistant", "content": raw_action})

            # Step env
            try:
                result = env.step(action)
            except Exception as e:
                result    = env.step({"action_type": "noop"})
                error_msg = str(e)[:80]

            reward = result.reward
            done   = result.done

            rewards.append(reward)
            log.actions_taken.append({**action, "day": day})
            log.daily_rewards.append(reward)
            log.total_reward += reward
            steps_taken = day

            # Emit [STEP] log
            action_str = json.dumps(action, separators=(',', ':'))
            log_step(step=day, action=action_str, reward=reward, done=done, error=error_msg)

            state = result.state
            if done:
                break

            time.sleep(0.1)

        # Finalise
        log.stockout_events      = state.stockout_events
        log.revenue_total        = state.revenue_total
        log.overstock_cost_total = state.overstock_cost_total
        log.budget_spent         = log.initial_budget - state.budget_remaining
        log.days_completed       = state.day

        grading = grade(log)
        score   = grading["score"]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for task_id in ALL_TASKS:
        run_episode(task_id, seed=SEED)


if __name__ == "__main__":
    main()
