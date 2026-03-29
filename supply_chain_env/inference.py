"""
Inference Script — SupplyChainEnv
===================================
MANDATORY env vars:
  API_BASE_URL   The API endpoint for the LLM  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier           (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN       Your Hugging Face API key

Usage:
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py
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
    print("ERROR: HF_TOKEN or API_KEY environment variable must be set.")
    sys.exit(1)

# ── Env import ────────────────────────────────────────────────────────────────
from supply_chain_env.env.environment import SupplyChainEnv
from supply_chain_env.graders.graders import EpisodeLog, grade

# ── Config ────────────────────────────────────────────────────────────────────
MAX_STEPS   = 30          # one per day
TEMPERATURE = 0.2
MAX_TOKENS  = 300
ALL_TASKS   = ["easy_restock", "medium_balance", "hard_optimize"]
SEED        = 42

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


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
- Transfer costs $0.50/unit; only transfer if source has enough on_hand
- Markdown boosts demand by 1.5x the discount fraction
- Output ONLY a single JSON object. No explanation, no markdown fences.
"""


def build_user_prompt(state: dict, day: int) -> str:
    """Summarise the state into a concise prompt."""
    stock_lines = []
    for sw in state["stock"]:
        fc_match = next(
            (f for f in state["forecasts"]
             if f["sku_id"] == sw["sku_id"] and f["warehouse_id"] == sw["warehouse_id"]),
            None
        )
        fc_str = f"forecast_7d={fc_match['forecast_7d']}" if fc_match else "no forecast"
        sku    = state["skus"].get(sw["sku_id"], {})
        stock_lines.append(
            f"  {sw['sku_id']} @ {sw['warehouse_id']}: "
            f"on_hand={sw['on_hand']}, on_order={sw['on_order']}, "
            f"reorder_point={sku.get('reorder_point','?')}, "
            f"lead_time={sku.get('lead_time_days','?')}d, "
            f"cost=${sku.get('unit_cost','?')}/unit, "
            f"{fc_str}"
        )

    pending_lines = []
    for p in state["pending_orders"]:
        pending_lines.append(
            f"  {p['sku_id']} -> {p['warehouse_id']}: "
            f"qty={p['quantity']}, arrives day {p['arrive_on_day']}"
        )

    return f"""DAY {day} / 30
Budget remaining: ${state['budget_remaining']:.2f}
Cumulative stockouts: {state['stockout_events']}
Cumulative revenue: ${state['revenue_total']:.2f}

TASK: {state['task_description']}
HINT: {state['task_hint']}

STOCK LEVELS:
{chr(10).join(stock_lines) or '  (none)'}

PENDING ORDERS:
{chr(10).join(pending_lines) or '  (none)'}

Choose your action for today:"""


# ── Action parser ─────────────────────────────────────────────────────────────

def parse_action(raw: str) -> Dict[str, Any]:
    """Extract JSON action from LLM response, fallback to noop."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find first {...} block
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if "action_type" in action:
                return action
        except json.JSONDecodeError:
            pass
    return {"action_type": "noop"}


# ── Single episode runner ─────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = SEED, verbose: bool = True) -> Dict:
    env   = SupplyChainEnv(task_id=task_id, seed=seed)
    state = env.reset()

    log = EpisodeLog(
        task_id=task_id,
        initial_budget=state.budget_remaining,
    )

    conversation: List[Dict] = []   # rolling message history (last 4 turns)

    for day in range(1, MAX_STEPS + 1):
        state_dict = json.loads(state.model_dump_json())
        user_msg   = build_user_prompt(state_dict, day)

        # Keep a short rolling history to stay within token budget
        conversation.append({"role": "user", "content": user_msg})
        if len(conversation) > 8:
            conversation = conversation[-8:]

        # LLM call
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_action = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [LLM error day {day}]: {e}")
            raw_action = '{"action_type": "noop"}'

        action = parse_action(raw_action)
        conversation.append({"role": "assistant", "content": raw_action})

        # Step
        try:
            result = env.step(action)
        except Exception as e:
            print(f"  [env error day {day}]: {e}")
            result = env.step({"action_type": "noop"})

        log.actions_taken.append({**action, "day": day})
        log.daily_rewards.append(result.reward)
        log.total_reward += result.reward

        if verbose:
            print(f"  Day {day:2d} | {action['action_type']:8s} | "
                  f"reward={result.reward:8.2f} | "
                  f"budget=${result.state.budget_remaining:.2f} | "
                  f"stockouts={result.state.stockout_events}")

        state = result.state
        if result.done:
            break

        time.sleep(0.1)   # gentle rate limiting

    # Finalise log
    log.stockout_events      = state.stockout_events
    log.revenue_total        = state.revenue_total
    log.overstock_cost_total = state.overstock_cost_total
    log.budget_spent         = log.initial_budget - state.budget_remaining
    log.days_completed       = state.day

    grading = grade(log)
    return {
        "task_id":      task_id,
        "seed":         seed,
        "model":        MODEL_NAME,
        "total_reward": round(log.total_reward, 4),
        "score":        grading["score"],
        "breakdown":    grading["breakdown"],
        "stats":        grading["stats"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SupplyChainEnv — LLM Inference Script")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  API base   : {API_BASE_URL}")
    print("=" * 65)

    results = []
    total_start = time.time()

    for task_id in ALL_TASKS:
        print(f"\n▶  Task: {task_id}")
        t0 = time.time()
        r  = run_episode(task_id, seed=SEED, verbose=True)
        elapsed = time.time() - t0
        results.append(r)

        print(f"\n   ✅ Score     : {r['score']:.4f} / 1.0000")
        print(f"   Reward      : {r['total_reward']:.2f}")
        print(f"   Breakdown   : {r['breakdown']}")
        print(f"   Stats       : {r['stats']}")
        print(f"   Time        : {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    avg_score = sum(r["score"] for r in results) / len(results)

    print("\n" + "=" * 65)
    print(f"  FINAL AVERAGE SCORE : {avg_score:.4f}")
    print(f"  Total runtime       : {total_elapsed:.1f}s")
    print("=" * 65)

    # Write results to file for reproducibility
    with open("inference_results.json", "w") as f:
        json.dump({
            "model":      MODEL_NAME,
            "seed":       SEED,
            "results":    results,
            "avg_score":  avg_score,
            "runtime_s":  round(total_elapsed, 1),
        }, f, indent=2)
    print("\n  Results saved to inference_results.json")


if __name__ == "__main__":
    main()
