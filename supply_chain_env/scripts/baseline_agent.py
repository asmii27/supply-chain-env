#!/usr/bin/env python3
"""
baseline_agent.py — Reproducible baseline for SupplyChainEnv.

Runs a simple rule-based agent on all three tasks and prints scores.
Usage:
    python baseline_agent.py
    python baseline_agent.py --task easy_restock --seed 42
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from supply_chain_env.env.environment import SupplyChainEnv
from supply_chain_env.graders.graders import EpisodeLog, grade


# ── Rule-based baseline agent ─────────────────────────────────────────────────

def rule_based_agent(state_dict: dict, task_id: str, day: int) -> dict:
    """
    Simple heuristic agent:
      1. If any SKU is below its reorder_point → restock.
      2. If budget allows and large imbalance → transfer.
      3. If SKU has high on_hand vs forecast → apply markdown.
      4. Otherwise → noop.
    """
    stock     = state_dict["stock"]
    skus      = state_dict["skus"]
    forecasts = state_dict.get("forecasts", [])
    budget    = state_dict["budget_remaining"]

    # Build forecast lookup
    fc_map = {}
    for f in forecasts:
        fc_map[(f["sku_id"], f["warehouse_id"])] = f["forecast_7d"]

    # Priority 1: restock anything critically low
    for sw in stock:
        sku_id = sw["sku_id"]
        wh_id  = sw["warehouse_id"]
        sku    = skus[sku_id]
        on_hand = sw["on_hand"]
        reorder_point = sku["reorder_point"]
        reorder_qty   = sku["reorder_qty"]
        unit_cost     = sku["unit_cost"]
        cost = unit_cost * reorder_qty

        if on_hand <= reorder_point and cost <= budget:
            return {
                "action_type": "restock",
                "sku_id": sku_id,
                "quantity": reorder_qty,
            }

    # Priority 2: transfer from heavily overstocked warehouse to low one
    # Group on_hand by sku
    sku_wh = {}
    for sw in stock:
        sku_wh.setdefault(sw["sku_id"], {})[sw["warehouse_id"]] = sw["on_hand"]

    warehouses_list = list({sw["warehouse_id"] for sw in stock})
    if len(warehouses_list) >= 2:
        for sku_id, wh_stock in sku_wh.items():
            items = sorted(wh_stock.items(), key=lambda x: x[1])
            if len(items) >= 2:
                low_wh, low_qty   = items[0]
                high_wh, high_qty = items[-1]
                transfer_qty = (high_qty - low_qty) // 2
                transfer_cost = transfer_qty * 0.50
                if transfer_qty >= 5 and high_qty > 40 and transfer_cost <= budget:
                    return {
                        "action_type": "transfer",
                        "sku_id": sku_id,
                        "from_warehouse": high_wh,
                        "to_warehouse": low_wh,
                        "quantity": transfer_qty,
                    }

    # Priority 3: markdown slow-moving SKUs after day 5
    if day > 5:
        for sw in stock:
            sku_id   = sw["sku_id"]
            wh_id    = sw["warehouse_id"]
            on_hand  = sw["on_hand"]
            fc_7d    = fc_map.get((sku_id, wh_id), 0)
            # If we have more than 4× weekly demand on hand → mark down 15%
            if fc_7d > 0 and on_hand > fc_7d * 4:
                return {
                    "action_type": "markdown",
                    "sku_id": sku_id,
                    "discount_pct": 0.15,
                }

    return {"action_type": "noop"}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = 0, verbose: bool = False) -> dict:
    env   = SupplyChainEnv(task_id=task_id, seed=seed)
    state = env.reset()

    log = EpisodeLog(
        task_id=task_id,
        initial_budget=state.budget_remaining,
    )

    for day in range(30):
        state_dict = json.loads(state.model_dump_json())
        action     = rule_based_agent(state_dict, task_id, day)

        result = env.step(action)

        # Record for grader
        log.actions_taken.append({**action, "day": day})
        log.daily_rewards.append(result.reward)
        log.total_reward += result.reward

        if verbose:
            print(f"  Day {day+1:2d} | action={action['action_type']:8s} | "
                  f"reward={result.reward:8.2f} | "
                  f"budget={result.state.budget_remaining:8.2f} | "
                  f"stockouts={result.state.stockout_events}")

        state = result.state
        if result.done:
            break

    # Populate final log fields from terminal state
    log.stockout_events        = state.stockout_events
    log.revenue_total          = state.revenue_total
    log.overstock_cost_total   = state.overstock_cost_total
    log.budget_spent           = log.initial_budget - state.budget_remaining
    log.days_completed         = state.day
    log.final_stock            = {
        f"{sw.sku_id}@{sw.warehouse_id}": sw.on_hand for sw in state.stock
    }

    grading = grade(log)
    return {
        "task_id":       task_id,
        "seed":          seed,
        "total_reward":  round(log.total_reward, 4),
        "score":         grading["score"],
        "breakdown":     grading["breakdown"],
        "stats":         grading["stats"],
    }


# ── CLI entry ─────────────────────────────────────────────────────────────────

ALL_TASKS = ["easy_restock", "medium_balance", "hard_optimize"]


def main():
    parser = argparse.ArgumentParser(description="SupplyChainEnv Baseline Agent")
    parser.add_argument("--task", choices=ALL_TASKS + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tasks = ALL_TASKS if args.task == "all" else [args.task]

    print("=" * 62)
    print("  SupplyChainEnv — Rule-Based Baseline Agent Results")
    print("=" * 62)

    results = []
    for task_id in tasks:
        print(f"\n▶  Task: {task_id}")
        if args.verbose:
            print()
        r = run_episode(task_id, seed=args.seed, verbose=args.verbose)
        results.append(r)
        print(f"   Score       : {r['score']:.4f} / 1.0000")
        print(f"   Total reward: {r['total_reward']:.2f}")
        print(f"   Breakdown   : {r['breakdown']}")
        print(f"   Stats       : {r['stats']}")

    print("\n" + "=" * 62)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  Average score across {len(results)} task(s): {avg:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
