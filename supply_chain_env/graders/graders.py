"""
Graders — score an agent's episode on a 0.0–1.0 scale.

Each grader receives the full episode log and returns a score + breakdown dict.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any

MAX_DAYS = 30


@dataclass
class EpisodeLog:
    """Accumulator passed to graders after a full episode."""
    task_id: str
    total_reward: float         = 0.0
    stockout_events: int        = 0
    revenue_total: float        = 0.0
    overstock_cost_total: float = 0.0
    budget_spent: float         = 0.0
    initial_budget: float       = 2_000.0
    actions_taken: List[Dict]   = field(default_factory=list)
    daily_rewards: List[float]  = field(default_factory=list)
    final_stock: Dict           = field(default_factory=dict)  # sku_id -> total on_hand
    days_completed: int         = 0


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Easy grader ───────────────────────────────────────────────────────────────

def grade_easy(log: EpisodeLog) -> Dict[str, Any]:
    """
    Easy task: prevent stockouts, basic restocking.

    Scoring:
      40% — stockout avoidance  (0 stockouts = 1.0, linear penalty)
      40% — profitability       (revenue - costs normalised)
      20% — budget efficiency   (don't overspend or waste all budget)
    """
    # Stockout sub-score
    max_tolerable_stockouts = 10
    so_score = _clamp(1.0 - log.stockout_events / max_tolerable_stockouts)

    # Profitability: reference profit for a perfect agent ≈ $1,800
    reference_profit = 1_800.0
    profit = log.revenue_total - log.overstock_cost_total - log.budget_spent
    profit_score = _clamp(profit / reference_profit)

    # Budget efficiency: used some budget but not all (shows purposeful spending)
    used_ratio = log.budget_spent / max(1, log.initial_budget)
    budget_score = _clamp(1.0 - abs(used_ratio - 0.55) * 2)

    score = round(0.40 * so_score + 0.40 * profit_score + 0.20 * budget_score, 4)
    return {
        "score": score,
        "breakdown": {
            "stockout_avoidance": round(so_score, 4),
            "profitability":      round(profit_score, 4),
            "budget_efficiency":  round(budget_score, 4),
        },
        "stats": {
            "stockout_events":  log.stockout_events,
            "revenue_total":    log.revenue_total,
            "profit":           round(profit, 2),
        },
    }


# ── Medium grader ─────────────────────────────────────────────────────────────

def grade_medium(log: EpisodeLog) -> Dict[str, Any]:
    """
    Medium task: balance stock across warehouses, two SKUs.

    Scoring:
      35% — stockout avoidance
      35% — profitability
      20% — stock balance  (did agent use transfers to level inventory?)
      10% — action diversity (used ≥2 distinct action types)
    """
    so_score = _clamp(1.0 - log.stockout_events / 15)

    reference_profit = 3_500.0
    profit = log.revenue_total - log.overstock_cost_total - log.budget_spent
    profit_score = _clamp(profit / reference_profit)

    # Stock balance: reward if final stock per SKU is relatively even across warehouses
    # Proxy: if transfers were used, give partial credit
    transfer_actions = [a for a in log.actions_taken if a.get("action_type") == "transfer"]
    balance_score = _clamp(len(transfer_actions) / 5)  # full credit at 5 transfers

    # Diversity: how many distinct action types were used?
    types_used = {a.get("action_type") for a in log.actions_taken}
    diversity_score = _clamp(len(types_used) / 3)

    score = round(
        0.35 * so_score + 0.35 * profit_score +
        0.20 * balance_score + 0.10 * diversity_score, 4
    )
    return {
        "score": score,
        "breakdown": {
            "stockout_avoidance": round(so_score, 4),
            "profitability":      round(profit_score, 4),
            "stock_balance":      round(balance_score, 4),
            "action_diversity":   round(diversity_score, 4),
        },
        "stats": {
            "stockout_events": log.stockout_events,
            "revenue_total":   log.revenue_total,
            "transfers_used":  len(transfer_actions),
            "action_types":    list(types_used),
        },
    }


# ── Hard grader ───────────────────────────────────────────────────────────────

def grade_hard(log: EpisodeLog) -> Dict[str, Any]:
    """
    Hard task: multi-warehouse, multi-SKU, demand surge, tight budget.

    Scoring:
      30% — stockout avoidance
      30% — net profit
      20% — holding cost control (lower is better)
      10% — markdown strategy  (used at least one markdown action)
      10% — lead-time awareness (ordered SKU-D in first 5 days)
    """
    so_score = _clamp(1.0 - log.stockout_events / 20)

    reference_profit = 8_000.0
    profit = log.revenue_total - log.overstock_cost_total - log.budget_spent
    profit_score = _clamp(profit / reference_profit)

    # Holding cost: penalise if overstock > 20% of revenue
    hc_ratio = log.overstock_cost_total / max(1.0, log.revenue_total)
    holding_score = _clamp(1.0 - hc_ratio * 5)

    # Markdown usage
    md_actions = [a for a in log.actions_taken if a.get("action_type") == "markdown"]
    markdown_score = 1.0 if md_actions else 0.0

    # Lead-time awareness: did agent restock SKU-D early?
    early_skud = [
        a for a in log.actions_taken
        if a.get("action_type") == "restock"
        and a.get("sku_id") == "SKU-D"
        and a.get("day", 99) <= 5
    ]
    lead_time_score = 1.0 if early_skud else 0.0

    score = round(
        0.30 * so_score + 0.30 * profit_score +
        0.20 * holding_score + 0.10 * markdown_score +
        0.10 * lead_time_score, 4
    )
    return {
        "score": score,
        "breakdown": {
            "stockout_avoidance":   round(so_score, 4),
            "net_profit":           round(profit_score, 4),
            "holding_cost_control": round(holding_score, 4),
            "markdown_strategy":    round(markdown_score, 4),
            "lead_time_awareness":  round(lead_time_score, 4),
        },
        "stats": {
            "stockout_events":    log.stockout_events,
            "revenue_total":      log.revenue_total,
            "overstock_cost":     log.overstock_cost_total,
            "profit":             round(profit, 2),
            "markdown_actions":   len(md_actions),
            "skud_early_orders":  len(early_skud),
        },
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

GRADER_REGISTRY = {
    "easy_restock":   grade_easy,
    "medium_balance": grade_medium,
    "hard_optimize":  grade_hard,
}


def grade(log: EpisodeLog) -> Dict[str, Any]:
    """Grade an episode log. Returns score in [0.0, 1.0] + breakdown."""
    grader = GRADER_REGISTRY.get(log.task_id)
    if grader is None:
        raise ValueError(f"No grader for task_id '{log.task_id}'")
    return grader(log)
