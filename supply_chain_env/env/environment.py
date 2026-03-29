"""
SupplyChainEnv — OpenEnv-compatible supply chain / inventory environment.

API:
    env = SupplyChainEnv(task_id="easy_restock")
    state  = env.reset()
    result = env.step(action_dict)   # action_dict = {"action_type": ..., ...}
    state  = env.state()
"""

from __future__ import annotations
import random
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .models import (
    Action, RestockAction, TransferAction, MarkdownAction, NoOpAction,
    SKU, WarehouseStock, DemandForecast, PendingOrder,
    SupplyChainState, StepResult,
)
from ..tasks.registry import TASK_REGISTRY


# ── Simulation constants ─────────────────────────────────────────────────────

HOLDING_COST_PER_UNIT_PER_DAY = 0.05   # $ per unit per day stored
STOCKOUT_PENALTY              = 15.0   # $ per stockout event
TRANSFER_COST_PER_UNIT        = 0.50   # $ per unit transferred
MAX_DAYS                      = 30     # episode length


class SupplyChainEnv:
    """Real-world supply-chain planning environment."""

    def __init__(self, task_id: str = "easy_restock", seed: Optional[int] = None):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY)}")
        self.task_id   = task_id
        self._task_cfg = TASK_REGISTRY[task_id]
        self._seed     = seed
        self._rng      = random.Random(seed)
        self._state: Optional[SupplyChainState] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> SupplyChainState:
        """Reset to initial state and return it."""
        self._rng = random.Random(self._seed)
        cfg = deepcopy(self._task_cfg)

        self._skus: Dict[str, SKU] = {s.sku_id: s for s in cfg["skus"]}
        self._warehouses: List[str] = cfg["warehouses"]
        self._stock: List[WarehouseStock] = cfg["initial_stock"]
        self._budget = cfg["initial_budget"]
        self._day = 0
        self._pending_orders: List[PendingOrder] = []
        self._stockout_events = 0
        self._overstock_cost_total = 0.0
        self._revenue_total = 0.0
        self._markdown: Dict[str, float] = {}   # sku_id -> discount_pct

        self._state = self._build_state()
        return self._state

    def step(self, action_dict: Dict[str, Any]) -> StepResult:
        """Apply one action, simulate one day, return StepResult."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        action = self._parse_action(action_dict)
        reward = 0.0
        info: Dict[str, Any] = {}

        # 1. Apply action
        action_reward, action_info = self._apply_action(action)
        reward += action_reward
        info.update(action_info)

        # 2. Advance one day: receive pending orders
        self._day += 1
        arrived = self._receive_orders()
        info["arrived_orders"] = arrived

        # 3. Simulate demand
        sales, stockouts, revenue = self._simulate_demand()
        self._stockout_events += stockouts
        self._revenue_total   += revenue
        info["sales"]     = sales
        info["stockouts"] = stockouts
        info["revenue"]   = revenue

        # 4. Compute holding costs
        holding = self._compute_holding_cost()
        self._overstock_cost_total += holding
        info["holding_cost"] = holding

        # 5. Reward shaping
        reward += revenue
        reward -= holding
        reward -= stockouts * STOCKOUT_PENALTY
        reward  = round(reward, 4)

        done = self._day >= MAX_DAYS
        self._state = self._build_state()
        return StepResult(state=self._state, reward=reward, done=done, info=info)

    def state(self) -> SupplyChainState:
        """Return current state without advancing time."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ── Action handlers ───────────────────────────────────────────────────────

    def _parse_action(self, d: Dict) -> Action:
        t = d.get("action_type", "noop")
        if t == "restock":   return RestockAction(**d)
        if t == "transfer":  return TransferAction(**d)
        if t == "markdown":  return MarkdownAction(**d)
        return NoOpAction()

    def _apply_action(self, action: Action):
        reward = 0.0
        info: Dict = {}

        if isinstance(action, RestockAction):
            sku = self._skus.get(action.sku_id)
            if not sku:
                info["error"] = f"Unknown SKU {action.sku_id}"
                return reward, info
            cost = sku.unit_cost * action.quantity
            if cost > self._budget:
                action = NoOpAction()   # can't afford — fall through
                info["error"] = "Insufficient budget"
            else:
                self._budget -= cost
                arrive_day = self._day + sku.lead_time_days
                # Add to first warehouse that carries the SKU, or warehouse[0]
                wh = self._get_sku_primary_warehouse(action.sku_id)
                self._pending_orders.append(PendingOrder(
                    sku_id=action.sku_id,
                    warehouse_id=wh,
                    quantity=action.quantity,
                    arrive_on_day=arrive_day,
                ))
                # Update on_order count
                for sw in self._stock:
                    if sw.sku_id == action.sku_id and sw.warehouse_id == wh:
                        sw.on_order += action.quantity
                        sw.days_until_arrival = sku.lead_time_days
                info["ordered"] = {"sku": action.sku_id, "qty": action.quantity, "cost": cost}

        elif isinstance(action, TransferAction):
            cost = action.quantity * TRANSFER_COST_PER_UNIT
            src = self._get_stock(action.sku_id, action.from_warehouse)
            dst = self._get_stock(action.sku_id, action.to_warehouse)
            if src is None or dst is None:
                info["error"] = "Warehouse or SKU not found"
            elif src.on_hand < action.quantity:
                info["error"] = f"Insufficient stock in {action.from_warehouse}"
            elif cost > self._budget:
                info["error"] = "Insufficient budget for transfer"
            else:
                src.on_hand   -= action.quantity
                dst.on_hand   += action.quantity
                self._budget  -= cost
                reward        -= cost
                info["transferred"] = {"sku": action.sku_id, "qty": action.quantity}

        elif isinstance(action, MarkdownAction):
            self._markdown[action.sku_id] = action.discount_pct
            info["markdown"] = {"sku": action.sku_id, "discount": action.discount_pct}

        return reward, info

    # ── Simulation helpers ────────────────────────────────────────────────────

    def _receive_orders(self) -> List[Dict]:
        arrived = []
        remaining = []
        for order in self._pending_orders:
            if order.arrive_on_day <= self._day:
                sw = self._get_stock(order.sku_id, order.warehouse_id)
                if sw:
                    sw.on_hand  += order.quantity
                    sw.on_order  = max(0, sw.on_order - order.quantity)
                    sw.days_until_arrival = None
                arrived.append({"sku": order.sku_id, "qty": order.quantity})
            else:
                remaining.append(order)
        self._pending_orders = remaining
        return arrived

    def _simulate_demand(self):
        sales: Dict[str, int]    = {}
        stockouts = 0
        revenue   = 0.0

        for sw in self._stock:
            sku  = self._skus[sw.sku_id]
            # Base daily demand from forecast ± 30% noise
            fc   = self._get_forecast(sw.sku_id, sw.warehouse_id)
            base = fc.forecast_7d // 7 if fc else max(1, sku.reorder_qty // 14)
            demand = max(0, int(self._rng.gauss(base, base * 0.3)))

            # Markdown boosts demand
            md     = self._markdown.get(sw.sku_id, 0.0)
            demand = int(demand * (1 + md * 1.5))   # 20% off → +30% demand

            sold   = min(demand, sw.on_hand)
            unsatisfied = demand - sold
            sw.on_hand -= sold

            price   = sku.unit_price * (1 - md)
            revenue += sold * price
            stockouts += 1 if unsatisfied > 0 else 0
            key = f"{sw.sku_id}@{sw.warehouse_id}"
            sales[key] = sold

        return sales, stockouts, revenue

    def _compute_holding_cost(self) -> float:
        total = 0.0
        for sw in self._stock:
            total += sw.on_hand * HOLDING_COST_PER_UNIT_PER_DAY
        return round(total, 4)

    # ── State builder ─────────────────────────────────────────────────────────

    def _build_state(self) -> SupplyChainState:
        cfg = self._task_cfg
        return SupplyChainState(
            day=self._day,
            budget_remaining=round(self._budget, 2),
            skus=self._skus,
            stock=deepcopy(self._stock),
            forecasts=cfg.get("forecasts", []),
            pending_orders=deepcopy(self._pending_orders),
            stockout_events=self._stockout_events,
            overstock_cost_total=round(self._overstock_cost_total, 2),
            revenue_total=round(self._revenue_total, 2),
            task_description=cfg["task_description"],
            task_hint=cfg["task_hint"],
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _get_stock(self, sku_id: str, warehouse_id: str) -> Optional[WarehouseStock]:
        for sw in self._stock:
            if sw.sku_id == sku_id and sw.warehouse_id == warehouse_id:
                return sw
        return None

    def _get_forecast(self, sku_id: str, warehouse_id: str) -> Optional[DemandForecast]:
        for f in self._task_cfg.get("forecasts", []):
            if f.sku_id == sku_id and f.warehouse_id == warehouse_id:
                return f
        return None

    def _get_sku_primary_warehouse(self, sku_id: str) -> str:
        for sw in self._stock:
            if sw.sku_id == sku_id:
                return sw.warehouse_id
        return self._warehouses[0]
