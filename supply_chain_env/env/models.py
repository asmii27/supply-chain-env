"""
Typed models for the Supply Chain Inventory OpenEnv environment.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import uuid


# ── Action space ────────────────────────────────────────────────────────────

class RestockAction(BaseModel):
    """Order more units of a SKU from the supplier."""
    action_type: Literal["restock"] = "restock"
    sku_id: str
    quantity: int = Field(ge=1, description="Units to order")


class TransferAction(BaseModel):
    """Transfer stock between two warehouses."""
    action_type: Literal["transfer"] = "transfer"
    sku_id: str
    from_warehouse: str
    to_warehouse: str
    quantity: int = Field(ge=1)


class MarkdownAction(BaseModel):
    """Apply a price markdown to a SKU to accelerate sales."""
    action_type: Literal["markdown"] = "markdown"
    sku_id: str
    discount_pct: float = Field(ge=0.0, le=0.9, description="Fraction off, e.g. 0.2 = 20%")


class NoOpAction(BaseModel):
    """Do nothing this step."""
    action_type: Literal["noop"] = "noop"


Action = RestockAction | TransferAction | MarkdownAction | NoOpAction


# ── State space ─────────────────────────────────────────────────────────────

class SKU(BaseModel):
    sku_id: str
    name: str
    unit_cost: float       # $ per unit to restock
    unit_price: float      # $ sell price (before markdown)
    lead_time_days: int    # days until restock arrives
    reorder_point: int     # trigger point for restocking
    reorder_qty: int       # suggested order quantity


class WarehouseStock(BaseModel):
    warehouse_id: str
    sku_id: str
    on_hand: int
    on_order: int          # units in transit (not yet arrived)
    days_until_arrival: Optional[int] = None


class DemandForecast(BaseModel):
    sku_id: str
    warehouse_id: str
    forecast_7d: int       # expected demand next 7 days
    forecast_30d: int      # expected demand next 30 days
    confidence: float      # 0.0–1.0


class PendingOrder(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    sku_id: str
    warehouse_id: str
    quantity: int
    arrive_on_day: int     # absolute simulation day


class SupplyChainState(BaseModel):
    """Full observable state returned by state() / reset()."""
    day: int
    budget_remaining: float
    skus: Dict[str, SKU]
    stock: List[WarehouseStock]
    forecasts: List[DemandForecast]
    pending_orders: List[PendingOrder]
    stockout_events: int          # cumulative stockouts this episode
    overstock_cost_total: float   # cumulative holding cost
    revenue_total: float          # cumulative revenue
    task_description: str
    task_hint: str


# ── Step result ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    state: SupplyChainState
    reward: float
    done: bool
    info: Dict
