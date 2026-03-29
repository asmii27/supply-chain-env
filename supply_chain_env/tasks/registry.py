"""
Task registry — defines easy / medium / hard scenarios.

Each task is a dict consumed by SupplyChainEnv.
"""

from __future__ import annotations
from ..env.models import SKU, WarehouseStock, DemandForecast

# ── Shared SKU catalogue ──────────────────────────────────────────────────────

_WIDGET_A = SKU(
    sku_id="SKU-A",
    name="Widget Alpha",
    unit_cost=10.0,
    unit_price=25.0,
    lead_time_days=3,
    reorder_point=20,
    reorder_qty=50,
)

_WIDGET_B = SKU(
    sku_id="SKU-B",
    name="Gadget Beta",
    unit_cost=20.0,
    unit_price=55.0,
    lead_time_days=5,
    reorder_point=10,
    reorder_qty=30,
)

_WIDGET_C = SKU(
    sku_id="SKU-C",
    name="Gizmo Gamma",
    unit_cost=5.0,
    unit_price=12.0,
    lead_time_days=2,
    reorder_point=30,
    reorder_qty=100,
)

_WIDGET_D = SKU(
    sku_id="SKU-D",
    name="Device Delta",
    unit_cost=50.0,
    unit_price=130.0,
    lead_time_days=7,
    reorder_point=5,
    reorder_qty=20,
)

# ── TASK 1 — Easy: single warehouse, one SKU running low ─────────────────────

TASK_EASY = {
    "task_description": (
        "You manage a single warehouse stocking Widget Alpha (SKU-A). "
        "Current stock is critically low (8 units). Daily demand is ~6 units. "
        "Your goal: prevent stockouts over the next 30 days while maximising profit."
    ),
    "task_hint": (
        "Restock SKU-A immediately — lead time is 3 days and stock will run out "
        "in ~1-2 days. Consider ordering 60-80 units now to cover demand."
    ),
    "warehouses": ["WH-NORTH"],
    "initial_budget": 2_000.0,
    "skus": [_WIDGET_A],
    "initial_stock": [
        WarehouseStock(warehouse_id="WH-NORTH", sku_id="SKU-A", on_hand=8, on_order=0),
    ],
    "forecasts": [
        DemandForecast(sku_id="SKU-A", warehouse_id="WH-NORTH",
                       forecast_7d=42, forecast_30d=180, confidence=0.85),
    ],
}

# ── TASK 2 — Medium: two warehouses, two SKUs, imbalanced stock ───────────────

TASK_MEDIUM = {
    "task_description": (
        "You manage two warehouses (WH-NORTH, WH-SOUTH) with two SKUs: "
        "Widget Alpha (SKU-A) and Gadget Beta (SKU-B). "
        "WH-NORTH is overstocked on SKU-A; WH-SOUTH is nearly out. "
        "SKU-B is low in both locations and has a 5-day lead time. "
        "Goal: balance stock across warehouses and avoid stockouts for 30 days."
    ),
    "task_hint": (
        "Transfer some SKU-A from WH-NORTH to WH-SOUTH to cover immediate demand. "
        "Restock SKU-B early — 5-day lead time means you'll stock out before it arrives "
        "if you wait too long."
    ),
    "warehouses": ["WH-NORTH", "WH-SOUTH"],
    "initial_budget": 4_000.0,
    "skus": [_WIDGET_A, _WIDGET_B],
    "initial_stock": [
        WarehouseStock(warehouse_id="WH-NORTH", sku_id="SKU-A", on_hand=120, on_order=0),
        WarehouseStock(warehouse_id="WH-SOUTH", sku_id="SKU-A", on_hand=5,   on_order=0),
        WarehouseStock(warehouse_id="WH-NORTH", sku_id="SKU-B", on_hand=12,  on_order=0),
        WarehouseStock(warehouse_id="WH-SOUTH", sku_id="SKU-B", on_hand=8,   on_order=0),
    ],
    "forecasts": [
        DemandForecast(sku_id="SKU-A", warehouse_id="WH-NORTH",
                       forecast_7d=25, forecast_30d=100, confidence=0.80),
        DemandForecast(sku_id="SKU-A", warehouse_id="WH-SOUTH",
                       forecast_7d=35, forecast_30d=150, confidence=0.75),
        DemandForecast(sku_id="SKU-B", warehouse_id="WH-NORTH",
                       forecast_7d=12, forecast_30d=50,  confidence=0.70),
        DemandForecast(sku_id="SKU-B", warehouse_id="WH-SOUTH",
                       forecast_7d=14, forecast_30d=60,  confidence=0.70),
    ],
}

# ── TASK 3 — Hard: three warehouses, four SKUs, tight budget, demand spikes ──

TASK_HARD = {
    "task_description": (
        "You manage three warehouses (WH-NORTH, WH-SOUTH, WH-EAST) with four SKUs "
        "(SKU-A, SKU-B, SKU-C, SKU-D). Budget is tight ($6,000). "
        "Demand will spike mid-month (+40%) due to a promotional event. "
        "SKU-D has a 7-day lead time and high unit value — ordering too much risks "
        "overstock; too little causes costly stockouts. "
        "Goal: maximise 30-day profit across all warehouses while holding costs low."
    ),
    "task_hint": (
        "Prioritise SKU-D restocking immediately due to its long lead time. "
        "Use markdown on SKU-C to clear slow-moving stock and free capital. "
        "Transfer SKU-A from WH-EAST (overstocked) to WH-SOUTH before day 10."
    ),
    "warehouses": ["WH-NORTH", "WH-SOUTH", "WH-EAST"],
    "initial_budget": 6_000.0,
    "skus": [_WIDGET_A, _WIDGET_B, _WIDGET_C, _WIDGET_D],
    "initial_stock": [
        # SKU-A
        WarehouseStock(warehouse_id="WH-NORTH", sku_id="SKU-A", on_hand=30,  on_order=0),
        WarehouseStock(warehouse_id="WH-SOUTH", sku_id="SKU-A", on_hand=6,   on_order=0),
        WarehouseStock(warehouse_id="WH-EAST",  sku_id="SKU-A", on_hand=90,  on_order=0),
        # SKU-B
        WarehouseStock(warehouse_id="WH-NORTH", sku_id="SKU-B", on_hand=15,  on_order=0),
        WarehouseStock(warehouse_id="WH-SOUTH", sku_id="SKU-B", on_hand=20,  on_order=0),
        WarehouseStock(warehouse_id="WH-EAST",  sku_id="SKU-B", on_hand=4,   on_order=0),
        # SKU-C
        WarehouseStock(warehouse_id="WH-NORTH", sku_id="SKU-C", on_hand=200, on_order=0),
        WarehouseStock(warehouse_id="WH-SOUTH", sku_id="SKU-C", on_hand=180, on_order=0),
        WarehouseStock(warehouse_id="WH-EAST",  sku_id="SKU-C", on_hand=50,  on_order=0),
        # SKU-D (low everywhere, long lead time)
        WarehouseStock(warehouse_id="WH-NORTH", sku_id="SKU-D", on_hand=3,   on_order=0),
        WarehouseStock(warehouse_id="WH-SOUTH", sku_id="SKU-D", on_hand=2,   on_order=0),
        WarehouseStock(warehouse_id="WH-EAST",  sku_id="SKU-D", on_hand=4,   on_order=0),
    ],
    "forecasts": [
        DemandForecast(sku_id="SKU-A", warehouse_id="WH-NORTH",
                       forecast_7d=20,  forecast_30d=85,  confidence=0.72),
        DemandForecast(sku_id="SKU-A", warehouse_id="WH-SOUTH",
                       forecast_7d=28,  forecast_30d=120, confidence=0.68),
        DemandForecast(sku_id="SKU-A", warehouse_id="WH-EAST",
                       forecast_7d=10,  forecast_30d=40,  confidence=0.80),
        DemandForecast(sku_id="SKU-B", warehouse_id="WH-NORTH",
                       forecast_7d=14,  forecast_30d=55,  confidence=0.65),
        DemandForecast(sku_id="SKU-B", warehouse_id="WH-SOUTH",
                       forecast_7d=14,  forecast_30d=55,  confidence=0.65),
        DemandForecast(sku_id="SKU-B", warehouse_id="WH-EAST",
                       forecast_7d=10,  forecast_30d=45,  confidence=0.60),
        DemandForecast(sku_id="SKU-C", warehouse_id="WH-NORTH",
                       forecast_7d=30,  forecast_30d=100, confidence=0.78),
        DemandForecast(sku_id="SKU-C", warehouse_id="WH-SOUTH",
                       forecast_7d=30,  forecast_30d=100, confidence=0.78),
        DemandForecast(sku_id="SKU-C", warehouse_id="WH-EAST",
                       forecast_7d=20,  forecast_30d=80,  confidence=0.75),
        DemandForecast(sku_id="SKU-D", warehouse_id="WH-NORTH",
                       forecast_7d=4,   forecast_30d=18,  confidence=0.60),
        DemandForecast(sku_id="SKU-D", warehouse_id="WH-SOUTH",
                       forecast_7d=4,   forecast_30d=18,  confidence=0.60),
        DemandForecast(sku_id="SKU-D", warehouse_id="WH-EAST",
                       forecast_7d=3,   forecast_30d=14,  confidence=0.55),
    ],
}

TASK_REGISTRY: dict = {
    "easy_restock":   TASK_EASY,
    "medium_balance": TASK_MEDIUM,
    "hard_optimize":  TASK_HARD,
}
