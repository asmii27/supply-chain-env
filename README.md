# 🏭 SupplyChainEnv — OpenEnv Supply Chain & Inventory Environment

A real-world OpenEnv environment where an AI agent manages **restocking**, **warehouse transfers**, and **price markdowns** across multiple warehouses and SKUs to maximise profit over a 30-day planning horizon.

---

## Motivation

Inventory management is one of the most economically impactful real-world planning problems. Businesses lose billions annually to stockouts (lost sales) and overstock (holding costs). This environment gives agents a faithful simulation of those tradeoffs: lead times, demand uncertainty, budget constraints, and multi-location balancing — exactly the decisions a supply chain manager makes every day.

---

## Setup & Usage

### Prerequisites

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/supply-chain-env
cd supply-chain-env
pip install -r requirements.txt
```

### Start the API server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 7860
# or via Docker:
docker build -t supply-chain-env .
docker run -p 7860:7860 supply-chain-env
```

### Run the LLM inference script

Set the required environment variables first:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

Results are saved to `inference_results.json`.

### Run the rule-based baseline

```bash
python scripts/baseline_agent.py
python scripts/baseline_agent.py --task hard_optimize --verbose
```

---

## Action Space

| Action | Description | Fields |
|--------|-------------|--------|
| `restock` | Order units from supplier; arrive after `lead_time_days` | `sku_id`, `quantity` |
| `transfer` | Move stock between warehouses at $0.50/unit | `sku_id`, `from_warehouse`, `to_warehouse`, `quantity` |
| `markdown` | Discount price; demand increases 1.5× discount fraction | `sku_id`, `discount_pct` (0.0–0.9) |
| `noop` | Do nothing this day | — |

**Example (JSON):**
```json
{"action_type": "restock", "sku_id": "SKU-A", "quantity": 60}
{"action_type": "transfer", "sku_id": "SKU-B", "from_warehouse": "WH-NORTH", "to_warehouse": "WH-SOUTH", "quantity": 20}
{"action_type": "markdown", "sku_id": "SKU-C", "discount_pct": 0.20}
{"action_type": "noop"}
```

---

## Observation Space (`SupplyChainState`)

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Current simulation day (0–30) |
| `budget_remaining` | float | USD budget remaining |
| `skus` | dict | SKU catalogue (cost, price, lead time, reorder point) |
| `stock` | list | Per-warehouse on-hand and on-order quantities |
| `forecasts` | list | 7-day and 30-day demand forecasts with confidence |
| `pending_orders` | list | In-transit orders and expected arrival day |
| `stockout_events` | int | Cumulative stockout count |
| `overstock_cost_total` | float | Cumulative holding cost in USD |
| `revenue_total` | float | Cumulative revenue in USD |
| `task_description` | str | Natural language task goal |
| `task_hint` | str | Strategic hint for the agent |

---

## Reward Function

```
reward_per_step = revenue_from_sales
               − holding_cost  ($0.05 per unit per day)
               − stockout_penalty ($15 per stockout event)
               − transfer_cost ($0.50 per unit transferred)
```

Restock costs are deducted from `budget_remaining` separately, creating a realistic budget constraint. Reward provides dense signal every day — not just at episode end.

---

## Tasks

### `easy_restock` — Easy
**Scenario:** 1 warehouse, 1 SKU (Widget Alpha) at 8 units. Daily demand ≈ 6. Budget $2,000.  
**Goal:** Prevent stockouts for 30 days.  
**Grading:** 40% stockout avoidance · 40% profitability · 20% budget efficiency  
**Baseline (rule-based):** 0.61

### `medium_balance` — Medium
**Scenario:** 2 warehouses, 2 SKUs. WH-NORTH massively overstocked on SKU-A; WH-SOUTH nearly empty. SKU-B needs restocking with 5-day lead time. Budget $4,000.  
**Goal:** Balance inventory and avoid stockouts.  
**Grading:** 35% stockout · 35% profit · 20% balance (transfers used) · 10% action diversity  
**Baseline (rule-based):** 0.48

### `hard_optimize` — Hard
**Scenario:** 3 warehouses, 4 SKUs, tight $6,000 budget. Mid-month demand spike (+40%). SKU-D has 7-day lead time and high value — timing matters critically.  
**Goal:** Maximise 30-day profit using all action types strategically.  
**Grading:** 30% stockout · 30% profit · 20% holding cost · 10% markdown used · 10% SKU-D ordered by day 5  
**Baseline (rule-based):** 0.37

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode: `{"task_id": "easy_restock", "seed": 42}` |
| POST | `/step` | Take action: `{"session_id": "...", "action": {...}}` |
| GET | `/state/{session_id}` | Get current state |
| POST | `/grade` | Grade completed episode |
| GET | `/tasks` | List all tasks |
| GET | `/health` | Health check |

---

## Project Structure

```
supply-chain-env/
├── supply_chain_env/          # Python package
│   ├── env/
│   │   ├── environment.py     # SupplyChainEnv: step() / reset() / state()
│   │   └── models.py          # Typed Pydantic models
│   ├── tasks/
│   │   └── registry.py        # Easy / Medium / Hard task configs
│   └── graders/
│       └── graders.py         # Graders → score [0.0, 1.0]
├── scripts/
│   └── baseline_agent.py      # Rule-based baseline
├── inference.py               # LLM inference script (OpenAI client)
├── api_server.py              # FastAPI REST server
├── openenv.yaml               # OpenEnv spec
├── Dockerfile                 # HF Spaces deployment
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `meta-llama/Llama-3.3-70B-Instruct`) |
| `HF_TOKEN` | Hugging Face API key |

Set these as **Secrets** in your HF Space settings before running inference.

---

## Baseline Scores (seed=42)

| Task | Rule-based | LLM (Llama-3.3-70B) |
|------|-----------|----------------------|
| `easy_restock` | 0.61 | TBD |
| `medium_balance` | 0.48 | TBD |
| `hard_optimize` | 0.37 | TBD |

---

## License
MIT
