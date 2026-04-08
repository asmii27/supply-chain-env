# 🏭 SupplyChainEnv — OpenEnv

> A real-world OpenEnv environment where AI agents learn to manage warehouse inventory, prevent stockouts, and maximise profit over a 30-day planning horizon.

🔗 **Live API:** https://asmii27-supply-chain-env.hf.space  
📖 **Interactive Docs:** https://asmii27-supply-chain-env.hf.space/docs  
🤗 **HF Space:** https://huggingface.co/spaces/asmii27/supply-chain-env

---

## 🌍 Why Supply Chain?

Inventory management is one of the most economically impactful real-world planning problems. Businesses lose billions annually to stockouts, overstock, and poor timing. This environment gives AI agents a faithful simulation of those exact tradeoffs — the same decisions a supply chain manager makes every day.

---

## 🎮 How It Works

Every day the AI agent observes the warehouse state and picks one action:

| Action | Description |
|--------|-------------|
| `restock` | Order units from supplier (arrives after lead_time_days) |
| `transfer` | Move stock between warehouses ($0.50/unit) |
| `markdown` | Discount a product to boost demand by 1.5x |
| `noop` | Do nothing this day |

**Reward per step:**
```
reward = revenue - holding_cost($0.05/unit/day) - stockout_penalty($15) - transfer_cost
```

---

## 🎯 3 Tasks (Easy → Medium → Hard)

### 🟢 Easy — `easy_restock`
- **Setup:** 1 warehouse, 1 SKU critically low (8 units), daily demand ~6
- **Goal:** Prevent stockouts for 30 days
- **Budget:** $2,000 | **Baseline score:** 0.61

### 🟡 Medium — `medium_balance`
- **Setup:** 2 warehouses, 2 SKUs, WH-NORTH overstocked, WH-SOUTH empty, 5-day lead time
- **Goal:** Balance inventory across locations, avoid stockouts
- **Budget:** $4,000 | **Baseline score:** 0.48

### 🔴 Hard — `hard_optimize`
- **Setup:** 3 warehouses, 4 SKUs, tight budget, +40% demand spike mid-month, SKU-D has 7-day lead time
- **Goal:** Maximise 30-day profit using all action types strategically
- **Budget:** $6,000 | **Baseline score:** 0.37

---

## 👁️ Observation Space
```json
{
  "day": 5,
  "budget_remaining": 1450.0,
  "skus": { "SKU-A": { "unit_cost": 10, "unit_price": 25, "lead_time_days": 3 } },
  "stock": [{ "warehouse_id": "WH-NORTH", "sku_id": "SKU-A", "on_hand": 12 }],
  "forecasts": [{ "forecast_7d": 42, "forecast_30d": 180, "confidence": 0.85 }],
  "pending_orders": [{ "arrive_on_day": 8, "quantity": 50 }],
  "stockout_events": 1,
  "revenue_total": 625.0,
  "task_description": "...",
  "task_hint": "..."
}
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | List all 3 tasks |
| POST | `/reset` | Start new episode `{"task_id": "easy_restock", "seed": 42}` |
| POST | `/step` | Take action `{"session_id": "...", "action": {...}}` |
| GET | `/state/{session_id}` | Get current state |
| POST | `/grade` | Score the episode (returns 0.0 - 1.0) |

**Try it live:** https://asmii27-supply-chain-env.hf.space/docs

---

## 🚀 Quick Start
```bash
git clone https://github.com/asmii27/supply-chain-env
cd supply-chain-env
pip install -r requirements.txt
python -m uvicorn api_server:app --port 7860
# Visit http://localhost:7860/docs
```

**Run with Docker:**
```bash
docker build -t supply-chain-env .
docker run -p 7860:7860 supply-chain-env
```

**Run LLM inference:**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token"
python inference.py
```

---

## 📁 Project Structure
```
supply-chain-env/
├── supply_chain_env/
│   ├── env/
│   │   ├── environment.py   # Core env: step() / reset() / state()
│   │   └── models.py        # Typed Pydantic models
│   ├── tasks/
│   │   └── registry.py      # Easy / Medium / Hard task configs
│   └── graders/
│       └── graders.py       # Scoring: 0.0 → 1.0
├── inference.py             # LLM agent using OpenAI client
├── api_server.py            # FastAPI REST server
├── openenv.yaml             # OpenEnv spec
├── Dockerfile               # HF Spaces deployment
└── requirements.txt
```

---

## 📊 Grading Breakdown

| Task | Stockout Avoidance | Profitability | Other |
|------|--------------------|---------------|-------|
| Easy | 40% | 40% | 20% budget efficiency |
| Medium | 35% | 35% | 20% stock balance + 10% action diversity |
| Hard | 30% | 30% | 20% holding cost + 10% markdown + 10% lead time |

---

## 🏆 Built For
# SupplyChainEnv
OpenEnv supply chain inventory management environment.

*Helping AI agents learn real-world supply chain decision making.*
Live API: https://asmii27-supply-chain-env.hf.space
Docs: https://asmii27-supply-chain-env.hf.space/docs
GitHub: https://github.com/asmii27/supply-chain-env
