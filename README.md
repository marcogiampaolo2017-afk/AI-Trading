# 🤖 TITANY AI — Reinforcement Learning Forex Trader

> Autonomous EURUSD H1 trading bot powered by **Recurrent PPO (LSTM)** running live on MetaTrader 5.
> Targets **+€1/day** on a €100 AvaTrade demo account with 3:1 R:R and multi-layer risk protection.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-RecurrentPPO-orange.svg?style=for-the-badge)](https://github.com/DLR-RM/stable-baselines3)
[![MetaTrader5](https://img.shields.io/badge/MetaTrader-5-1565c0.svg?style=for-the-badge)](https://www.metatrader5.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen.svg?style=for-the-badge)](tests/)

[Quick Start](#quick-start) · [Architecture](#architecture) · [Live Bot](#live-bot) · [Training](#training) · [Configuration](#configuration) · [Contributing](#contributing)

---

## What You Get

- **Recurrent PPO (LSTM)** model trained on 16 years of EURUSD H1 data (2010–2026)
- **Live GUI terminal** — real-time charts, quantum multiverse, trailing stops, indicator panel
- **Shadow Trainer** — auto-retrains V13 every 30 min from live market experience (no downtime)
- **6-layer protection system**: LAZARUS · VSA · Anti-Parabólica · FOMO · ICT/Vision360 · Meta Diaria
- **H4 Macro Override** — corrects LSTM mean-reversion bias; forces SELL in H4 bearish regimes
- **Adaptive filter thresholds** — LAZARUS and VSA relax when H4 trend strength > 50 pips
- **Daily meta protection** — caps exposure at 1 recovery trade after hitting daily goal (€1.00)
- **Trailing stop** — auto-locks profit as price moves in favor
- **AvaTrade Demo ready** — MT5 credentials injected via env vars (never hardcoded)

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/marcogiampaolo2017-afk/AI-Trading.git
cd AI-Trading
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r Requirements.txt
```

### 2. Set MT5 Credentials

```bash
# Windows PowerShell
$env:MT5_ACCOUNT_ID = "101717253"
$env:MT5_PASSWORD   = "your_password"
$env:MT5_SERVER     = "Ava-Demo 1-MT5"
```

> **Never** commit credentials. Use env vars or a `.env` file (git-ignored).

### 3. Run the Live Bot

```bash
python core/TITANY_AI_Terminal_Pro.py
```

**Headless mode** (no GUI, for VPS/server):

```bash
python core/TITANY_AI_Terminal_Pro.py --headless
```

### 4. Run Tests

```bash
pytest tests/ -v
# 67 tests — skips automatically if master CSV is missing
```

---

## Architecture

```
AI-Trading/
├── config.py                        ← Single source of truth (SL/TP, WIN_SIZE, lot sizes)
├── core/
│   ├── TITANY_AI_Terminal_Pro.py    ← Live GUI terminal + trading logic (main bot)
│   ├── titany_continuous_trainer.py ← Shadow Trainer (retrains V13 from live experience)
│   ├── titany_multiverse.py         ← Quantum Multiverse Monte Carlo visualizer
│   ├── live_trading_system.py       ← Secondary headless trading engine
│   └── api.py                       ← Base44 sync engine (mobile dashboard)
├── env/
│   ├── trading_env.py               ← Gymnasium-compatible Forex env (train/backtest)
│   └── indicators.py                ← 26+ features: FER, VAM, Z-Score, VSA, ICT, HMM…
├── research/
│   ├── train_agent.py               ← 3-phase training script (3M steps, SubprocVecEnv)
│   └── test_agent.py                ← Backtesting & evaluation script
├── models/
│   ├── model_eurusd_titany_v13_live.zip  ← Active live model (auto-updated by Shadow Trainer)
│   └── vec_normalize.pkl                 ← Observation normalizer (must match model)
├── brain_memory/                    ← Runtime state (git-ignored)
│   ├── synaptic_memory.json         ← Synaptic weight & last trade timestamp
│   ├── market_experience.csv        ← Live experience collected for Shadow Trainer
│   └── lstm_memory_v12.pkl          ← LSTM hidden states (warm-up cache)
├── tests/                           ← 67 pytest unit/integration tests
│   ├── test_trading_env.py
│   ├── test_indicators.py
│   └── test_integration.py
└── data/                            ← Historical CSVs (git-ignored, distribute separately)
```

### Filter Pipeline (per H1 candle)

```
Model predicts action (HOLD / SELL / BUY)
    ↓ H4 Macro Override    — forces direction to match H4 EMA50/EMA200 gap when > 50p
    ↓ LAZARUS (FER)        — blocks if H1 fractal energy < threshold (adaptive to H4 strength)
    ↓ VSA Filter           — checks normalized volume/spread ratio (adaptive to H4 strength)
    ↓ Anti-Parabólica      — blocks BUY on bearish hammer / SELL on bullish pinbar
    ↓ FOMO Dynamic         — blocks if price already > N pips from MA50 (adaptive to H4)
    ↓ ICT / Vision 360     — blocks against FVG zones, liquidity sweeps, engulfing candles
    ↓ Impulse Shield       — blocks counter-trend Z10 score entries
    ↓ Anti-Cohete          — blocks buying extreme dips / selling extreme peaks
    ↓ Multiverso           — Monte Carlo: blocks if simulated SL probability > 55%
    ↓ Meta Diaria          — blocks after daily goal (€1.00); allows 1 TENDENCIA trade
    ↓ EXECUTE via MT5
```

---

## Live Bot

### Key Settings (`config.py`)

| Constant | Value | Description |
|---|---|---|
| `SL_TP_PAIRS` | `[(20, 60)]` | 20-pip SL / 60-pip TP → 3:1 R:R |
| `WIN_SIZE` | `30` | LSTM observation window (H1 candles) |
| `ALLOW_MANUAL_CLOSE` | `False` | All exits via SL/TP only |
| `META_DIARIA` | `€1.00` | Daily profit goal |
| `LIVE_LOT_SIZE_LOTS` | `0.01` | Base lot size |
| `MAX_DRAWDOWN_FRACTION` | `0.30` | Circuit breaker at 30% equity loss |

### Protection Layers

| Layer | Trigger | Action |
|---|---|---|
| **LAZARUS** | FER < threshold (0.04–0.28, adaptive) | Block entry — H1 too chaotic |
| **VSA** | Volume ratio < threshold (adaptive) | Block entry — no institutional signal |
| **Anti-Parabólica** | BUY on bearish candle / SELL on bullish | Block counter-structure entry |
| **Meta Diaria** | daily_profit ≥ €1.00 | Stop after goal; allow 1 TENDENCIA extra |
| **TENDENCIA limit** | 1 extra trade per day after META | Prevents SL erasing all daily profit |
| **Catastrophe guard** | equity_delta < −€1.00 from session start | Hard block for rest of day |

### Shadow Trainer

Runs alongside the live bot. Every 30 minutes it:
1. Reads `brain_memory/market_experience.csv` (live predictions + market state)
2. Trains V13 model in a teacher-mode Gym env (rewards align with LAZARUS/VSA/ICT rules)
3. Saves updated model → hot-injected into live terminal without restart
4. LSTM context **preserved** on injection (no warm-up disruption)

---

## Training

### From Scratch (3-Phase, 3M Steps)

```bash
# Standard (with matplotlib progress window)
python research/train_agent.py

# Headless (server / VPS — no display required)
python research/train_agent.py --headless
```

Training auto-resumes from checkpoints. Phase detection based on `model.num_timesteps`.

| Phase | Steps | Focus |
|---|---|---|
| 1 | 0 – 500k | Basic entry/exit learning |
| 2 | 500k – 1.5M | R:R discipline + filter internalization (teacher mode) |
| 3 | 1.5M – 3M | Live-like conditions, HMM regime, full indicator stack |

> After training, copy `models/model_eurusd_titany_vXX.zip` and `models/vec_normalize.pkl` to the live machine.
> ⚠️ `vec_normalize.pkl` **must** match the model — if you add features, both need regenerating.

---

## Configuration

### Environment Variables (required for live trading)

```bash
MT5_ACCOUNT_ID=101717253        # MT5 account number
MT5_PASSWORD=your_password      # MT5 password
MT5_SERVER=Ava-Demo 1-MT5       # MT5 server name
```

### Changing SL/TP

Edit `config.py`:

```python
SL_TP_PAIRS = [(20, 60)]   # Only this pair — 3:1 R:R, breakeven at 25% win rate
```

> Changing SL/TP requires **full retraining** — the model's action space changes.

### Graceful Shutdown

```bash
# Create this file to trigger emergency model save + clean shutdown:
New-Item stop_training.flag -ItemType File
```

---

## Critical Conventions

- **All constants in `config.py`** — never hardcode values elsewhere
- **Comments/docstrings in Spanish** — consistent with the existing codebase
- **No data leakage** — never use future candles in indicator calculations
- **`regime_proxy`** must always be in `feature_cols` (train/live consistency)
- **`ALLOW_MANUAL_CLOSE = False`** — model only opens; SL/TP handle all exits

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feat/your-feature`
2. Run tests before submitting: `pytest tests/ -v`
3. Open a Pull Request — use the labels `bug`, `enhancement`, or `filter` accordingly
4. Reference the relevant issue number in your PR description

See open [Issues](https://github.com/marcogiampaolo2017-afk/AI-Trading/issues) for tasks to pick up.

---

## Project Links

- [Report a bug or request a feature](https://github.com/marcogiampaolo2017-afk/AI-Trading/issues)
- [Open Pull Requests](https://github.com/marcogiampaolo2017-afk/AI-Trading/pulls)
- [Live session logs](testos_para_entender/) *(local only)*

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Powered by [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) · [sb3-contrib RecurrentPPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) · [MetaTrader5](https://www.metatrader5.com/) · [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)*
