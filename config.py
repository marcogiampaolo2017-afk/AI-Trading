# -*- coding: utf-8 -*-
"""config.py
Central configuration module for ReinforcementTrading_Part_1-main.
All paths are relative to the project root (the directory that contains this file).
"""
import os

# Root directory (project folder)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Data directory
DATA_DIR = os.path.join(ROOT_DIR, "data")
# Main historical CSV (master)
MASTER_CSV = os.path.join(DATA_DIR, "EURUSD_H1_201005121400_202605181400.csv")

# Models directory
MODELS_DIR = os.path.join(ROOT_DIR, "models")
# Default model filename (can be overridden at runtime)
DEFAULT_MODEL = os.path.join(MODELS_DIR, "model_eurusd_titany_v13_live.zip")

# Brain memory directory (experience, lstm, etc.)
BRAIN_MEMORY_DIR = os.path.join(ROOT_DIR, "brain_memory")

# Logs directory and main log file
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MAIN_LOG_FILE = os.path.join(LOGS_DIR, "log_horita.txt")

# Misc constants
TIMESTAMP_FORMAT = "%H:%M:%S"

# ── Trading constants shared between training AND live (MUST stay in sync) ──
# If you change these, retrain the model from scratch.
#
# IMPORTANT: only R:R >= 2:1 pairs are allowed.
# The old Cartesian product (SL x TP) created 6 bad combinations (e.g. SL=100
# TP=40 → R:R 0.4:1) that caused the model to blow accounts.  Using explicit
# pairs removes those combinations from the action space entirely so the model
# physically cannot choose a losing-expectancy trade structure.
#
# Math: with 52% win-rate and R:R >= 2:1 →
#   EV = 0.52 × 40 + 0.48 × (-20) = +11.3 pip/trade  →  profitable
SL_TP_PAIRS = [
    # SINGLE PAIR — only (20, 60):
    # - R:R 3:1  → breakeven at 25% win rate
    # - Model achieves ~32% win rate → EV = +5.3 pip/trade → profitable
    # - Previous 3-pair config caused the model to default to (20,40)
    #   which needs 33.3% win rate — just above the model's 31.7% → PF 0.98
    # - Simpler action space (3 actions: HOLD/BUY/SELL) → easier to learn
    (20, 60),    # R:R 3:1  → needs only 25.0% win rate
]
# Derived lists (still needed by live_trading_system action_map builder)
SL_OPTIONS = sorted(set(sl for sl, _ in SL_TP_PAIRS))   # [20, 50, 100]
TP_OPTIONS = sorted(set(tp for _, tp in SL_TP_PAIRS))   # [40, 100, 200]
WIN_SIZE   = 30   # LSTM observation window (H1 candles)

# ── Account parameters for a 100 EUR AvaTrade account ───────────────────────
# Live execution uses 0.01 micro-lots → pip value ≈ $0.10/pip
LIVE_LOT_SIZE_LOTS      = 0.01     # lot volume sent to MT5
# Training environment: proportionally equivalent to 0.01 lots
# (notional = 0.01 × 100,000 = 1,000 USD → usd_per_pip = 0.0001 × 1000 = $0.10)
TRAIN_LOT_UNITS         = 1000.0   # lot_size parameter for ForexTradingEnv
TRAIN_INITIAL_EQUITY_USD = 110.0   # ≈ 100 EUR; realistic starting capital
# Risk cap: if equity drops below this fraction of start, end episode early
MAX_DRAWDOWN_FRACTION   = 0.30     # 30% → 33 USD max loss on 110 USD account
# Allow the model to manually close a position before SL/TP is hit.
# MUST be False: when True the model learns to cut winners short (TP was hit
# only 1% of the time in backtest) while letting losers reach SL → guaranteed
# loss. With False the model only decides WHEN/HOW to open; all exits are
# handled automatically by MT5 SL/TP orders — clean R:R guaranteed.
ALLOW_MANUAL_CLOSE      = False

# Ensure directories exist
for _dir in [DATA_DIR, MODELS_DIR, BRAIN_MEMORY_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Exported symbols for easy import
__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "MASTER_CSV",
    "MODELS_DIR",
    "DEFAULT_MODEL",
    "BRAIN_MEMORY_DIR",
    "LOGS_DIR",
    "MAIN_LOG_FILE",
    "TIMESTAMP_FORMAT",
    # Trading constants
    "SL_TP_PAIRS",
    "SL_OPTIONS",
    "TP_OPTIONS",
    "WIN_SIZE",
    "LIVE_LOT_SIZE_LOTS",
    "TRAIN_LOT_UNITS",
    "TRAIN_INITIAL_EQUITY_USD",
    "MAX_DRAWDOWN_FRACTION",
    "ALLOW_MANUAL_CLOSE",
]
