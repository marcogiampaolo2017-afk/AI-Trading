# Contributing to TITANY AI

Thank you for your interest. This bot runs real money — be careful with changes.

## Before You Start

Read [README.md](README.md) and [AGENTS.md](AGENTS.md) fully. Understand the filter pipeline and protection conventions before modifying any threshold.

## Development Setup

```bash
git clone https://github.com/marcogiampaolo2017-afk/AI-Trading.git
cd AI-Trading
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r Requirements.txt
pytest tests/ -v                # must pass before any PR
```

## How to Contribute

1. **Open an issue first** — discuss the change before writing code
2. **Fork & branch**: `git checkout -b fix/lazarus-threshold`
3. **Make changes** — follow the conventions below
4. **Run tests**: `pytest tests/ -v` — all 67 must pass
5. **Open a Pull Request** — reference the issue, fill the PR template

## Conventions (Non-Negotiable)

| Rule | Details |
|---|---|
| **All constants in `config.py`** | Never hardcode SL, TP, lot sizes, window sizes |
| **Spanish comments** | All docstrings and inline comments in Spanish |
| **No future data** | Indicators must only use past candles |
| **`regime_proxy` required** | Always in `feature_cols` for train/live consistency |
| **`ALLOW_MANUAL_CLOSE = False`** | Model only opens; all exits via MT5 SL/TP |
| **No model retrain without tests** | Any change to `env/` requires `pytest tests/` |

## Branching Strategy

```
main          ← stable live code
feat/*        ← new features
fix/*         ← bug fixes
filter/*      ← threshold/filter changes
model/*       ← training/architecture changes
docs/*        ← documentation only
```

## Labels

| Label | When to use |
|---|---|
| `bug` | Something is broken in production |
| `enhancement` | New feature or improvement |
| `filter` | Trading filter threshold change |
| `model` | AI/LSTM model or training related |
| `protection` | Risk management / meta diaria |
| `documentation` | README, AGENTS.md, comments |
| `priority: high` | Blocking live trading |
| `priority: medium` | Important but not blocking |
| `priority: low` | Nice to have |
| `good first issue` | Simple, well-scoped task |

## Code Review Criteria

Pull requests are merged only if:
- All tests pass
- Syntax verified: `python -m py_compile core/TITANY_AI_Terminal_Pro.py`
- No runtime data committed (check `.gitignore`)
- No secrets or credentials in code
- TENDENCIA_COUNT logic not broken (max 1 extra trade per day after META)
