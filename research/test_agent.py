"""
test_agent.py  —  V12 Sniper Backtest
Evaluates the trained RecurrentPPO model on the out-of-sample test set.
Produces: equity curve, trade log CSV, summary stats, and a PNG report.

Usage:
    python research/test_agent.py              # shows plots interactively
    python research/test_agent.py --headless   # saves to reports/, no GUI
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_RESEARCH_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

if "--headless" in sys.argv:
    matplotlib.use("Agg")

import config
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.indicators import (
    load_and_preprocess_data,
    add_quant_features,
    add_hmm_regime_proxy,
    add_physics_features,
    add_golden_strategy_features,
    add_volume_sniper_features,
)
from env.trading_env import ForexTradingEnv


# ─────────────────────────────────────────────────────────────────────────────
def build_feature_pipeline(csv_path):
    """Replicates the exact same pipeline used in train_agent.py."""
    df, feature_cols = load_and_preprocess_data(csv_path)
    df, q = add_quant_features(df);           feature_cols.extend(q)
    df, r = add_hmm_regime_proxy(df);          feature_cols.extend(r)
    df, p = add_physics_features(df);          feature_cols.extend(p)
    df, g = add_golden_strategy_features(df);  feature_cols.extend(g)
    df, s = add_volume_sniper_features(df);    feature_cols.extend(s)

    # Apply same correlation filter as training
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    _PROTECTED = {"regime_proxy"}
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95) and c not in _PROTECTED]
    feature_cols = [c for c in feature_cols if c not in to_drop]
    return df, feature_cols


def run_backtest(model, vec_env, initial_equity):
    """
    Runs ONE full deterministic episode with LSTM state tracking.
    Returns (equity_curve, closed_trades_list).

    Equity is read from the info dict, NOT via get_attr.
    Reason: DummyVecEnv auto-resets the env immediately after done=True, so
    get_attr("equity_usd") returns the reset value ($110) on the terminal step.
    The info dict is captured BEFORE the auto-reset and always holds the correct
    terminal equity.

    Trades are deduplicated by step index because last_trade_info persists in
    the env between steps (it is only overwritten on OPEN/CLOSE events).
    """
    obs = vec_env.reset()
    lstm_states   = None
    episode_starts = np.ones((1,), dtype=bool)

    equity_curve  = [float(initial_equity)]   # seed with known start
    closed_trades = []
    last_close_step = -1                       # dedup key

    while True:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        step_out = vec_env.step(action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        else:
            obs, _, dones, infos = step_out
            done = bool(dones[0])

        episode_starts = np.array([done])

        # ── equity: always from info (valid even at terminal step) ────────────
        eq = float(infos[0].get("equity_usd", equity_curve[-1]))
        equity_curve.append(eq)

        # ── closed trades: deduplicate by step index ──────────────────────────
        trade = infos[0].get("last_trade_info")
        if (isinstance(trade, dict)
                and trade.get("event") == "CLOSE"
                and trade.get("step", -1) != last_close_step):
            closed_trades.append(trade)
            last_close_step = trade.get("step", -1)

        if done:
            break

    return equity_curve, closed_trades


def print_summary(equity_curve, closed_trades, initial_equity):
    """Prints key performance metrics to stdout."""
    final_equity = equity_curve[-1]
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100

    if not closed_trades:
        print("⚠️  No trades were closed during the test episode.")
        return {}

    trades_df = pd.DataFrame(closed_trades)
    wins   = trades_df[trades_df["net_pips"] > 0]
    losses = trades_df[trades_df["net_pips"] <= 0]

    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_win  = wins["net_pips"].mean()  if len(wins)   > 0 else 0
    avg_loss = losses["net_pips"].mean() if len(losses) > 0 else 0
    profit_factor = (wins["net_pips"].sum() / abs(losses["net_pips"].sum())
                     if len(losses) > 0 and losses["net_pips"].sum() != 0 else float("inf"))

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / peak
    max_dd_pct = dd.max() * 100

    print("\n" + "="*50)
    print("  V12 SNIPER — BACKTEST RESULTS (out-of-sample)")
    print("="*50)
    print(f"  Initial equity : ${initial_equity:.2f}")
    print(f"  Final equity   : ${final_equity:.2f}")
    print(f"  Total return   : {total_return_pct:+.2f}%")
    print(f"  Max drawdown   : {max_dd_pct:.2f}%")
    print(f"  Total trades   : {len(trades_df)}")
    print(f"  Win rate       : {win_rate:.1f}%")
    print(f"  Avg win        : +{avg_win:.1f} pips")
    print(f"  Avg loss       : {avg_loss:.1f} pips")
    print(f"  Profit factor  : {profit_factor:.2f}")
    print("="*50)

    # Targets for 100 EUR account
    target_return_pct = 20.0   # 20 EUR/month on 100 EUR = 20%
    if total_return_pct >= target_return_pct:
        print(f"  [OK] TARGET MET: {total_return_pct:+.1f}% >= +{target_return_pct:.0f}%")
    else:
        gap = target_return_pct - total_return_pct
        print(f"  [NO] TARGET NOT MET: {total_return_pct:+.1f}% (gap: {gap:.1f}%)")

    return {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "max_dd_pct": max_dd_pct,
        "n_trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def main():
    # ── 1. Build feature pipeline ─────────────────────────────────────────────
    print("Loading and preprocessing data...")
    df, feature_cols = build_feature_pipeline(config.MASTER_CSV)
    print(f"Features: {len(feature_cols)}")

    # ── 2. Out-of-sample split (last 20%, NOT seen during training) ───────────
    total  = len(df)
    chunk  = total // 5
    # WFW window 2 (same as train_agent.py windows[2]): train = 0:4*chunk, test = 4*chunk:end
    test_df = df.iloc[chunk * 4 :].reset_index(drop=True)
    print(f"Test set: {len(test_df)} candles (out-of-sample)")

    # ── 3. Build env ──────────────────────────────────────────────────────────
    def make_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=config.WIN_SIZE,
            sl_tp_pairs=config.SL_TP_PAIRS,
            sl_options=config.SL_OPTIONS,
            tp_options=config.TP_OPTIONS,
            feature_columns=feature_cols,
            lot_size=config.TRAIN_LOT_UNITS,
            initial_equity_usd=config.TRAIN_INITIAL_EQUITY_USD,
            # No episode cap for backtest — run through the full test set
            episode_max_steps=None,
            # Use the SAME circuit breaker as live trading so the backtest
            # reflects what would actually happen on the real account.
            max_drawdown_fraction=config.MAX_DRAWDOWN_FRACTION,   # 0.30
            random_start=False,
            spread_pips=1.0,
            max_slippage_pips=0.2,        # realistic slippage
            teacher_mode=False,           # no teacher — pure model evaluation
            allow_manual_close=config.ALLOW_MANUAL_CLOSE,
        )

    vec_env = DummyVecEnv([make_env])

    # ── 4. Load normalizer ────────────────────────────────────────────────────
    norm_path = os.path.join(config.MODELS_DIR, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("VecNormalize loaded.")
    else:
        print("⚠️  vec_normalize.pkl not found — running without normalization.")

    # ── 5. Load model ─────────────────────────────────────────────────────────
    model_path = os.path.join(config.MODELS_DIR, "model_eurusd_titany_v12_sniper.zip")
    if not os.path.exists(model_path):
        # Fall back to best_model from EvalCallback
        model_path = os.path.join(config.MODELS_DIR, "best_models", "best_model.zip")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = RecurrentPPO.load(model_path, env=vec_env, device="cpu")

    # ── 6. Run backtest ───────────────────────────────────────────────────────
    print("Running backtest (deterministic)...")
    equity_curve, closed_trades = run_backtest(model, vec_env, config.TRAIN_INITIAL_EQUITY_USD)

    # ── 7. Summary stats ──────────────────────────────────────────────────────
    stats = print_summary(equity_curve, closed_trades, config.TRAIN_INITIAL_EQUITY_USD)

    # ── 8. Save trade log ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(_ROOT_DIR, "reports"), exist_ok=True)
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        trade_csv = os.path.join(_ROOT_DIR, "reports", "v12_trade_log.csv")
        trades_df.to_csv(trade_csv, index=False)
        print(f"Trade log saved: reports/v12_trade_log.csv ({len(trades_df)} trades)")

    # ── 9. Plot equity curve ──────────────────────────────────────────────────
    initial = config.TRAIN_INITIAL_EQUITY_USD
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Equity curve
    ax = axes[0]
    ax.plot(equity_curve, color="#2196F3", linewidth=1, label="Equity")
    ax.axhline(initial,         color="gray",   linestyle="--", alpha=0.6, label=f"Start ${initial:.0f}")
    ax.axhline(initial * 1.50, color="#4CAF50", linestyle="--", alpha=0.8, label="+50% Target")
    ax.axhline(initial * 0.70, color="#F44336", linestyle="--", alpha=0.8, label="-30% Max DD")
    ax.fill_between(range(len(equity_curve)), equity_curve,
                    initial, where=[e > initial for e in equity_curve],
                    alpha=0.15, color="#4CAF50")
    ax.fill_between(range(len(equity_curve)), equity_curve,
                    initial, where=[e < initial for e in equity_curve],
                    alpha=0.15, color="#F44336")
    ax.set_title("V12 Sniper — Equity Curve (Out-of-Sample)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd_pct = (peak - eq) / peak * 100
    ax2 = axes[1]
    ax2.fill_between(range(len(dd_pct)), dd_pct, color="#F44336", alpha=0.4, label="Drawdown %")
    ax2.axhline(30, color="#F44336", linestyle="--", alpha=0.7, label="30% limit")
    ax2.invert_yaxis()
    ax2.set_title("Drawdown (%)", fontsize=11)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add stats annotation
    if stats:
        txt = (f"Return: {stats['total_return_pct']:+.1f}%  |  "
               f"WinRate: {stats['win_rate']:.0f}%  |  "
               f"PF: {stats['profit_factor']:.2f}  |  "
               f"MaxDD: {stats['max_dd_pct']:.1f}%  |  "
               f"Trades: {stats['n_trades']}")
        fig.text(0.5, 0.01, txt, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    report_path = os.path.join(_ROOT_DIR, "reports", "v12_backtest_report.png")
    plt.savefig(report_path, dpi=150)
    print(f"Chart saved: reports/v12_backtest_report.png")

    if "--headless" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
