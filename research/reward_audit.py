"""
reward_audit.py — descomposición del reward del entorno de TITANY AI.

Sirve para responder una pregunta concreta antes de gastar 15 h de CPU:
¿la política está optimizando P&L o el "shaping" (bonificaciones/castigos
manuales que agregamos en teacher_mode y en el circuito de chaos/drawdown)?

Para cada combinación de política (hold / random / modelo entrenado) y de
teacher_mode (on/off) corre el entorno out-of-sample y acumula la contribución
exacta de cada término del reward (env.step() devuelve info["reward_breakdown"]).

Uso:
    python research/reward_audit.py                 # 3000 velas, todas las políticas
    python research/reward_audit.py --bars 5000
    python research/reward_audit.py --policies hold,random
"""
import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_RESEARCH_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

import config
import test_agent as ta
from env.trading_env import ForexTradingEnv

TERMS = ["teacher", "open_cost", "chaos_penalty", "realized",
         "unrealized_delta", "drawdown", "fer_block"]
# Términos que existen solo por el diseño del reward (no son P&L realizado).
SHAPING = {"teacher", "open_cost", "chaos_penalty", "drawdown", "fer_block"}

_SLICE = None


def build_test_slice(bars: int):
    df, feature_cols = ta.build_feature_pipeline(config.MASTER_CSV)
    chunk = len(df) // 5
    test_df = df.iloc[chunk * 4:].reset_index(drop=True)
    # Necesitamos ventana + bars: recortamos desde el inicio del test set.
    return df, test_df.head(bars + config.WIN_SIZE).reset_index(drop=True), feature_cols


def make_env(test_df, feature_cols, teacher_mode: bool):
    return ForexTradingEnv(
        df=test_df,
        window_size=config.WIN_SIZE,
        sl_tp_pairs=config.SL_TP_PAIRS,
        sl_options=config.SL_OPTIONS,
        tp_options=config.TP_OPTIONS,
        feature_columns=feature_cols,
        lot_size=config.TRAIN_LOT_UNITS,
        initial_equity_usd=config.TRAIN_INITIAL_EQUITY_USD,
        episode_max_steps=None,
        max_drawdown_fraction=config.MAX_DRAWDOWN_FRACTION,
        random_start=False,
        spread_pips=1.0,
        max_slippage_pips=0.2,
        teacher_mode=teacher_mode,
        allow_manual_close=config.ALLOW_MANUAL_CLOSE,
    )


def load_model_policy(feature_cols: list[str]):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_path = os.path.join(config.MODELS_DIR, "model_eurusd_titany_v12_sniper.zip")
    norm_path = os.path.join(config.MODELS_DIR, "vec_normalize.pkl")
    if not (os.path.exists(model_path) and os.path.exists(norm_path)):
        return None
    model = RecurrentPPO.load(model_path, device="cpu")

    # Env mínimo con las MISMAS features para que VecNormalize valide el espacio.
    cols = ["Open", "High", "Low", "Close", "Volume"] + list(feature_cols)
    dummy = ForexTradingEnv(
        df=pd.DataFrame({c: np.zeros(60) for c in cols}),
        window_size=config.WIN_SIZE, feature_columns=feature_cols,
        sl_tp_pairs=config.SL_TP_PAIRS, sl_options=config.SL_OPTIONS,
        tp_options=config.TP_OPTIONS,
        initial_equity_usd=config.TRAIN_INITIAL_EQUITY_USD,
    )
    vec = VecNormalize.load(norm_path, DummyVecEnv([lambda: dummy]))
    vec.training = False
    vec.norm_reward = False
    return model, vec


def run_policy(policy, bars, teacher_mode, data, model_bundle: tuple[Any, Any] | None = None):
    _full, test_df, feature_cols = data
    env = make_env(test_df, feature_cols, teacher_mode)
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(7)

    model = vec = None
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    if policy == "model":
        if model_bundle is None:
            raise RuntimeError("política 'model' sin bundle de modelo/normalizador")
        model, vec = model_bundle

    totals: dict[str, Any] = {t: 0.0 for t in TERMS}
    trades = 0
    prev_pos = 0
    steps = 0
    info = {}
    for _ in range(bars):
        if policy == "hold":
            action = 0
        elif policy == "random":
            action = int(rng.integers(0, env.action_space.n))
        else:
            if model is None or vec is None:
                raise RuntimeError("modelo/normalizador no cargados")
            obs_n = vec.normalize_obs(np.asarray(obs, dtype=np.float32)[None])
            act, lstm_states = model.predict(
                obs_n, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            action = int(np.asarray(act).ravel()[0])
            episode_start = np.zeros((1,), dtype=bool)
        obs, reward, terminated, truncated, info = env.step(action)
        for t, v in info["reward_breakdown"].items():
            totals[t] += v
        pos = int(info["position"])
        if prev_pos != 0 and pos == 0:
            trades += 1
        prev_pos = pos
        steps += 1
        if terminated or truncated:
            break
    totals["total"] = sum(totals[t] for t in TERMS)
    totals["shaping"] = sum(totals[t] for t in TERMS if t in SHAPING)
    totals["pnl"] = totals["total"] - totals["shaping"]
    totals["equity"] = float(info.get("equity_usd", 0.0))
    totals["trades"] = trades
    totals["steps"] = steps
    return totals


def main(argv=None):
    global _SLICE
    parser = argparse.ArgumentParser(description="Auditoría del reward")
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument("--policies", default="hold,random,model")
    args = parser.parse_args(argv)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    print("Cargando datos: " + config.MASTER_CSV)
    full_df, test_df, feature_cols = build_test_slice(args.bars)
    _SLICE = (full_df, test_df, feature_cols)
    print("Ventana OOS: " + str(args.bars) + " velas desde el inicio del test set ("
          + str(len(full_df)) + " velas totales)")

    model_bundle = load_model_policy(feature_cols) if "model" in policies else None
    if "model" in policies and model_bundle is None:
        print("Modelo no disponible: se omite la política 'model'.")

    rows = []
    for teacher in (False, True):
        for policy in policies:
            if policy == "model" and model_bundle is None:
                continue
            res = run_policy(policy, args.bars, teacher, _SLICE, model_bundle)
            res["teacher_mode"] = teacher
            res["policy"] = policy
            rows.append(res)
            print("  " + policy.ljust(6) + " | teacher=" + ("on " if teacher else "off")
                  + " | steps=" + str(res["steps"]) + " trades=" + str(res["trades"])
                  + " equity=" + format(res["equity"], ".2f")
                  + " total=" + format(res["total"], ".1f"))

    # rows ya vienen ordenadas: teacher_mode False/True y luego por política
    table = pd.DataFrame(rows)[
        ["policy", "teacher_mode", "steps", "trades", "equity", "pnl", "shaping", "total"] + TERMS
    ]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("\n=== Contribución por término del reward (pips, suma sobre la ventana) ===")
    print(table.to_string(index=False, float_format="%.2f"))

    reports = os.path.join(_ROOT_DIR, "reports")
    os.makedirs(reports, exist_ok=True)
    out = os.path.join(reports, "reward_audit.csv")
    table.to_csv(out, index=False)
    print("\nCSV guardado en " + out)

    print("\nLectura: 'pnl' = realized + unrealized_delta (lo que refleja el resultado),")
    print("'shaping' = teacher + open_cost + chaos_penalty + drawdown + fer_block.")
    print("Si shaping domina a PnL, la política está optimizando el diseño del reward.")


if __name__ == "__main__":
    main()
