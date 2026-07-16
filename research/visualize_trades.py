"""
visualize_trades.py  —  TITANY AI Trade Visualizer
====================================================
Muestra en un gráfico interactivo TODAS las operaciones que el modelo IA
realizó durante el test set:
  ▲ Triángulo VERDE = BUY (compra para subir)
  ▼ Triángulo ROJO  = SELL (venta para bajar)
  ○ Círculo AZUL    = Cierre con ganancia
  ○ Círculo NARANJA = Cierre con pérdida
El eje X = tiempo (velas H1), el eje Y = precio EURUSD.
Se imprime también un resumen estadístico al final.
Uso:
    python visualize_trades.py
    python visualize_trades.py --headless          # Sin ventana emergente
    python visualize_trades.py --window 200        # Solo últimas 200 velas del test
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from env.indicators import (
    load_and_preprocess_data,
    add_quant_features,
    add_hmm_regime_proxy,
    add_physics_features,
    add_golden_strategy_features,
    add_volume_sniper_features,
)
from env.trading_env import ForexTradingEnv

_ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILE_PATH  = config.MASTER_CSV
# FIX: use absolute paths so the script works from any CWD
MODEL_PATH = os.path.join(config.MODELS_DIR, "model_eurusd_titany_v12_sniper.zip")
VEC_NORM   = os.path.join(config.MODELS_DIR, "vec_normalize.pkl")
REPORTS_DIR = os.path.join(_ROOT_DIR, "reports")
SL_OPTS    = [20, 50, 100]
TP_OPTS    = [40, 100, 200]
WIN        = 30
def build_test_env(test_df, feature_cols):
    """Crea el entorno de evaluación (sin normalización de rewards)."""
    def _make():
        return ForexTradingEnv(
            df=test_df, window_size=WIN,
            sl_options=SL_OPTS, tp_options=TP_OPTS,
            feature_columns=feature_cols, random_start=False
        )
    vec = DummyVecEnv([_make])
    if os.path.exists(VEC_NORM):
        vec = VecNormalize.load(VEC_NORM, vec)
        vec.training = False
        vec.norm_reward = False
    return vec
def run_episode(vec_env, model):
    """
    Ejecuta un episodio completo y recoge:
      - prices      : precio de cierre en cada paso
      - trades      : lista de dicts con info de cada operación
      - equity_curve: equity en cada paso
    """
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    prices       = []
    equity_curve = []
    trades       = []                                    
    open_trade   = None                                            
    step_idx     = 0
    done = False
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        obs, reward, done_arr, info_arr = vec_env.step(action)
        episode_starts = done_arr
        done = bool(done_arr[0])
        info = info_arr[0]
        ti   = info.get("last_trade_info")
        if ti is None:
            ti = {}
        price = info.get("close_price", None)
        if price is None or price == 0:
            price = prices[-1] if prices else 1.17
        prices.append(float(price))
        equity_curve.append(info.get("equity_usd", 10000.0))
        if ti.get("event") == "OPEN":
            open_trade = {
                "open_step"    : step_idx,
                "open_price"   : ti.get("entry_price", price),
                "direction"    : ti.get("position", 0),                     
                "sl"           : ti.get("sl_price"),
                "tp"           : ti.get("tp_price"),
            }
        elif ti.get("event") == "CLOSE" and open_trade is not None:
            net_pips = ti.get("net_pips", 0.0)
            open_trade.update({
                "close_step"   : step_idx,
                "close_price"  : ti.get("exit_price", price),
                "net_pips"     : net_pips,
                "reason"       : ti.get("reason", "?"),
                "win"          : net_pips > 0,
            })
            trades.append(open_trade)
            open_trade = None
        step_idx += 1
    return prices, trades, equity_curve
def plot_trades(prices, trades, equity_curve, window=None):
    """Genera el gráfico principal de trades sobre el precio."""
    n = len(prices)
    x = np.arange(n)
    if window:
        start = max(0, n - window)
        prices       = prices[start:]
        equity_curve = equity_curve[start:]
        x            = np.arange(len(prices))
        adj_trades = []
        for t in trades:
            os_ = t["open_step"]  - start
            cs_ = t["close_step"] - start
            if cs_ >= 0 and os_ < len(prices):
                t2 = dict(t)
                t2["open_step"]  = max(0, os_)
                t2["close_step"] = min(len(prices)-1, cs_)
                adj_trades.append(t2)
        trades = adj_trades
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 11),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
    ax1.plot(x, prices, color="#58a6ff", linewidth=0.8, alpha=0.85, zorder=1)
    ax1.set_ylabel("Precio EURUSD", color="white", fontsize=11)
    ax1.yaxis.label.set_color("white")
    ax1.set_title("TITANY AI — Operaciones sobre el Precio (Test Set)",
                  color="white", fontsize=14, pad=14)
    ax1.grid(True, color="#21262d", linewidth=0.5)
    for t in trades:
        color = "#3fb950" if t["win"] else "#f85149"
        ax1.plot(
            [t["open_step"], t["close_step"]],
            [t["open_price"], t["close_price"]],
            color=color, linewidth=0.6, alpha=0.4, zorder=2
        )
    buy_x  = [t["open_step"]  for t in trades if t["direction"] ==  1]
    sell_x = [t["open_step"]  for t in trades if t["direction"] == -1]
    buy_p  = [t["open_price"] for t in trades if t["direction"] ==  1]
    sell_p = [t["open_price"] for t in trades if t["direction"] == -1]
    ax1.scatter(buy_x,  buy_p,  marker="^", s=70,  color="#3fb950",
                zorder=5, label=f"BUY  ({len(buy_x)})")
    ax1.scatter(sell_x, sell_p, marker="v", s=70,  color="#f85149",
                zorder=5, label=f"SELL ({len(sell_x)})")
    win_close_x  = [t["close_step"]  for t in trades if t["win"]]
    win_close_p  = [t["close_price"] for t in trades if t["win"]]
    loss_close_x = [t["close_step"]  for t in trades if not t["win"]]
    loss_close_p = [t["close_price"] for t in trades if not t["win"]]
    ax1.scatter(win_close_x,  win_close_p,  marker="o", s=40,
                color="#58a6ff", zorder=4, alpha=0.8, label="Cierre ✔")
    ax1.scatter(loss_close_x, loss_close_p, marker="o", s=40,
                color="#d29922", zorder=4, alpha=0.8, label="Cierre ✘")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=9, loc="upper left")
    eq = np.array(equity_curve)
    ax2.fill_between(x, eq, 10000, where=(eq >= 10000),
                     color="#3fb950", alpha=0.3)
    ax2.fill_between(x, eq, 10000, where=(eq <  10000),
                     color="#f85149", alpha=0.3)
    ax2.plot(x, eq, color="#58a6ff", linewidth=1.0)
    ax2.axhline(10000, color="#8b949e", linestyle="--", linewidth=0.7)
    ax2.set_ylabel("Equity ($)", color="white", fontsize=10)
    ax2.set_xlabel("Velas H1 (pasos)", color="white", fontsize=10)
    ax2.yaxis.label.set_color("white")
    ax2.xaxis.label.set_color("white")
    ax2.grid(True, color="#21262d", linewidth=0.5)
    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, "reporte_trades_visualizados.png")
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"✅ Gráfico guardado → {out}")
    return fig
def print_summary(trades):
    """Imprime estadísticas de las operaciones."""
    if not trades:
        print("⚠️  No se registraron operaciones.")
        return
    total  = len(trades)
    wins   = sum(1 for t in trades if t["win"])
    losses = total - wins
    wr     = wins / total * 100 if total else 0
    pips   = [t["net_pips"] for t in trades]
    total_pips = sum(pips)
    avg_win    = np.mean([p for p in pips if p > 0]) if wins   else 0
    avg_loss   = np.mean([p for p in pips if p <= 0]) if losses else 0
    buys       = sum(1 for t in trades if t["direction"] ==  1)
    sells      = sum(1 for t in trades if t["direction"] == -1)
    print("\n" + "═"*52)
    print("       TITANY AI — Resumen de Operaciones")
    print("═"*52)
    print(f"  Total operaciones : {total}")
    print(f"  BUY  ▲            : {buys}   ({buys/total*100:.1f}%)")
    print(f"  SELL ▼            : {sells}  ({sells/total*100:.1f}%)")
    print(f"  Ganadas           : {wins}   ({wr:.1f}%)")
    print(f"  Perdidas          : {losses}")
    print(f"  Pips totales netos: {total_pips:+.1f}")
    print(f"  Ganancia media    : +{avg_win:.1f} pips")
    print(f"  Pérdida media     :  {avg_loss:.1f} pips")
    print("═"*52 + "\n")
    df_t = pd.DataFrame(trades)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_out = os.path.join(REPORTS_DIR, "reporte_trades_detalle.csv")
    df_t.to_csv(csv_out, index=False)
    print(f"✅ Detalle guardado → {csv_out}\n")
def main():
    parser = argparse.ArgumentParser(description="TITANY AI Trade Visualizer")
    parser.add_argument("--headless", action="store_true",
                        help="No mostrar ventana gráfica")
    parser.add_argument("--window", type=int, default=None,
                        help="Mostrar solo las últimas N velas del test")
    args = parser.parse_args()
    if args.headless:
        matplotlib.use("Agg")
    print("📂 Cargando datos...")
    df, feature_cols = load_and_preprocess_data(FILE_PATH)
    df, quant_cols   = add_quant_features(df);           feature_cols.extend(quant_cols)
    # FIX: include regime_proxy to match the training pipeline
    df, regime_cols  = add_hmm_regime_proxy(df);         feature_cols.extend(regime_cols)
    df, phys_cols    = add_physics_features(df);          feature_cols.extend(phys_cols)
    df, gold_cols    = add_golden_strategy_features(df);  feature_cols.extend(gold_cols)
    df, sniper_cols  = add_volume_sniper_features(df);   feature_cols.extend(sniper_cols)
    total_len  = len(df)
    chunk_size = total_len // 5
    test_df    = df.iloc[chunk_size*4 : total_len]
    print(f"✅ Test set: {len(test_df)} velas H1")
    print("🧠 Cargando modelo...")
    model   = RecurrentPPO.load(MODEL_PATH)
    vec_env = build_test_env(test_df, feature_cols)
    print("🚀 Ejecutando episodio de evaluación...")
    prices, trades, equity_curve = run_episode(vec_env, model)
    print(f"   Pasos totales: {len(prices)} | Trades: {len(trades)}")
    print_summary(trades)
    fig = plot_trades(prices, trades, equity_curve, window=args.window)
    if not args.headless:
        plt.show()
if __name__ == "__main__":
    main()
