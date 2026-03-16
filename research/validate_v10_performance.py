import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import indicators
from trading_env import ForexTradingEnv

# CONFIGURACIÓN EXACTA DE V10 GOLDEN
WIN_SIZE = 30
SL_OPTS = [20, 50, 100]
TP_OPTS = [40, 100, 200]
MODEL_PATH = "best_models/best_model.zip"
NORM_PATH = "vec_normalize.pkl"
DATA_PATH = "data/EURUSD_Hourly_2010_2026.csv"

def run_backtest():
    print("🚀 Iniciando validación profunda del Modelo V10 Golden...")
    
    # 1. Cargar Datos
    df, feature_cols = indicators.load_and_preprocess_data(DATA_PATH)
    df, _ = indicators.add_quant_features(df)
    df, _ = indicators.add_hmm_regime_proxy(df)
    df, _ = indicators.add_physics_features(df)
    df, _ = indicators.add_golden_strategy_features(df)
    
    # Seleccionar las 23 columnas base
    base_cols = ["rsi_14", "rsi_50", "adx_14", "atr_14", "bb_upper_diff", "bb_lower_diff", 
                 "close_ma20_diff", "close_ma50_diff", "ma_spread", "ma_spread_slope", "volume_zscore", "pv_divergence"]
    quant_cols = ["quant_fer", "quant_vam", "quant_zscore", "quant_entropy"]
    regime_col = ["regime_proxy"]
    phys_cols = ["phys_pressure", "phys_viscosity", "phys_fisher", "phys_hawkes"]
    golden_cols = ["golden_trend", "golden_cross"] # V10 usa los 2 primeros usualmente
    
    final_cols = base_cols + quant_cols + regime_col + phys_cols + golden_cols
    
    # Usar una ventana mayor para ver movimiento
    test_df = df.tail(5000).copy()
    
    print(f"📊 Verificando integridad de features (Primeras 5 filas):")
    print(test_df[final_cols].head())
    
    if test_df[final_cols].isnull().values.any():
        print("⚠️ ALERTA: Hay valores NaN en las features. Eso puede romper el modelo.")
        test_df = test_df.fillna(0)

    def make_env():
        return ForexTradingEnv(df=test_df, window_size=WIN_SIZE, sl_options=SL_OPTS, 
                             tp_options=TP_OPTS, feature_columns=final_cols, random_start=False)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(NORM_PATH, venv)
    venv.training = False
    venv.norm_reward = False
    
    model = RecurrentPPO.load(MODEL_PATH, env=venv)
    
    obs = venv.reset()
    equity_curve = []
    actions_taken = []
    
    print(f"📈 Procesando {len(test_df)} velas de historial...")
    
    for i in range(len(test_df) - WIN_SIZE - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        equity_curve.append(info[0]['equity_usd'])
        actions_taken.append(action[0])
        
        if i % 500 == 0:
            print(f"Paso {i}: Balance={info[0]['equity_usd']:.2f}")

    unique_actions, counts = np.unique(actions_taken, return_counts=True)
    print("📋 Resumen de Acciones:")
    for a, c in zip(unique_actions, counts):
        print(f"  Acción {a}: {c} veces")
            
    # Graficar
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity V10 Golden", color="#00ffcc")
    plt.title(f"BACKTEST V10 GOLDEN (Últimas {len(equity_curve)} horas)")
    plt.xlabel("Pasos (Horas)")
    plt.ylabel("Balance ($)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig("validacion_v10_resultado.png")
    print("✅ Gráfico guardado como 'validacion_v10_resultado.png'")
    
    # Guardar CSV
    pd.DataFrame(equity_curve, columns=["equity"]).to_csv("backtest_v10_data.csv", index=False)
    print("✅ Datos guardados en 'backtest_v10_data.csv'")

if __name__ == "__main__":
    run_backtest()
