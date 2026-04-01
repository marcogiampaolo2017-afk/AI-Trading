import pandas as pd
import numpy as np
import indicators
from trading_env import ForexTradingEnv
import os
def verify():
    file_path = "data/EURUSD_Hourly_2010_2026.csv"
    if not os.path.exists(file_path):
        print(f"Data file {file_path} not found. Skipping.")
        return
    print("--- Testing Indicators ---")
    df, cols = indicators.load_and_preprocess_data(file_path)
    df, q_cols = indicators.add_quant_features(df)
    df, p_cols = indicators.add_physics_features(df)
    df, r_cols = indicators.add_hmm_regime_proxy(df)
    all_features = cols + q_cols + p_cols + r_cols
    print(f"Feature set integrated: {len(all_features)} items.")
    print(f"Sample features: {all_features[:5]}...{all_features[-5:]}")
    print("\n--- Testing Environment ---")
    env = ForexTradingEnv(
        df=df, window_size=30, 
        sl_options=[10, 20, 30], 
        tp_options=[20, 40, 60],
        feature_columns=all_features
    )
    reset_res = env.reset()
    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
    print(f"Initial Obs Shape: {obs.shape}")
    for i in range(5):
        action = env.action_space.sample()
        step_res = env.step(action)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
        else:
            obs, reward, done, info = step_res
        print(f"Step {i}: Action={action} | Reward={reward:.6f} | Total Equity={info['equity_usd']}")
    print("\n✅ PHASE 6 INTEGRATION VERIFIED.")
if __name__ == "__main__":
    verify()
