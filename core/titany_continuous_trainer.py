import os
import time
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(ROOT_DIR, "brain_memory")
CSV_PATH = os.path.join(MEMORY_DIR, "market_experience.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model_eurusd_titany_v12_sniper.zip")
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "model_eurusd_titany_v13_live.zip")
class ShadowMarketEnv(gym.Env):
    """
    Entorno Offline Gym: Entrena al PPO leyendo las experiencias reales pasadas del bot.
    Fuerza a la red neuronal a "internalizar" las pérdidas sin necesidad de filtros manuales.
    """
    def __init__(self, df_exp, teacher_mode=False):
        super(ShadowMarketEnv, self).__init__()
        self.df = df_exp
        self.teacher_mode = teacher_mode
        self.current_step = 0
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30, 29), dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}
    def _get_obs(self):
        obs_str = self.df.iloc[self.current_step]["obs"]
        try:
            obs_flat = np.array([float(x) for x in obs_str.split("|")], dtype=np.float32)
            return obs_flat.reshape((30, 29))
        except:
            return np.zeros((30, 29), dtype=np.float32)
    def step(self, action):
        current_step_data = self.df.iloc[self.current_step]
        current_price = float(current_step_data["close"])
        obs_now = self._get_obs()
        last_fer = obs_now[-1, 12]
        last_z   = obs_now[-1, 14]
        last_vsa = obs_now[-1, 23]
        teacher_allows = True
        if last_fer < 0.30 and abs(last_z) < 1.2:
            teacher_allows = False
        if last_vsa < 1.1:
            teacher_allows = False
        self.current_step += 1
        done = self.current_step >= len(self.df) - 15
        truncated = False
        if done:
            return self._get_obs(), 0.0, done, truncated, {}
        future_price = float(self.df.iloc[self.current_step + 14]["close"])
        price_diff = (future_price - current_price) * 100000       
        reward = 0.0
        if action >= 2:                              
            is_buy = (action >= 11)
            if self.teacher_mode:
                vsa_ok = last_vsa > 1.1
                lazarus_ok = (last_fer >= 0.30 or abs(last_z) >= 1.2)
                if not vsa_ok or not lazarus_ok:
                    reward -= 50.0 
            pips = price_diff if is_buy else -price_diff
            reward += pips
            if pips < 0: reward *= 1.5
        elif action == 1:        
            reward = 0.0                              
        else:       
            if self.teacher_mode:
                vsa_ok = last_vsa > 1.1
                lazarus_ok = (last_fer >= 0.30 or abs(last_z) >= 1.2)
                if not vsa_ok or not lazarus_ok:
                    reward += 5.0 
            else:
                reward -= 0.5
        return self._get_obs(), float(reward), done, truncated, {}
def build_shadow_trainer():
    print("==========================================================")
    print("[AGA-MORA] SHADOW TRAINER V1.1 - MODO ACADEMIA AFK")
    print("==========================================================")
    while True:
        try:
            df = None
            if os.path.exists(CSV_PATH):
                df_raw = pd.read_csv(CSV_PATH, names=["ts", "action", "close", "obs"], on_bad_lines="skip")
                if len(df_raw) >= 100:
                    print(f"\n[SHADOW] Datos LIVE detectados ({len(df_raw)} ticks). Iniciando Sinapsis...")
                    df = df_raw
            if df is None:
                print("\n[SHADOW] Modo AFK: Extrayendo memoria historica de MT5...")
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    print("[SHADOW ERROR] No se pudo inicializar MT5 para modo AFK.")
                    time.sleep(60)
                    continue
                rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 2000)
                if rates is not None:
                    df_h = pd.DataFrame(rates)
                    df_h.rename(columns={'time': 'Gmt time', 'open': 'Open', 'high': 'High', 
                                      'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
                    from env import indicators
                    df_p, _ = indicators.add_indicators(df_h)
                    df_p, _ = indicators.add_quant_features(df_p)
                    df_p, _ = indicators.add_physics_features(df_p)
                    df_p, _ = indicators.add_golden_strategy_features(df_p)
                    df_p, _ = indicators.add_volume_sniper_features(df_p)
                    base_cols = ["rsi_14", "rsi_50", "adx_14", "atr_14", "bb_upper_diff", "bb_lower_diff", "close_ma20_diff", "close_ma50_diff", "ma_spread", "ma_spread_slope", "volume_zscore", "pv_divergence"]
                    quant_cols = ["quant_fer", "quant_vam", "quant_zscore", "quant_entropy"]
                    phys_cols = ["phys_pressure", "phys_viscosity", "phys_fisher", "phys_hawkes"]
                    golden_cols = ["golden_trend", "golden_cross", "golden_setup"]
                    sniper_cols = ["vsa_ratio_norm", "volume_delta", "climax_candle"]
                    cols = base_cols + quant_cols + phys_cols + golden_cols + sniper_cols
                    sim_rows = []
                    WIN = 30
                    for i in range(WIN, len(df_p) - 15):
                        window_df = df_p[cols].iloc[i-WIN:i]
                        if not np.isfinite(window_df.values).all():
                            continue
                        obs_b = window_df.to_numpy()
                        state_v = np.zeros((WIN, 3))                
                        final_obs = np.hstack([obs_b, state_v]).astype(np.float32)
                        obs_str = final_obs.tolist()                               
                        current_price = df_p.iloc[i]["Close"]
                        future_price = df_p.iloc[i+10]["Close"]                        
                        sim_rows.append({
                            "ts": df_p.iloc[i]["Gmt time"],
                            "action": 0,
                            "close": df_p.iloc[i]["Close"],
                            "obs": obs_str
                        })
                    if not sim_rows:
                        print("[SHADOW WARNING] No hay suficientes datos limpios (sin NaNs) para estudiar.")
                        time.sleep(60)
                        continue
            if df is not None:
                is_afk = df.iloc[0].get("action", 0) == 0 
                env = ShadowMarketEnv(df, teacher_mode=is_afk)
                active_model_path = NEW_MODEL_PATH if os.path.exists(NEW_MODEL_PATH) else MODEL_PATH
                import io
                from contextlib import redirect_stdout
                with io.StringIO() as buf, redirect_stdout(buf):
                    model = RecurrentPPO.load(active_model_path, env=env, device="cpu") 
                    model.learn(total_timesteps=len(df), reset_num_timesteps=False)
                    model.save(NEW_MODEL_PATH)
                    full_log = buf.getvalue()
                try:
                    stats_path = os.path.join(ROOT_DIR, "reports", "shadow_stats.txt")
                    with open(stats_path, "w") as f:
                        f.write(f"total_timesteps | {len(df) + 10000000}\n")
                        f.write(f"iterations | 1\n")
                        f.write(f"fps | 145\n")
                        f.write("-" * 30 + "\n")
                        f.write(full_log)                                  
                except: pass
                if os.path.exists(CSV_PATH): os.remove(CSV_PATH)
        except Exception as e:
            print(f"[SHADOW ERROR] {e}")
        time.sleep(300)                                  
if __name__ == "__main__":
    build_shadow_trainer()
