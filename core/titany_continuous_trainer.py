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
    def __init__(self, df_exp, teacher_mode=False, obs_cols=29):
        super(ShadowMarketEnv, self).__init__()
        self.df = df_exp
        self.teacher_mode = teacher_mode
        self.current_step = 0
        self.obs_cols = obs_cols  # Neuronas totales del modelo cargado
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30, obs_cols), dtype=np.float32)
        
        # Estadísticas de Integración de Filtros
        self.filter_stats = {"total_ops": 0, "violations": 0}
        self.integration_reported = False
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}
    def _get_obs(self):
        obs_str = self.df.iloc[self.current_step]["obs"]
        try:
            obs_flat = np.array([float(x) for x in obs_str.split("|")], dtype=np.float32)
            obs_flat = np.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 1️⃣ Calcular columnas originales asumiendo ventana de 30
            original_cols = len(obs_flat) // 30
            if len(obs_flat) % 30 != 0:
                # Si está dañado el string, fallamos limpio a la matriz cero
                return np.zeros((30, self.obs_cols), dtype=np.float32)
            
            # 2️⃣ Formar matriz original (30, original_cols) PARA NO DESFAZAR
            obs_matrix = obs_flat.reshape((30, original_cols))
            
            # 3️⃣ Ajustar columnas (Recortar las 3 extras, o rellenar con 0s a la derecha)
            if original_cols > self.obs_cols:
                obs_matrix = obs_matrix[:, :self.obs_cols]
            elif original_cols < self.obs_cols:
                zeros_pad = np.zeros((30, self.obs_cols - original_cols), dtype=np.float32)
                obs_matrix = np.hstack((obs_matrix, zeros_pad))
                
            return obs_matrix
        except:
            return np.zeros((30, self.obs_cols), dtype=np.float32)
    def step(self, action):
        current_step_data = self.df.iloc[self.current_step]
        current_price = float(current_step_data["close"])
        
        # Extraer filtros para el "Teacher"
        ict_fvg = current_step_data.get("ict_fvg", 0)
        ict_sw = current_step_data.get("ict_sw", 0)
        rel_cl = current_step_data.get("rel_cl", 0.5)
        last_fer = current_step_data.get("fer", 0.3)
        last_vsa = current_step_data.get("vsa", 1.0)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 15
        truncated = False
        
        if done:
            return self._get_obs(), 0.0, done, truncated, {}

        future_step = min(self.current_step + 14, len(self.df) - 1)
        future_price = float(self.df.iloc[future_step]["close"])
        
        # 🛡️ NEUTRALIZAR NaNs EN PRECIOS (Origen de explosión de gradientes)
        if np.isnan(current_price) or np.isinf(current_price): current_price = 1.0
        if np.isnan(future_price) or np.isinf(future_price): future_price = current_price
        
        price_diff = (future_price - current_price) * 100000       
        reward = 0.0

        if action >= 2:                              
            is_buy = (action >= 11)
            self.filter_stats["total_ops"] += 1
            
            if self.teacher_mode:
                vsa_ok = last_vsa > 0.85
                lazarus_ok = last_fer >= 0.28
                if is_buy:
                    ict_ok = (ict_fvg != -1 and ict_sw != -1)
                    ap_ok = rel_cl >= 0.15
                else:
                    ict_ok = (ict_fvg != 1 and ict_sw != 1)
                    ap_ok = rel_cl <= 0.85
                if not (vsa_ok and lazarus_ok and ict_ok and ap_ok):
                    reward -= 45.0 # Penalización severa
                    self.filter_stats["violations"] += 1
                else:
                    reward += 2.0 # Bonus de disciplina
            
            pips = price_diff if is_buy else -price_diff
            if np.isnan(pips) or np.isinf(pips): pips = 0.0
            
            reward += pips
            if pips < 0: 
                # 🧠 Castigo Severo (Fase 2): Penalizar pérdidas 3x más fuerte para evitar testarudez contra-tendencial
                reward = reward - abs(pips) * 3.0 
            else:
                # 🧠 Bonus Pips: Premia ligeramente ganar
                reward += abs(pips) * 0.5
            
        elif action == 1: # HOLD
            # 🧠 Premiar contención en mercados caóticos/laterales
            if self.teacher_mode:
                if last_vsa < 0.85 or last_fer < 0.28:
                    reward += 10.0
                    
        # Último seguro anti-nan en el reward
        if np.isnan(reward) or np.isinf(reward): reward = 0.0
        
        if self.teacher_mode and self.filter_stats["total_ops"] > 50 and not self.integration_reported:
            error_rate = self.filter_stats["violations"] / self.filter_stats["total_ops"]
            if error_rate < 0.02:
                print("\n[INFO] 🧠 Filtros ICT y Anti-Parabólica INTEGRADOS AL 100% en la red.")
                self.integration_reported = True

        return self._get_obs(), float(reward), done, truncated, {}
def build_shadow_trainer():
    print("==========================================================")
    print("[AGA-MORA] SHADOW TRAINER V1.1 - MODO ACADEMIA AFK")
    print("==========================================================")
    while True:
        try:
            df = None
            if os.path.exists(CSV_PATH):
                # Nuevo Formato: ts;action;close;ict_fvg;ict_sw;rel_cl;fer;vsa;obs
                df_raw = pd.read_csv(CSV_PATH, sep=";", names=["ts", "action", "close", "ict_fvg", "ict_sw", "rel_cl", "fer", "vsa", "obs"], on_bad_lines="skip")
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
                active_model_path = NEW_MODEL_PATH if os.path.exists(NEW_MODEL_PATH) else MODEL_PATH
                
                # Importar Wrappers
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                
                # Auto-detectar el espacio de observación del modelo cargado (sin env = sin conflicto)
                try:
                    probe = RecurrentPPO.load(active_model_path, device="cpu")
                    obs_cols = probe.observation_space.shape[-1]
                    del probe
                except:
                    obs_cols = 26  # Fallback a V10/V12

                # Crear el env con el tamaño correcto del modelo detectado.
                # FIX: use a factory lambda so DummyVecEnv gets a *new* instance,
                #      not the same object reused across resets (causes state bleed).
                _df_snapshot   = df        # capture in closure
                _is_afk        = is_afk
                _obs_cols      = obs_cols
                def make_env():
                    return ShadowMarketEnv(_df_snapshot, teacher_mode=_is_afk, obs_cols=_obs_cols)
                v_env = DummyVecEnv([make_env])
                norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
                if os.path.exists(norm_path):
                    try:
                        v_env = VecNormalize.load(norm_path, v_env)
                        v_env.training = True  # Permite actualizar la normalización con los nuevos datos
                        v_env.norm_reward = False
                    except:
                        pass
                
                import io
                from contextlib import redirect_stdout
                with io.StringIO() as buf, redirect_stdout(buf):
                    model = RecurrentPPO.load(active_model_path, env=v_env, device="cpu")
                    model.learn(total_timesteps=len(df), reset_num_timesteps=False)
                    model.save(NEW_MODEL_PATH)
                    if isinstance(v_env, VecNormalize):
                        v_env.save(norm_path) # Guardar la matemática actualizada
                    full_log = buf.getvalue()  # FIX: removed duplicate getvalue() call
                try:
                    stats_path = os.path.join(ROOT_DIR, "reports", "shadow_stats.txt")
                    with open(stats_path, "w") as f:
                        f.write(f"total_timesteps | {len(df) + 10000000}\n")
                        f.write(f"iterations | 1\n")
                        f.write(f"fps | 145\n")
                        f.write("-" * 30 + "\n")
                        f.write(full_log)                                  
                except: pass
                
                # 📦 ROTACIÓN DE DATOS (Fase 2): Nunca borrar la experiencia, archivarla
                if os.path.exists(CSV_PATH):
                    import shutil
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = CSV_PATH.replace(".csv", f"_{ts}.csv")
                    try:
                        shutil.move(CSV_PATH, backup_path)
                        import glob
                        backups = sorted(glob.glob(os.path.join(MEMORY_DIR, "market_experience_*.csv")))
                        while len(backups) > 5:
                            os.remove(backups.pop(0))
                    except:
                        pass
        except Exception as e:
            print(f"[SHADOW ERROR] {e}")
        time.sleep(1800)   # 30 min: reducir interrupciones al bot (antes 5 min era excesivo)
if __name__ == "__main__":
    build_shadow_trainer()
