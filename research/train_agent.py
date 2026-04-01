import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch as th
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor                              
from sb3_contrib import RecurrentPPO                           
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.indicators import load_and_preprocess_data, add_quant_features, augment_data_noise, add_hmm_regime_proxy, add_physics_features, add_golden_strategy_features, add_volume_sniper_features
from env.trading_env import ForexTradingEnv
def main():
    N_CORES = min(6, max(1, os.cpu_count() - 1)) 
    th.set_num_threads(1)                                            
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    flag_path = os.path.join(root_dir, "stop_training.flag")
    if os.path.exists(flag_path):
        os.remove(flag_path) # Limpiar bandera vieja antes de empezar

    print(f"[MODO ESTABLE] Iniciando entrenamiento: Usando {N_CORES} núcleos.")
    print(f"[REGLA] Solo abre si Multiverso y Genetica dan LUZ VERDE.")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_dir, "data", "EURUSD_Hourly_2010_2026.csv")
    df, feature_cols = load_and_preprocess_data(file_path)
    df, quant_cols = add_quant_features(df)
    feature_cols.extend(quant_cols)
    df, physics_cols = add_physics_features(df)
    feature_cols.extend(physics_cols)
    df, golden_cols = add_golden_strategy_features(df)
    feature_cols.extend(golden_cols)
    df, sniper_cols = add_volume_sniper_features(df)
    feature_cols.extend(sniper_cols)
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    feature_cols = [c for c in feature_cols if c not in to_drop]
    print(f"Features seleccionadas: {len(feature_cols)} (Ruido eliminado: {len(to_drop)} columnas).")
    total_len = len(df)
    chunk_size = total_len // 5                                                 
    windows = [
        (df.iloc[0 : chunk_size*2], df.iloc[chunk_size*2 : chunk_size*3]),                                        
        (df.iloc[0 : chunk_size*3], df.iloc[chunk_size*3 : chunk_size*4]),                                            
        (df.iloc[0 : chunk_size*4], df.iloc[chunk_size*4 : total_len]),                                                   
        (df.iloc[chunk_size : chunk_size*4], df.iloc[chunk_size*4 : total_len])                                                            
    ]
    train_df, test_df = windows[2]                                        
    print(f"[INFO] Dataset WFW Seleccionado: Train={len(train_df)} | Test={len(test_df)}")
    train_df = augment_data_noise(train_df, probability=0.4)
    print("[INFO] Data Augmentation Activa (Shadow-Noise: ±0.5 pips).")
    SL_OPTS = [20, 50, 100]                                                
    TP_OPTS = [40, 100, 200]                                  
    WIN = 30
    def make_train_env():
        env = ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            random_start=True, feature_columns=feature_cols,
            hold_reward_weight=0.001,                                            
            open_penalty_pips=0.1,                                                                 
            time_penalty_pips=0.001,                                                                       
            unrealized_delta_weight=0.10,                                       
            teacher_mode=True                                                              
        )
        return Monitor(env)                                                
    train_vec_env = SubprocVecEnv([make_train_env] * N_CORES)
    best_model_path = os.path.join(root_dir, 'models', 'best_model.zip')
    norm_save_path = os.path.join(root_dir, "models", "vec_normalize.pkl")
    if os.path.exists(norm_save_path):
        print(f"[MEMORIA] Cargando estadísticas de normalización de {norm_save_path}")
        train_vec_env = VecNormalize.load(norm_save_path, train_vec_env)
        train_vec_env.training = True
        train_vec_env.norm_reward = True
    else:
        print("[MEMORIA] Creando nueva normalización...")
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    def make_test_eval_env():
        env = ForexTradingEnv(df=test_df, window_size=WIN, sl_options=SL_OPTS, 
                               tp_options=TP_OPTS, feature_columns=feature_cols, 
                               random_start=False, teacher_mode=True)
        return Monitor(env)
    test_eval_env = SubprocVecEnv([make_test_eval_env])
    test_eval_env = VecNormalize(test_eval_env, norm_obs=True, norm_reward=False, training=False)
    if os.path.exists(norm_save_path):
        test_eval_env.obs_rms = train_vec_env.obs_rms
    eval_callback = EvalCallback(
        test_eval_env, 
        best_model_save_path=os.path.join(root_dir, 'models', 'best_models'),
        log_path=os.path.join(root_dir, 'logs'), 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    class StopTrainingCallback(BaseCallback):
        def __init__(self, flag_path, verbose=0):
            super().__init__(verbose)
            self.flag_path = flag_path
        def _on_step(self):
            if os.path.exists(self.flag_path):
                print("\n[INFO] Señal de detención AFK detectada. Guardando progreso de emergencia...", flush=True)
                try:
                    # Hacemos el guardado forzado DENTRO del callback
                    emergency_path = os.path.join(os.path.dirname(self.flag_path), "models", "model_eurusd_titany_v12_sniper.zip")
                    self.model.save(emergency_path)
                    print("✅ [EMERGENCIA] Modelo Neural guardado correctamente.", flush=True)
                except Exception as e:
                    print(f"❌ Error al guardar en emergencia: {e}", flush=True)
                
                try:
                    os.remove(self.flag_path)
                except: pass
                
                # Asesinato limpio del proceso para evitar cuelgues de memoria del SubprocVecEnv
                import sys
                sys.exit(0)
                
            return True
            
    stop_callback = StopTrainingCallback(flag_path)
    if os.path.exists(best_model_path):
        print(f"[MEMORIA] Cargando cerebro evolucionado: {best_model_path}")
        model = RecurrentPPO.load(
            best_model_path,
            env=train_vec_env,
            verbose=1,
            learning_rate=0.00003,
            device="cpu"
        )
    else:
        print("[MEMORIA] No se encontró cerebro previo. Iniciando arquitectura Singularidad...")
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],                  
                vf=[512, 512, 512]                                          
            ),
            activation_fn=th.nn.Tanh,                                              
            lstm_hidden_size=128,                                               
            n_lstm_layers=2,
            enable_critic_lstm=True
        )
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_vec_env,
            verbose=1,
            learning_rate=0.00003,
            n_steps=1536,
            batch_size=384,
            ent_coef=0.05,
            gamma=0.995,
            policy_kwargs=policy_kwargs
        )
    print("[INFO] Configurando Entorno Quantico...")
    model.ent_coef = 0.05
    model.learn(total_timesteps=2500000, callback=[eval_callback, stop_callback])
    
    if not os.path.exists(flag_path):
        print("[INFO] Refinando Estrategia de Alta Precisión (5M pasos)...")
        model.ent_coef = 0.01
        model.learn(total_timesteps=5000000, callback=[eval_callback, stop_callback], reset_num_timesteps=False)
        
    if not os.path.exists(flag_path):
        print("[INFO] Fase de Endurecimiento: Miedo a la Pérdida (2.5M pasos)...")
        model.ent_coef = 0.005
        model.learn(total_timesteps=2500000, callback=[eval_callback, stop_callback], reset_num_timesteps=False)
        
    model_name = "model_eurusd_titany_v12_sniper"
    model_save_path = os.path.join(root_dir, "models", model_name)
    model.save(model_save_path)
    norm_save_path = os.path.join(root_dir, "models", "vec_normalize.pkl")
    train_vec_env.save(norm_save_path)
    print(f"🧬 ¡V12 SNIPER COMPLETADO! Modelo guardado en 'models/{model_name}'.", flush=True)
    print(f"📊 Estrategia EMA200+MACD integrada en el núcleo neuronal.", flush=True)
    
    if os.path.exists(flag_path):
        os.remove(flag_path)
        
    print("Generando gráfico de progreso del entrenamiento completo...", flush=True)
    eval_log_path = "./logs/evaluations.npz"
    graph_generated = False
    if os.path.exists(eval_log_path):
        try:
            data = np.load(eval_log_path)
            timesteps = data["timesteps"]                                                  
            results   = data["results"]                                                           
            ep_lengths = data.get("ep_lengths", None)
            mean_rewards = results.mean(axis=1)
            std_rewards  = results.std(axis=1)
            plt.figure(figsize=(14, 10))
            plt.subplot(2, 1, 1)
            plt.plot(timesteps, mean_rewards, color='#1f77b4', label='Recompensa Media')
            plt.fill_between(timesteps,
                             mean_rewards - std_rewards,
                             mean_rewards + std_rewards,
                             alpha=0.2, color='#1f77b4', label='±1 Desv. Std.')
            plt.axhline(y=0, color='red', linestyle='--', label='Referencia 0')
            plt.axvline(x=150000, color='orange', linestyle=':', label='Fin Fase 1 (150k)')
            plt.axvline(x=450000, color='green',  linestyle=':', label='Fin Fase 2 (450k)')
            plt.title("TITANY AI — Progreso de Entrenamiento (600k pasos)")
            plt.xlabel("Pasos de Entrenamiento")
            plt.ylabel("Recompensa Media")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.subplot(2, 1, 2)
            if ep_lengths is not None:
                mean_ep_len = ep_lengths.mean(axis=1)
                plt.plot(timesteps, mean_ep_len, color='#ff7f0e', label='Duración Media de Episodes')
                plt.ylabel("Pasos por Episode")
            else:
                plt.plot(timesteps, np.cumsum(mean_rewards), color='#2ca02c', label='Recompensa Acumulada')
                plt.ylabel("Recompensa Acumulada")
            plt.xlabel("Pasos de Entrenamiento")
            plt.legend()
            plt.grid(True, alpha=0.3)
            reports_dir = os.path.join(root_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_img_path = os.path.join(reports_dir, "reporte_600k_pasos_pro_max.png")
            plt.savefig(report_img_path, dpi=150)
            print(f"✅ Gráfico guardado en 'reports/reporte_600k_pasos_pro_max.png'")
            df_curves = pd.DataFrame({
                'Timestep': timesteps,
                'Mean_Reward': mean_rewards,
                'Std_Reward': std_rewards,
            })
            if ep_lengths is not None:
                df_curves['Mean_Episode_Length'] = ep_lengths.mean(axis=1)
            report_csv_path = os.path.join(reports_dir, "reporte_equity_data.csv")
            df_curves.to_csv(report_csv_path, index=False)
            print(f"✅ Datos guardados en 'reports/reporte_equity_data.csv'")
        except Exception as e:
            print(f"⚠️  Error leyendo logs: {e}. Usando método alternativo...")
    if not graph_generated:
        print("Usando método fallback (equity sobre test set)...")
        def get_equity(v_env, m):
            obs = v_env.reset()
            curve = []
            done = False
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            while not done:
                action, lstm_states = m.predict(obs, state=lstm_states,
                                                episode_start=episode_starts,
                                                deterministic=True)
                obs, reward, done, info = v_env.step(action)
                episode_starts = done
                curve.append(info[0].get('equity_usd', 10000.0))
            return curve
        test_res = get_equity(test_eval_env, model)
        plt.figure(figsize=(14, 5))
        plt.plot(test_res, color='#ff7f0e', label='Equity Out-of-sample')
        plt.axhline(y=10000, color='red', linestyle='--')
        plt.title("Rendimiento en Test Set")
        plt.xlabel("Velas H1")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reporte_600k_pasos_pro_max.png", dpi=150)
        print(f"✅ Gráfico fallback guardado ({len(test_res)} velas)")
if __name__ == "__main__":
    import sys
    is_headless = "--headless" in sys.argv
    if is_headless:
        import matplotlib
        matplotlib.use('Agg')
    os.makedirs("./best_models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./tensorboard_logs/", exist_ok=True)
    main()
    if not is_headless:
        plt.show()
