import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO # TRUE MEMORY ARCHITECTURE

from indicators import load_and_preprocess_data, add_quant_features, augment_data_noise, add_hmm_regime_proxy, add_physics_features, add_golden_strategy_features
from trading_env import ForexTradingEnv

def main():
    # --- OPTIMIZACIÓN MULTINÚCLEO PARA VELOCIDAD EXTREMA ---
    N_CORES = max(1, os.cpu_count() - 1)  # Deja 1 núcleo libre para el OS
    th.set_num_threads(N_CORES)           # Limita PyTorch para evitar peleas de CPU
    print(f"⚡ Iniciando en MODO TURBO: Usando {N_CORES} núcleos paralelos.")
    
    file_path = "data/EURUSD_Hourly_2010_2026.csv"
    df, feature_cols = load_and_preprocess_data(file_path)
    
    # === INYECCIÓN DE CONOCIMIENTO AVANZADO ===
    df, quant_cols = add_quant_features(df)
    feature_cols.extend(quant_cols)
    
    # === INYECCIÓN V6: INFERENCIA FÍSICA (Navier-Stokes, Fisher, Hawkes) ===
    df, physics_cols = add_physics_features(df)
    feature_cols.extend(physics_cols)
    
    # === INYECCIÓN V10: GOLDEN STRATEGY (EMA200 + MACD) ===
    df, golden_cols = add_golden_strategy_features(df)
    feature_cols.extend(golden_cols)
    
    # === FEATURE SELECTION PRO (Pillar 6) ===
    # Filtramos por correlación para eliminar ruido (> 0.95 redundancy)
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    feature_cols = [c for c in feature_cols if c not in to_drop]
    
    print(f"Features seleccionadas: {len(feature_cols)} (Ruido eliminado: {len(to_drop)} columnas).")

    # === ESTRATEGIA WALK-FORWARD (Pillar 5) ===
    # Dividimos en 4 bloques para validar robustez
    total_len = len(df)
    chunk_size = total_len // 5 # Divide data into 5 chunks (P0, P1, P2, P3, P4)
    
    # Create 4 Walk-Forward windows
    # Each window trains on an expanding dataset and tests on the next chunk
    windows = [
        (df.iloc[0 : chunk_size*2], df.iloc[chunk_size*2 : chunk_size*3]), # Window 1: Train on P0, P1; Test on P2
        (df.iloc[0 : chunk_size*3], df.iloc[chunk_size*3 : chunk_size*4]), # Window 2: Train on P0, P1, P2; Test on P3
        (df.iloc[0 : chunk_size*4], df.iloc[chunk_size*4 : total_len]),    # Window 3: Train on P0, P1, P2, P3; Test on P4
        (df.iloc[chunk_size : chunk_size*4], df.iloc[chunk_size*4 : total_len]) # Window 4: Train on P1, P2, P3; Test on P4 (shifted start)
    ]
    
    train_df, test_df = windows[2] # Window 3: Mayor profundidad histórica
    print(f"✅ Dataset WFW Seleccionado: Train={len(train_df)} | Test={len(test_df)}")
    
    # APLICAR DATA AUGMENTATION (Ruido Sintético - Fase 5)
    train_df = augment_data_noise(train_df, probability=0.4)
    print("🧪 Data Augmentation Activa (Shadow-Noise: ±0.5 pips).")

    # === OPCIONES DE SL/TP (V9 Lazarus - Simplificadas) ===
    # Reducción 74 -> 20 acciones para mejor convergencia
    SL_OPTS = [20, 50, 100]   # 3 opciones: conservador, balanceado, amplio
    TP_OPTS = [40, 100, 200]  # 3 opciones: mantiene ratio 2:1
    WIN = 30

    def make_train_env():
        return ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            random_start=True, feature_columns=feature_cols,
            hold_reward_weight=0.001,      # V9: Pequeño incentivo a la paciencia
            open_penalty_pips=0.1,         # V11: Reducción masiva para quitar cobardía (antes 1.2)
            time_penalty_pips=0.001,       # V11: Reducción para enseñar a surfear tendencias (antes 0.003)
            unrealized_delta_weight=0.10   # V9: Reducido de V8 para estabilidad
        )

    # Entorno Multipropósito (Varios procesos a la vez)
    train_vec_env = SubprocVecEnv([make_train_env] * N_CORES)
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)

    def make_test_eval_env():
        return ForexTradingEnv(df=test_df, window_size=WIN, sl_options=SL_OPTS, 
                               tp_options=TP_OPTS, feature_columns=feature_cols, random_start=False)

    test_eval_env = SubprocVecEnv([make_test_eval_env])
    test_eval_env = VecNormalize(test_eval_env, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(
        test_eval_env, 
        best_model_save_path='./best_models/',
        log_path='./logs/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )

    # --- PLAN CYBER-V5 SINGULARITY (ACTOR-CRITIC DESACOPLADO) ---
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],      # Actor ágil
            vf=[512, 512, 512]  # Critic masivo para evaluación profunda
        ),
        activation_fn=th.nn.Tanh,  # V9: Corrección - usar th.nn en lugar de nn
        lstm_hidden_size=128,    # V9: Reducido de 256 para anti-overfitting
        n_lstm_layers=2,
        enable_critic_lstm=True
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_vec_env,
        verbose=1,
        learning_rate=0.00003,  # V9: Entre V7 y V8 para estabilidad
        n_steps=1536,           # V9: Reducido de V8 para evitar overfitting
        batch_size=384,         # V9: Equilibrado entre V7 y V8
        ent_coef=0.05,          # V9: Alta exploración inicial
        gamma=0.995,            # V9: Pensamiento a largo plazo
        policy_kwargs=policy_kwargs
        # tensorboard_log disabled - not critical for training
    )

    print("🛡️ FASE 1: V10 Golden Hybrid - Asimilación de Estrategia (150k pasos)...")
    # Fase 1: Alta exploración para descubrir CÓMO usar las señales Golden
    model.ent_coef = 0.05
    model.learn(total_timesteps=150000, callback=eval_callback)

    print("💎 FASE 2: V10 Refinamiento - Maximización de Profit (300k pasos)...")
    model.ent_coef = 0.01
    model.learn(total_timesteps=300000, callback=eval_callback, reset_num_timesteps=False)

    print("🔥 FASE 3: V10 Hardening - Preparación Live (150k pasos)...")
    model.ent_coef = 0.005
    model.learn(total_timesteps=150000, callback=eval_callback, reset_num_timesteps=False)

    # GUARDAR RESULTADOS
    model_name = "model_eurusd_titany_v10_golden"
    model.save(model_name)
    train_vec_env.save("vec_normalize.pkl")
    print(f"🧬 ¡V10 GOLDEN HYBRID COMPLETADO! Modelo guardado como '{model_name}'.")
    print(f"📊 Estrategia EMA200+MACD integrada en el núcleo neuronal.")

    # --- GENERACIÓN AUTOMÁTICA DEL GRÁFICO ---
    # Lee los logs del EvalCallback que guardan el progreso REAL durante los 600k pasos
    print("Generando gráfico de progreso del entrenamiento completo...")

    eval_log_path = "./logs/evaluations.npz"
    graph_generated = False

    if os.path.exists(eval_log_path):
        try:
            data = np.load(eval_log_path)
            timesteps = data["timesteps"]       # Pasos totales de entrenamiento (0 → 600k)
            results   = data["results"]         # Recompensas por evaluación (n_evals, n_episodes)
            ep_lengths = data.get("ep_lengths", None)

            mean_rewards = results.mean(axis=1)
            std_rewards  = results.std(axis=1)

            plt.figure(figsize=(14, 10))

            # --- Subplot 1: Curva de Recompensa Media ---
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

            # --- Subplot 2: Longitud de Episodes (si disponible) ---
            plt.subplot(2, 1, 2)
            if ep_lengths is not None:
                mean_ep_len = ep_lengths.mean(axis=1)
                plt.plot(timesteps, mean_ep_len, color='#ff7f0e', label='Duración Media de Episodes')
                plt.ylabel("Pasos por Episode")
            else:
                # Alternativa: recompensa acumulada
                plt.plot(timesteps, np.cumsum(mean_rewards), color='#2ca02c', label='Recompensa Acumulada')
                plt.ylabel("Recompensa Acumulada")
            plt.xlabel("Pasos de Entrenamiento")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("reporte_600k_pasos_pro_max.png", dpi=150)
            print(f"✅ Gráfico guardado: reporte_600k_pasos_pro_max.png")
            print(f"   → Timesteps totales: {timesteps[-1]:,} | Evaluaciones: {len(timesteps)}")
            print(f"   → Mejor recompensa: {mean_rewards.max():.4f} (paso {timesteps[mean_rewards.argmax()]:,})")
            graph_generated = True

            # Exportar CSV de los datos del gráfico
            df_curves = pd.DataFrame({
                'Timestep': timesteps,
                'Mean_Reward': mean_rewards,
                'Std_Reward': std_rewards,
            })
            if ep_lengths is not None:
                df_curves['Mean_Episode_Length'] = ep_lengths.mean(axis=1)
            df_curves.to_csv("reporte_equity_data.csv", index=False)
            print("✅ Datos guardados en 'reporte_equity_data.csv'")

        except Exception as e:
            print(f"⚠️  Error leyendo logs: {e}. Usando método alternativo...")

    if not graph_generated:
        # Fallback: gráfico simple de equity sobre el test set
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

        test_res = get_equity(test_eval_env, best_model)
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