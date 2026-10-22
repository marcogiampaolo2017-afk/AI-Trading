import argparse
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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor                              
from sb3_contrib import RecurrentPPO                           
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from env.indicators import (
    load_and_preprocess_data,
    add_quant_features,
    augment_data_noise,
    add_hmm_regime_proxy,
    add_physics_features,
    add_golden_strategy_features,
    add_volume_sniper_features,
)
from env.trading_env import ForexTradingEnv


def parse_args(argv=None):
    """CLI para diagnosticar runs cortos sin tocar el run de referencia de 3M."""
    parser = argparse.ArgumentParser(description="Entrenamiento RecurrentPPO TITANY AI")
    parser.add_argument("--headless", action="store_true",
                        help="sin ventanas de matplotlib")
    parser.add_argument("--steps", type=int, default=3_000_000,
                        help="total de pasos a entrenar (default: 3M)")
    parser.add_argument("--teacher-mode", choices=["on", "off"], default="on",
                        help="shaping de filtros en el reward del entorno")
    parser.add_argument("--dd-penalty-scale", type=float, default=1.0,
                        help="escala del castigo por drawdown por paso (0.0 lo desactiva)")
    parser.add_argument("--model-name", default="model_eurusd_titany_v12_sniper",
                        help="nombre base del .zip de salida")
    parser.add_argument("--out-dir", default=None,
                        help="carpeta de salida (modelo, normalizador, checkpoints, logs). "
                             "Default: models/ y logs/ del repo")
    parser.add_argument("--seed", type=int, default=None,
                        help="semilla para comparar corridas")
    parser.add_argument("--fresh", action="store_true",
                        help="ignorar modelo/normalizador existentes en out-dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    total_steps = max(2048, int(args.steps))
    teacher = args.teacher_mode == "on"
    # os.cpu_count() puede devolver None según la plataforma -> fallback seguro
    N_CORES = min(6, max(1, (os.cpu_count() or 1) - 1)) 
    th.set_num_threads(1)                                            
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    flag_path = os.path.join(root_dir, "stop_training.flag")
    if os.path.exists(flag_path):
        os.remove(flag_path) # Limpiar bandera vieja antes de empezar

    # Salidas: el run de referencia usa models/ + logs/; los diagnósticos usan
    # --out-dir para no pisar el modelo bueno ni mezclar curvas de evaluación.
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(root_dir, "models")
    best_dir = os.path.join(out_dir, "best_models")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    run_logs = os.path.join(out_dir, "logs") if args.out_dir else os.path.join(root_dir, "logs")
    for _d in (out_dir, best_dir, ckpt_dir, run_logs):
        os.makedirs(_d, exist_ok=True)
    print(f"[RUN] steps={total_steps:,} | teacher={args.teacher_mode} | "
          f"seed={args.seed} | model={args.model_name}")
    print(f"[RUN] salida: {out_dir}")

    print(f"[MODO ESTABLE] Iniciando entrenamiento: Usando {N_CORES} núcleos.")
    print(f"[REGLA] Solo abre si Multiverso y Genetica dan LUZ VERDE.")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = config.MASTER_CSV
    df, feature_cols = load_and_preprocess_data(file_path)
    df, quant_cols = add_quant_features(df)
    feature_cols.extend(quant_cols)
    # regime_proxy is used in the live system — keep training consistent
    df, regime_cols = add_hmm_regime_proxy(df)
    feature_cols.extend(regime_cols)
    df, physics_cols = add_physics_features(df)
    feature_cols.extend(physics_cols)
    df, golden_cols = add_golden_strategy_features(df)
    feature_cols.extend(golden_cols)
    df, sniper_cols = add_volume_sniper_features(df)
    feature_cols.extend(sniper_cols)
    corr_matrix = df.loc[:, feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool), other=np.nan)
    # FIX: pin features that must ALWAYS be present for train/live consistency
    _PROTECTED_FEATURES = {"regime_proxy"}
    to_drop = [
        column for column in upper.columns
        if any(upper[column] > 0.95) and column not in _PROTECTED_FEATURES
    ]
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
    WIN = config.WIN_SIZE
    def make_train_env():
        env = ForexTradingEnv(
            df=train_df, window_size=WIN,
            # Use explicit (sl,tp) pairs — NO Cartesian product.
            # Only R:R >= 2:1 combinations are available so the model cannot
            # physically choose a trade with negative expected value.
            sl_tp_pairs=config.SL_TP_PAIRS,
            sl_options=config.SL_OPTIONS,   # still needed for action_map fallback
            tp_options=config.TP_OPTIONS,
            random_start=True, feature_columns=feature_cols,
            hold_reward_weight=0.001,                                            
            open_penalty_pips=0.1,                                                                 
            time_penalty_pips=0.001,                                                                       
            unrealized_delta_weight=0.10,                                       
            teacher_mode=True,
            # Realistic account parameters for a 100 EUR AvaTrade account
            # lot_size=1000 → 0.01 micro-lot → usd_per_pip=$0.10
            lot_size=config.TRAIN_LOT_UNITS,
            initial_equity_usd=config.TRAIN_INITIAL_EQUITY_USD,
            # Safety: cut episode if equity falls >30% (mirrors live risk limit)
            max_drawdown_fraction=config.MAX_DRAWDOWN_FRACTION,
            drawdown_penalty_scale=args.dd_penalty_scale,
            # No manual close: model only opens/holds, SL/TP handle all exits.
            # Prevents "cut winners short, let losers run" (TP hit only 1%).
            allow_manual_close=config.ALLOW_MANUAL_CLOSE,
            # Cap episode length so the LSTM sees coherent windows
            episode_max_steps=2048,
        )
        return Monitor(env)                                                
    train_vec_env = SubprocVecEnv([make_train_env] * N_CORES)
    # Resume priority: saved final model → EvalCallback best model.
    _saved_model     = os.path.join(out_dir, args.model_name + ".zip")
    _eval_best_model = os.path.join(best_dir, "best_model.zip")
    if args.fresh:
        best_model_path = None
    else:
        best_model_path = _saved_model if os.path.exists(_saved_model) else _eval_best_model
    norm_save_path = os.path.join(out_dir, "vec_normalize.pkl")
    if os.path.exists(norm_save_path) and not args.fresh:
        try:
            _loaded_norm = VecNormalize.load(norm_save_path, train_vec_env)
            train_vec_env = _loaded_norm
            train_vec_env.training = True
            train_vec_env.norm_reward = True
            print(f"[MEMORIA] Normalización cargada de {norm_save_path}")
        except AssertionError as _e:
            # obs-space changed (e.g. more features added) → start fresh
            print(f"[MEMORIA] Normalizador antiguo incompatible ({_e}).")
            print("[MEMORIA] El obs-space cambió — creando nueva normalización desde cero.")
            train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    else:
        print("[MEMORIA] Creando nueva normalización...")
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    def make_test_eval_env():
        env = ForexTradingEnv(
            df=test_df, window_size=WIN,
            sl_tp_pairs=config.SL_TP_PAIRS,
            sl_options=config.SL_OPTIONS,
            tp_options=config.TP_OPTIONS,
            feature_columns=feature_cols,
            random_start=False, teacher_mode=teacher,
            lot_size=config.TRAIN_LOT_UNITS,
            initial_equity_usd=config.TRAIN_INITIAL_EQUITY_USD,
            max_drawdown_fraction=config.MAX_DRAWDOWN_FRACTION,
            allow_manual_close=config.ALLOW_MANUAL_CLOSE,
            drawdown_penalty_scale=args.dd_penalty_scale,
            # Fixed-length eval episodes: without a cap an eval episode runs
            # until the drawdown breaker trips (up to ~20k steps), which makes
            # each eval slow AND makes EvalCallback compare total rewards of
            # episodes with different durations -> the "best model" becomes
            # noise. 2000 steps = ~2 LSTM windows of coherent signal.
            episode_max_steps=2000,
        )
        return Monitor(env)
    test_eval_env = SubprocVecEnv([make_test_eval_env])
    test_eval_env = VecNormalize(test_eval_env, norm_obs=True, norm_reward=False, training=False)
    if os.path.exists(norm_save_path):
        test_eval_env.obs_rms = train_vec_env.obs_rms
    eval_callback = EvalCallback(
        test_eval_env, 
        best_model_save_path=best_dir,
        log_path=run_logs, 
        eval_freq=10000,
        # 3 bounded episodes instead of 1: EvalCallback saves best_model only
        # when the mean improves, so a single noisy episode is not enough.
        n_eval_episodes=3,
        deterministic=True, 
        render=False
    )
    
    class StopTrainingCallback(BaseCallback):
        def __init__(self, flag_path, save_path, verbose=0):
            super().__init__(verbose)
            self.flag_path = flag_path
            self.save_path = save_path
        def _on_step(self):
            if os.path.exists(self.flag_path):
                print("\n[INFO] Señal de detención AFK detectada. Guardando progreso de emergencia...", flush=True)
                try:
                    # Hacemos el guardado forzado DENTRO del callback
                    self.model.save(self.save_path)
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
            
    stop_callback = StopTrainingCallback(flag_path, os.path.join(out_dir, args.model_name))
    # FIX: persist progress regularly so a Ctrl+C / crash never loses everything.
    # save_freq is per-env-step, so divide the target wall interval by N_CORES.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(250000 // N_CORES, 1),
        save_path=ckpt_dir,
        name_prefix="titany_ckpt",
        save_vecnormalize=True,
    )
    callbacks = [eval_callback, stop_callback, checkpoint_callback]
    # ── Try to resume from a previous checkpoint ──────────────────────────────
    model = None
    if best_model_path is not None and os.path.exists(best_model_path):
        try:
            model = RecurrentPPO.load(
                best_model_path,
                env=train_vec_env,
                verbose=1,
                learning_rate=0.00003,
                device="cpu"
            )
            # FIX: apply anti-collapse guardrails to the resumed model too.
            # clip_range is stored as a schedule callable, so wrap the constant.
            model.target_kl = 0.03
            model.n_epochs = 5
            model.clip_range = lambda _progress_remaining: 0.15
            model.max_grad_norm = 0.5
            print(f"[MEMORIA] Cerebro evolucionado cargado: {best_model_path}")
        except Exception as _me:
            # obs-space mismatch or any other incompatibility → retrain from scratch
            print(f"[MEMORIA] Modelo antiguo incompatible ({_me}).")
            print("[MEMORIA] Iniciando arquitectura Singularidad desde cero...")
            model = None

    if model is None:
        print("[MEMORIA] Construyendo nueva red neuronal Singularidad...")
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
            # FIX: hard brake against policy collapse. When the KL between the
            # old and new policy exceeds this, PPO stops the epoch early. This
            # is what prevents the approx_kl=0.46 runaway seen before.
            target_kl=0.03,
            # FIX: fewer epochs per rollout -> less overfitting / smaller updates
            n_epochs=5,
            # FIX: tighter trust region + gradient clipping for stability
            clip_range=0.15,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
        )
    print("[INFO] Configurando Entorno Quantico...")
    model_name      = args.model_name
    model_save_path = os.path.join(out_dir, model_name)
    norm_save_path  = os.path.join(out_dir, "vec_normalize.pkl")

    def _save_progress(tag=""):
        """Persist model + normalizer — only if current steps > already saved steps.

        This prevents an early Ctrl+C (e.g. after 4608 steps) from overwriting a
        much better model that was loaded from a previous long training run.
        """
        try:
            current_steps = getattr(model, "num_timesteps", 0)
            # Read the step count of the model already on disk (if any)
            existing_steps = 0
            if os.path.exists(model_save_path + ".zip"):
                try:
                    import zipfile, json
                    with zipfile.ZipFile(model_save_path + ".zip") as z:
                        if "data" in z.namelist():
                            meta = json.loads(z.read("data").decode())
                            existing_steps = meta.get("num_timesteps", 0)
                except Exception:
                    existing_steps = 0
            if current_steps >= existing_steps:
                model.save(model_save_path)
                train_vec_env.save(norm_save_path)
                print(f"[SAVE{tag}] Guardado: {current_steps:,} pasos "
                      f"(anterior: {existing_steps:,}).", flush=True)
            else:
                print(f"[SAVE{tag}] Omitido: modelo actual ({current_steps:,} pasos) "
                      f"< modelo guardado ({existing_steps:,} pasos). "
                      f"Usando checkpoint existente.", flush=True)
        except Exception as _se:
            print(f"Error guardando progreso: {_se}", flush=True)

    # ── Resume detection: skip phases already completed ───────────────────────
    # When a checkpoint is loaded, model.num_timesteps reflects how far training
    # progressed. We use this to jump directly to the correct phase so a restart
    # never re-runs phases that are already done.
    # Mismas proporciones que el run de referencia de 3M (1M / 2.2M / 3M),
    # pero escaladas a --steps para que los diagnósticos cortos sean comparables.
    _P1 = max(1, total_steps // 3)
    _P2 = max(_P1 + 1, int(total_steps * 22 / 30))
    _P3 = total_steps
    _done = getattr(model, "num_timesteps", 0)
    print(f"[MEMORIA] Pasos ya completados: {_done:,} / {_P3:,}", flush=True)

    # ── 3 phases wrapped so a KeyboardInterrupt still saves the model ──────────
    try:
        if _done < _P1:
            remaining = _P1 - _done
            print(f"[INFO] Fase 1 — Exploración ({remaining:,} pasos restantes)...")
            model.ent_coef = 0.05
            model.learn(total_timesteps=remaining, callback=callbacks,
                        reset_num_timesteps=(_done == 0))

        if not os.path.exists(flag_path) and _done < _P2:
            remaining = _P2 - max(_done, _P1)
            print(f"[INFO] Fase 2 — Refinamiento ({remaining:,} pasos restantes)...")
            model.ent_coef = 0.03
            model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)

        if not os.path.exists(flag_path) and _done < _P3:
            remaining = _P3 - max(_done, _P2)
            print(f"[INFO] Fase 3 — Endurecimiento ({remaining:,} pasos restantes)...")
            model.ent_coef = 0.02
            model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento interrumpido (Ctrl+C). Guardando progreso actual...", flush=True)
        _save_progress(" INTERRUPT")
        print("✅ Progreso guardado. Puedes reanudar relanzando el script.", flush=True)
        return

    _save_progress()
    print(f"🧬 ¡ENTRENAMIENTO COMPLETADO! Modelo guardado en "
          f"'{os.path.relpath(model_save_path, root_dir)}'.", flush=True)
    print(f"📊 Estrategia EMA200+MACD integrada en el núcleo neuronal.", flush=True)
    
    if os.path.exists(flag_path):
        os.remove(flag_path)
        
    print("Generando gráfico de progreso del entrenamiento completo...", flush=True)
    eval_log_path = os.path.join(run_logs, "evaluations.npz")
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
            if timesteps[-1] > _P1:
                plt.axvline(x=_P1, color='orange', linestyle=':',
                            label=f'Fin Fase 1 ({_P1:,})')
            if timesteps[-1] > _P2:
                plt.axvline(x=_P2, color='green', linestyle=':',
                            label=f'Fin Fase 2 ({_P2:,})')
            plt.title(f"TITANY AI — Progreso de Entrenamiento ({int(timesteps[-1]):,} pasos)")
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
            reports_dir = (os.path.join(out_dir, "reports") if args.out_dir
                           else os.path.join(root_dir, "reports"))
            os.makedirs(reports_dir, exist_ok=True)
            report_img_path = os.path.join(reports_dir, "reporte_entrenamiento_pro_max.png")
            plt.savefig(report_img_path, dpi=150)
            print(f"✅ Gráfico guardado en '{os.path.relpath(report_img_path, root_dir)}'")
            df_curves = pd.DataFrame({
                'Timestep': timesteps,
                'Mean_Reward': mean_rewards,
                'Std_Reward': std_rewards,
            })
            if ep_lengths is not None:
                df_curves['Mean_Episode_Length'] = ep_lengths.mean(axis=1)
            report_csv_path = os.path.join(reports_dir, "reporte_equity_data.csv")
            df_curves.to_csv(report_csv_path, index=False)
            print(f"✅ Datos guardados en '{os.path.relpath(report_csv_path, root_dir)}'")
            graph_generated = True          # ← mark success so fallback is skipped
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
        fallback_reports = (os.path.join(out_dir, "reports") if args.out_dir
                            else os.path.join(root_dir, "reports"))
        os.makedirs(fallback_reports, exist_ok=True)
        fallback_img = os.path.join(fallback_reports, "reporte_entrenamiento_pro_max.png")
        plt.savefig(fallback_img, dpi=150)
        print(f"✅ Gráfico fallback guardado ({len(test_res)} velas)")
if __name__ == "__main__":
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
