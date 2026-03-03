# trading_env.py

from __future__ import annotations

import numpy as np
import pandas as pd
import math

# Prefer gymnasium if available (SB3 supports it), fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class ForexTradingEnv(gym.Env):
    """
    ENTORNO DE SIMULACIÓN DE TRADING:
    Esta clase define las reglas del juego para la IA:
    - Qué ve (Observación): Velas + Estado de la cuenta.
    - Qué puede hacer (Acciones): Comprar, Vender, Cerrar, Esperar.
    - Qué gana o pierde (Recompensa): Pips obtenidos menos costes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options=None,
        tp_options=None,
        feature_columns = None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,              # cost in pips per round-trip (approx)
        commission_pips: float = 0.0,          # cost in pips per round-trip
        max_slippage_pips: float = 0.0,        # random extra pips (0..max) applied on fills
        lot_size: float = 100000.0,            # 1.0 lot = 100k units (for equity in $)
        reward_scale: float = 1.0,             # optional scaling of rewards
        unrealized_delta_weight: float = 0.02, # shaping weight on delta-unrealized while holding
        random_start: bool = True,
        min_episode_steps: int = 300,          # minimum steps per episode (for random starts)
        episode_max_steps: int | None = None,  # optional cap (truncation)
        feature_mean: np.ndarray | None = None, # optional normalization (train-fitted)
        feature_std: np.ndarray | None = None,  # optional normalization (train-fitted)
        allow_flip: bool = False,               # if True, OPEN while in position flips (close+open). Default False.
        hold_reward_weight: float = 0.005,   # tuned below
        open_penalty_pips: float = 0.1,      # NEW V11: Reduced from 0.5 to stop panic
        time_penalty_pips: float = 0.001,    # NEW V11: Reduced from 0.02 to allow holding long trends
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)  # fallback: everything
        else:
            self.feature_columns = list(feature_columns)

        if sl_options is None or tp_options is None:
            raise ValueError("sl_options and tp_options must be provided (e.g. [15,20,30]).")
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe is too short for the given window_size.")

        self.window_size = int(window_size)
        self.pip_value = float(pip_value)

        # Friction
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)

        # Equity accounting (approx): for EURUSD 1 pip per 1 lot ≈ $10
        # pip_value (price) * lot_size (units) ≈ $ per 1.0 price move.
        # 1 pip = pip_value price move, so $/pip ≈ pip_value * lot_size.
        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size

        # Reward handling
        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)

        # Episode handling
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        # Optional normalization (fit on train only, pass arrays here)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.allow_flip = bool(allow_flip)

        # --- Actions ---
        # 0: HOLD
        # 1: CLOSE
        # 2..: OPEN(direction, sl, tp)
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:  # 0=short, 1=long
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))

        self.action_space = spaces.Discrete(len(self.action_map))

        # Observation features: df columns + 3 state features
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

        # Internal state
        self._reset_state()

    # ----------------------------
    # Core Helpers
    # ----------------------------

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Position state
        self.position = 0              # 0=flat, +1=long, -1=short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        # Accounting
        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd
        self.prev_equity = self.initial_equity_usd # Seguimiento para log-reward (V5)
        self.max_equity_usd = self.initial_equity_usd
        
        # Neuro-Quantum State
        self.synaptic_weight = 1.0 # Plasticidad inicial
        self.prev_drawdown_usd = 0.0
        self.max_equity_usd = self.initial_equity_usd

        # Logging
        self.equity_curve = []
        self.last_trade_info = None

    def _get_state_features(self):
        # position in [-1,0,1], time normalized, unrealized in pips (scaled)
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 1000.0
        unreal_pips = float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
        unreal_scaled = unreal_pips / 100.0  # prevent huge magnitudes
        return np.array([pos, t_norm, unreal_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "Close"])
        if self.position == 1:
            pnl_price = close_price - self.entry_price
        else:
            pnl_price = self.entry_price - close_price
        return pnl_price / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        mean = self.feature_mean.reshape(1, 1, -1)
        std = self.feature_std.reshape(1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        return (obs - mean) / std

    def _get_observation(self):
        start = self.current_step - self.window_size
        if start < 0:
            start = 0

        obs_df = self.df.iloc[start:self.current_step].copy()
        # use only selected feature columns for the agent
        obs_df = obs_df[self.feature_columns]

        # If empty (safety), use the first row repeated
        if len(obs_df) == 0:
            base = np.tile(self.df.iloc[0].values.astype(np.float32), (self.window_size, 1))
        else:
            base = obs_df.values.astype(np.float32)
            if base.shape[0] < self.window_size:
                pad_rows = self.window_size - base.shape[0]
                pad = np.tile(base[0], (pad_rows, 1))
                base = np.vstack([pad, base])

        # Append state features (same for each row)
        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)

        # Optional normalization (only if user passes train-fitted mean/std matching obs dims)
        obs = self._apply_optional_normalization(obs)

        return obs

    def _sample_slippage_pips(self) -> float:
        """Modela deslizamiento realista y latencia (Phase 5 - Punto 13)"""
        if self.max_slippage_pips <= 0: return 0.0
        # Base slippage + latencia aleatoria (10% de probabilidad de spike)
        base = np.random.uniform(0.0, self.max_slippage_pips)
        spike = np.random.uniform(1.0, 5.0) if np.random.rand() > 0.9 else 0.0
        return float(base + spike)

    def _cost_pips_round_trip(self) -> float:
        # Simple friction model (round-trip)
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int, sl_pips: float, tp_pips: float):
        # Entry on current close + slippage; costs applied on close (round-trip model)
        close_price = float(self.df.loc[self.current_step, "Close"])
        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value


        # === VOLATILITY ANCHORING (Pillar 2) ===
        # Escala los pips elegidos por la volatilidad actual (ATR)
        vol_multiplier = 1.0
        if "atr_14" in self.df.columns:
            # Normalizamos respecto a un ATR "promedio" (aprox 0.0010 para EURUSD H1)
            current_atr = float(self.df.loc[self.current_step, "atr_14"])
            vol_multiplier = current_atr / 0.0010
            # Limites de seguridad para evitar extremos
            vol_multiplier = max(0.5, min(2.5, vol_multiplier))
            
        adj_sl_pips = sl_pips * vol_multiplier
        adj_tp_pips = tp_pips * vol_multiplier

        if direction == 1:  # long
            entry = close_price + slip_price
            sl_price = entry - adj_sl_pips * self.pip_value
            tp_price = entry + adj_tp_pips * self.pip_value
            self.position = 1
        else:               # short
            entry = close_price - slip_price
            sl_price = entry + adj_sl_pips * self.pip_value
            tp_price = entry - adj_tp_pips * self.pip_value
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price
        }

    def _close_position(self, reason: str, exit_price: float):
        # Realized pips
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value

        # Costs in pips (round-trip)
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips

        # === NUEVO V11: CURA CONTRA EL SÍNDROME DE PÁNICO ===
        # Modificamos el "reward" de salida para enseñarle lo que está mal.
        adjusted_reward = net_pips

        if reason == "MANUAL_CLOSE":
            # Castigo brutal si cierra manualmente una operación en menos de 5 horas ganando migajas (< 10 pips)
            # Esto evita el comportamiento que vimos donde abría y cerraba en la siguiente vela solo por 2 pips.
            if self.time_in_trade < 5 and net_pips < 10:
                adjusted_reward -= 15.0  # Fuerte multa por ser cobarde

            # Premio extra por paciencia: Si la cerró ganando bien (>20 pips) y la aguantó más de 10 horas
            if self.time_in_trade >= 10 and net_pips >= 20:
                adjusted_reward += (self.time_in_trade * 0.5) # Bono de "Surfeador"
        
        # Premio épico si cierra por Take Profit (fomenta confiar en sus SL/TP)
        if reason == "TP_HIT":
            adjusted_reward += (net_pips * 0.2)

        # Update equity in USD
        self.equity_usd += net_pips * self.usd_per_pip

        # === NEURO-PLASTICITY UPDATE ===
        # Matches live bot: Win = Strength, Loss = Depression
        if net_pips > 0:
            self.synaptic_weight = min(1.5, self.synaptic_weight * 1.05)
        else:
            self.synaptic_weight = max(0.5, self.synaptic_weight * 0.95)

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }

        # Reset position state
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        self.last_trade_info = trade_info
        return adjusted_reward

    def _check_sl_tp_intrabar_and_maybe_close(self) -> float:
        """
        Checks SL/TP on the *next bar* range [Low, High].
        Conservative rule if both touched: assume SL hits first (worst case).
        Returns realized net pips if closed; otherwise None.
        """
        if self.position == 0:
            return None

        # If last bar, close on close
        if self.current_step >= self.n_steps - 2:
            exit_price = float(self.df.loc[self.current_step, "Close"])
            net_pips = self._close_position("END_OF_DATA", exit_price)
            return net_pips

        next_high = float(self.df.loc[self.current_step + 1, "High"])
        next_low = float(self.df.loc[self.current_step + 1, "Low"])

        if self.position == 1:
            sl_hit = next_low <= self.sl_price
            tp_hit = next_high >= self.tp_price
            if sl_hit and tp_hit:
                # conservative: SL first
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            sl_hit = next_high >= self.sl_price
            tp_hit = next_low <= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)

        return None

    # ----------------------------
    # Gym API
    # ----------------------------

    def _calculate_phi_score(self) -> float:
        """
        Proxy de Integrated Information (Φ). 
        Mide la cohesión de los indicadores. Si están dispersos/contradictorios, Φ es bajo.
        """
        try:
            # Seleccionamos una muestra de features recientes
            start = max(0, self.current_step - 10)
            sample = self.df[self.feature_columns].iloc[start:self.current_step]
            # Correlación cruzada entre indicadores
            corr = sample.corr().abs().values
            # Φ Proxy = Promedio de correlación (cohesión informativa)
            phi = np.mean(corr[np.triu_indices(corr.shape[0], k=1)])
            return float(phi) if not np.isnan(phi) else 0.5
        except:
            return 0.5

    def _causal_what_if_simulator(self) -> bool:
        """
        Proxy de Razonador Causal GPT-5.
        Genera un escenario "What-If" de volatilidad brusca.
        """
        if "atr_14" not in self.df.columns: return True
        
        current_atr = float(self.df.loc[self.current_step, "atr_14"])
        
        # Simulamos un spike de 3 sigmas (evento cisne negro local)
        spike_risk = current_atr * 3
        # Si el spike toca un nivel de liquidación teórica, marcamos riesgo alto
        # (Supongamos un Stop Loss promedio de 30 pips)
        if spike_risk > (30 * self.pip_value):
            return False # Riesgo causal demasiado alto
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_state()

        # Choose start
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(np.random.randint(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        obs = self._get_observation()

        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            # If someone steps after done, just return current obs with 0 reward
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1

        # Reward components
        reward_pips = 0.0
        info = {}

        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]

        # 1) Apply action logic
        
        # === NEURO-QUANTUM REGIME FILTERS (STOCHASTIC GRADIENTS) ===
        # Phase 6: No longer binary blocks. We apply reward "friction"
        chaos_penalty = 0.0
        if act_type == "OPEN":
            # Check FER (Fractal Efficiency)
            if "quant_fer" in self.df.columns:
                fer = float(self.df.loc[self.current_step, "quant_fer"])
                zscore = float(self.df.loc[self.current_step, "quant_zscore"]) if "quant_zscore" in self.df.columns else 0.0
                if fer < 0.45 and abs(zscore) < 2.0:
                    chaos_penalty += 0.2  # Soft penalty for trading in noise
            
            # Check Φ-Score & Causal Multi-Risk
            phi_score = self._calculate_phi_score()
            is_causally_safe = self._causal_what_if_simulator()
            if phi_score < 0.3 or not is_causally_safe:
                chaos_penalty += 0.3

        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                # Close at current close (with slippage)
                close_price = float(self.df.loc[self.current_step, "Close"])
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)

        elif act_type == "OPEN":
            if self.position == 0:
                self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                reward_pips -= (self.open_penalty_pips + chaos_penalty)
            else:
                if self.allow_flip and self.position != direction:
                    close_price = float(self.df.loc[self.current_step, "Close"])
                    reward_pips += self._close_position("FLIP_CLOSE", close_price)
                    self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                    reward_pips -= (self.open_penalty_pips + chaos_penalty)

        # === IDLE PENALTY (Phase 3 - Punto 5) ===
        # Si hay tendencia clara (FER > 0.6) y estamos fuera, penalizamos inactividad
        if self.position == 0 and "quant_fer" in self.df.columns:
            fer = float(self.df.loc[self.current_step, "quant_fer"])
            if fer > 0.6:
                reward_pips -= 0.005 # Pequeño "pinchazo" para motivar trade


        # 2) If position is open, check SL/TP on next bar intrabar
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            reward_pips += realized_now

        # 3) If still open, apply reward shaping based on delta-unrealized pips
        # 3) If still open, apply reward shaping
        if self.position != 0:
            self.time_in_trade += 1

            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips

            # === GEOMETRIC REWARD SHAPING (Phase 6 - True Equity) ===
            # Recompensa basada en crecimiento logarítmico del Equity Total (Balance + Flotante)
            total_equity_now = self.equity_usd + unreal_now * self.usd_per_pip
            log_equity_diff = math.log(max(1.0, total_equity_now)) - math.log(max(1.0, self.prev_equity))
            reward_pips += log_equity_diff * 1500 # Agresividad aumentada para V6
            
            # self.prev_equity = total_equity_now # Actualizamos al final del step
            
            # === DRAWDOWN PROTECTION (Phase 7 - Survival) ===
            # Penalizamos si el equity actual cae por debajo del máximo alcanzado
            if total_equity_now < self.max_equity_usd:
                dd_ratio = (self.max_equity_usd - total_equity_now) / self.max_equity_usd
                # Penalización exponencial si el DD supera el 10%
                if dd_ratio > 0.10:
                    reward_pips -= (dd_ratio * 10.0) 
            
            # Actualizamos High Water Mark
            if total_equity_now > self.max_equity_usd:
                self.max_equity_usd = total_equity_now

            self.prev_unrealized_pips = unreal_now
        else:
            # Si estamos fuera, el equity total es solo el balance
            total_equity_now = self.equity_usd
            if total_equity_now > self.max_equity_usd:
                self.max_equity_usd = total_equity_now
        
        # 4) Advance time
        self.current_step += 1
        
        # 5) FINAL UPDATE: Sincronización de Equity para el próximo log(diff)
        self.prev_equity = total_equity_now

        # 5) Termination / truncation
        if self.current_step >= self.n_steps - 1:
            self.terminated = True

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # 6) Log Total Equity (Phase 6 Fix - No more flat lines)
        unreal_final = self._compute_unrealized_pips() if self.position != 0 else 0.0
        current_total_equity = float(self.equity_usd + unreal_final * self.usd_per_pip)
        self.equity_curve.append(current_total_equity)

        # 7) Build observation
        obs = self._get_observation()

        # 8) Final reward scaling (Pure Pips)
        reward = float(reward_pips) * self.reward_scale

        # 9) Info
        info.update({
            "equity_usd": current_total_equity,
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "last_trade_info": self.last_trade_info
        })

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = bool(self.terminated or self.truncated)
            return obs, reward, done, info


    def render(self):
        print(
            f"Step={self.current_step} | Equity=${self.equity_usd:,.2f} | "
            f"Pos={self.position} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}"
        )
