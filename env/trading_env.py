from __future__ import annotations
import numpy as np
import pandas as pd
import math
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
        spread_pips: float = 1.0,                                                    
        commission_pips: float = 0.0,                                       
        max_slippage_pips: float = 0.0,                                                     
        sl_tp_pairs=None,               # list of (sl_pips, tp_pips) tuples; if None falls back to sl_options×tp_options
        lot_size: float = 1000.0,       # 0.01 micro-lot notional (pip=$0.10)
        initial_equity_usd: float = 110.0,  # ~100 EUR realistic starting capital
        reward_scale: float = 1.0,                                          
        unrealized_delta_weight: float = 0.02,                                                   
        random_start: bool = True,
        min_episode_steps: int = 300,                                                         
        episode_max_steps: int | None = None,                             
        feature_mean: np.ndarray | None = None,                                        
        feature_std: np.ndarray | None = None,                                         
        allow_flip: bool = False,                                                                                   
        hold_reward_weight: float = 0.005,                
        open_penalty_pips: float = 0.1,                                               
        time_penalty_pips: float = 0.001,                                                             
        teacher_mode: bool = False,
        max_drawdown_fraction: float = 0.30,  # truncate episode if equity drops >30%
        allow_manual_close: bool = False,     # if False, removes CLOSE action; SL/TP only
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        if feature_columns is None:
            self.feature_columns = list(self.df.columns)                        
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
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)
        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size
        self._initial_equity_param = float(initial_equity_usd)
        self.reward_scale = float(reward_scale)
        self.max_drawdown_fraction = float(max_drawdown_fraction)
        self.allow_manual_close = bool(allow_manual_close)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)
        self.teacher_mode = bool(teacher_mode)
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.allow_flip = bool(allow_flip)
        # CLOSE action: only added when allow_manual_close=True.
        # When False the model can only HOLD or OPEN; all exits are via SL/TP.
        # This prevents the "cut winners short, let losers run" pattern that
        # caused TP to be hit only 1% of the time in the previous backtest.
        self.action_map = [("HOLD", None, None, None)]
        if self.allow_manual_close:
            self.action_map.append(("CLOSE", None, None, None))
        if sl_tp_pairs is not None:
            for direction in [0, 1]:
                for sl, tp in sl_tp_pairs:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))
        else:
            # Legacy fallback: Cartesian product
            for direction in [0, 1]:
                for sl in self.sl_options:
                    for tp in self.tp_options:
                        self.action_map.append(("OPEN", direction, float(sl), float(tp)))
        self.action_space = spaces.Discrete(len(self.action_map))
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )
        self._reset_state()
    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False
        self.position = 0                                         
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.initial_equity_usd = self._initial_equity_param
        self.equity_usd = self.initial_equity_usd
        self.prev_equity = self.initial_equity_usd                                   
        self.max_equity_usd = self.initial_equity_usd
        self.synaptic_weight = 1.0                      
        self.prev_drawdown_usd = 0.0
        self.max_equity_usd = self.initial_equity_usd
        self.equity_curve = []
        self.last_trade_info = None
    def _get_state_features(self):
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 1000.0
        unreal_pips = float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
        unreal_scaled = unreal_pips / 100.0                           
        return np.array([pos, t_norm, unreal_scaled], dtype=np.float32)
    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "Close"])
        if self.position == 1:
            pnl_price = close_price - self.entry_price
        else:
            pnl_price = self.entry_price - close_price
        # Guard against pip_value being 0 (floating-point safety)
        pip_val = self.pip_value if self.pip_value > 1e-10 else 1e-4
        return pnl_price / pip_val
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
        obs_df = obs_df[self.feature_columns]
        if len(obs_df) == 0:
            base = np.tile(self.df.iloc[0].values.astype(np.float32), (self.window_size, 1))
        else:
            base = obs_df.values.astype(np.float32)
            if base.shape[0] < self.window_size:
                pad_rows = self.window_size - base.shape[0]
                pad = np.tile(base[0], (pad_rows, 1))
                base = np.vstack([pad, base])
        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)
        obs = self._apply_optional_normalization(obs)
        return obs
    def _sample_slippage_pips(self) -> float:
        """Modela deslizamiento realista y latencia (Phase 5 - Punto 13)"""
        if self.max_slippage_pips <= 0: return 0.0
        base = np.random.uniform(0.0, self.max_slippage_pips)
        spike = np.random.uniform(1.0, 5.0) if np.random.rand() > 0.9 else 0.0
        return float(base + spike)
    def _cost_pips_round_trip(self) -> float:
        return self.spread_pips + self.commission_pips
    def _open_position(self, direction: int, sl_pips: float, tp_pips: float):
        close_price = float(self.df.loc[self.current_step, "Close"])
        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value
        vol_multiplier = 1.0
        if "atr_14" in self.df.columns:
            current_atr = float(self.df.loc[self.current_step, "atr_14"])
            vol_multiplier = current_atr / 0.0010
            vol_multiplier = max(0.5, min(2.5, vol_multiplier))
        adj_sl_pips = sl_pips * vol_multiplier
        adj_tp_pips = tp_pips * vol_multiplier
        if direction == 1:        
            entry = close_price + slip_price
            sl_price = entry - adj_sl_pips * self.pip_value
            tp_price = entry + adj_tp_pips * self.pip_value
            self.position = 1
        else:                      
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
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips
        adjusted_reward = net_pips
        if reason == "MANUAL_CLOSE":
            if self.time_in_trade < 5 and net_pips < 10:
                adjusted_reward -= 15.0                                
            if self.time_in_trade >= 10 and net_pips >= 20:
                adjusted_reward += (self.time_in_trade * 0.5)                      
        if reason == "TP_HIT":
            # Strong bonus: reward the model for letting winners run to TP.
            # Previous bonus (×0.2) was too weak — TP was hit only 1% of trades.
            adjusted_reward += (net_pips * 1.0)   # double the TP reward
        if net_pips < 0:
            adjusted_reward = net_pips * 1.5
        self.equity_usd += net_pips * self.usd_per_pip
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
    def _calculate_phi_score(self) -> float:
        """
        Proxy de Integrated Information (Φ). 
        Mide la cohesión de los indicadores. Si están dispersos/contradictorios, Φ es bajo.
        """
        try:
            start = max(0, self.current_step - 10)
            sample = self.df[self.feature_columns].iloc[start:self.current_step]
            corr = sample.corr().abs().values
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
        spike_risk = current_atr * 3
        if spike_risk > (30 * self.pip_value):
            return False                               
        return True
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
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
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}
        self.steps_in_episode += 1
        reward_pips = 0.0
        info = {}
        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]
        t_vsa = float(self.df.loc[self.current_step, "vsa_ratio_norm"]) if "vsa_ratio_norm" in self.df.columns else 1.0
        t_fer = float(self.df.loc[self.current_step, "quant_fer"]) if "quant_fer" in self.df.columns else 0.5
        t_z   = float(self.df.loc[self.current_step, "quant_zscore"]) if "quant_zscore" in self.df.columns else 0.0
        if self.teacher_mode:
            if act_type == "OPEN" and self.position == 0:
                vsa_ok = t_vsa > 1.1
                lazarus_ok = (t_fer >= 0.30 or abs(t_z) >= 1.2)
                if not vsa_ok or not lazarus_ok:
                    # Reduced from -20 to -5: the old penalty was so strong the
                    # model made only 1 trade per 241 candles (way too selective).
                    # A softer penalty still guides toward quality entries while
                    # allowing enough trade frequency to learn from PnL feedback.
                    reward_pips -= 5.0
                else:
                    reward_pips += 5.0

            if self.position != 0:
                cur_unreal = self._compute_unrealized_pips()
                # Si el volumen/fuerza mueren y tenemos ganancias, debemos CERRAR.
                if cur_unreal > 5.0 and t_vsa < 1.0 and t_fer < 0.25:
                    if act_type == "HOLD":
                        reward_pips -= 5.0                                 
                    elif act_type == "CLOSE":
                        reward_pips += 10.0                                              
                # Si el momento es enormemente fuerte, queremos HACER HOLD y exprimirlo.
                if t_vsa > 1.3 or t_fer > 0.45:
                    if act_type == "HOLD":
                        reward_pips += 5.0                                     
                    elif act_type == "CLOSE":
                        reward_pips -= 10.0                                                      
                m_p = getattr(self, "max_observed_pips", 0)
                self.max_observed_pips = max(m_p, cur_unreal)
                if self.max_observed_pips > 25.0 and cur_unreal < 5.0:
                    reward_pips -= 15.0                                            
            else:
                self.max_observed_pips = 0
        chaos_penalty = 0.0
        if act_type == "OPEN":
            phi_score = self._calculate_phi_score()
            is_causally_safe = self._causal_what_if_simulator()
            if phi_score < 0.3 or not is_causally_safe:
                chaos_penalty += 0.3
            # Encourage good R:R: bonus for TP >= 3×SL, neutral for 2×SL, small
            # penalty for anything tighter (should not happen with SL_TP_PAIRS
            # but left as safety net for the legacy Cartesian-product fallback).
            if sl_pips is not None and tp_pips is not None and sl_pips > 0:
                rr = tp_pips / sl_pips
                if rr >= 3.0:
                    chaos_penalty -= 0.5   # bonus: 3:1+ R:R
                elif rr < 1.5:
                    chaos_penalty += 2.0   # hard penalty: below 1.5:1
            if self.position == 0:
                self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                reward_pips -= (self.open_penalty_pips + chaos_penalty)
            elif self.allow_flip and self.position != direction:
                close_price = float(self.df.loc[self.current_step, "Close"])
                reward_pips += self._close_position("FLIP_CLOSE", close_price)
                self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                reward_pips -= (self.open_penalty_pips + chaos_penalty)
        elif act_type == "CLOSE":
            if self.position != 0:
                close_price = float(self.df.loc[self.current_step, "Close"])
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)
        if self.position == 0 and t_fer > 0.5:
            reward_pips -= 0.5                                              
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            reward_pips += realized_now
        if self.position != 0:
            self.time_in_trade += 1
            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips
            total_equity_now = self.equity_usd + unreal_now * self.usd_per_pip
            # Use a linear scaling for continuous reward based on unreleased profit rather than volatile log.
            reward_pips += delta_unreal * getattr(self, "unrealized_delta_weight", 0.1)                                
            if total_equity_now < self.max_equity_usd:
                dd_ratio = (self.max_equity_usd - total_equity_now) / self.max_equity_usd
                if dd_ratio > 0.10:
                    reward_pips -= (dd_ratio * 10.0) 
            if total_equity_now > self.max_equity_usd:
                self.max_equity_usd = total_equity_now
            self.prev_unrealized_pips = unreal_now
        else:
            total_equity_now = self.equity_usd
            if total_equity_now > self.max_equity_usd:
                self.max_equity_usd = total_equity_now
        self.current_step += 1
        self.prev_equity = total_equity_now
        if self.current_step >= self.n_steps - 1:
            self.terminated = True
        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True
        # Circuit breaker: protect the real account from catastrophic losses.
        # If equity drops below (1 - max_drawdown_fraction) × initial, end now.
        if total_equity_now < self.initial_equity_usd * (1.0 - self.max_drawdown_fraction):
            self.truncated = True
        unreal_final = self._compute_unrealized_pips() if self.position != 0 else 0.0
        current_total_equity = float(self.equity_usd + unreal_final * self.usd_per_pip)
        self.equity_curve.append(current_total_equity)
        obs = self._get_observation()
        reward = float(reward_pips) * self.reward_scale
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
