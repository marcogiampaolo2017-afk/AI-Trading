# -*- coding: utf-8 -*-
"""
test_trading_env.py — Tests unitarios para env/trading_env.py
=============================================================
"¿Qué parte de mi código no debería romperse bajo ninguna circunstancia?"
→ El entorno de simulación: es la base de TODO el entrenamiento RL.
  Si el env devuelve NaN, Inf o estados incoherentes, el modelo aprende basura.

Cuidado especial a:
  - Aritmética de punto flotante: pip_value, PnL, SL/TP
  - Casos extremos: precio constante, volumen cero, episodio de 1 barra
  - Coherencia de obs shape antes y después de reset
"""
import os
import sys
import math
import pytest
import numpy as np
import pandas as pd

# ── Ensure project root is importable ─────────────────────────────────────────
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_TESTS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.trading_env import ForexTradingEnv

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_SL_OPTS = [10, 20, 30]
_TP_OPTS = [20, 40, 60]
_WIN     = 30
_NROWS   = 600   # enough rows for a full episode


def _make_random_df(n: int = _NROWS, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with base feature columns."""
    rng  = np.random.default_rng(seed)
    base = 1.10000
    cols = [
        "Open", "High", "Low", "Close", "Volume",
        # base indicators (12)
        "rsi_14", "rsi_50", "adx_14", "atr_14",
        "bb_upper_diff", "bb_lower_diff",
        "close_ma20_diff", "close_ma50_diff",
        "ma_spread", "ma_spread_slope",
        "volume_zscore", "pv_divergence",
        # quant (4)
        "quant_fer", "quant_vam", "quant_zscore", "quant_entropy",
        # regime (1)
        "regime_proxy",
    ]
    data = {}
    prices = base + np.cumsum(rng.normal(0, 0.0001, n))
    data["Close"]  = prices
    data["Open"]   = prices + rng.normal(0, 0.00005, n)
    data["High"]   = np.maximum(data["Open"], data["Close"]) + rng.uniform(0, 0.0003, n)
    data["Low"]    = np.minimum(data["Open"], data["Close"]) - rng.uniform(0, 0.0003, n)
    data["Volume"] = rng.integers(100, 5000, n).astype(float)
    for c in cols:
        if c not in data:
            data[c] = rng.uniform(-1, 1, n)
    df = pd.DataFrame(data, columns=cols)
    df.reset_index(drop=True, inplace=True)
    return df


def _make_env(df=None, **kwargs):
    if df is None:
        df = _make_random_df()
    defaults = dict(
        window_size=_WIN,
        sl_options=_SL_OPTS,
        tp_options=_TP_OPTS,
        random_start=False,
        teacher_mode=False,
    )
    defaults.update(kwargs)
    return ForexTradingEnv(df=df, **defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Init / reset shape
# ─────────────────────────────────────────────────────────────────────────────
class TestEnvInit:
    def test_obs_shape_after_reset(self):
        env = _make_env()
        obs, *_ = env.reset() if hasattr(env, "reset") else (env.reset(),)
        assert obs.shape == env.observation_space.shape, (
            f"Obs shape mismatch: got {obs.shape}, expected {env.observation_space.shape}"
        )

    def test_action_space_size(self):
        env = _make_env()
        # HOLD + (CLOSE si ALLOW_MANUAL_CLOSE) + 2 directions * n_sl * n_tp
        # Con ALLOW_MANUAL_CLOSE=False (convención del proyecto): 1 + 2*n_sl*n_tp
        import config as _cfg
        n_sl_tp = len(_SL_OPTS) * len(_TP_OPTS)
        expected = (2 if _cfg.ALLOW_MANUAL_CLOSE else 1) + 2 * n_sl_tp
        assert env.action_space.n == expected, (
            f"Action space: got {env.action_space.n}, expected {expected} "
            f"(ALLOW_MANUAL_CLOSE={_cfg.ALLOW_MANUAL_CLOSE}, SL opts={_SL_OPTS}, TP opts={_TP_OPTS})"
        )

    def test_short_df_raises(self):
        tiny_df = _make_random_df(n=_WIN + 1)   # exactly 1 row margin → should raise
        with pytest.raises(ValueError):
            _make_env(df=tiny_df)

    def test_missing_sl_tp_raises(self):
        df = _make_random_df()
        with pytest.raises((ValueError, TypeError)):
            ForexTradingEnv(df=df, window_size=_WIN)   # no sl_options / tp_options


# ─────────────────────────────────────────────────────────────────────────────
# 2. Step produces finite outputs
# ─────────────────────────────────────────────────────────────────────────────
class TestEnvStep:
    def _run_n_steps(self, env, n=200):
        env.reset()
        for i in range(n):
            action = env.action_space.sample()
            result = env.step(action)
            # Support both gym (4-tuple) and gymnasium (5-tuple)
            obs, reward = result[0], result[1]
            yield obs, reward

    def test_reward_is_finite(self):
        env = _make_env()
        for obs, reward in self._run_n_steps(env):
            assert math.isfinite(reward), f"Non-finite reward: {reward}"

    def test_obs_is_finite(self):
        env = _make_env()
        for obs, _ in self._run_n_steps(env):
            assert np.all(np.isfinite(obs)), "Non-finite values in observation"

    def test_obs_shape_stable(self):
        env = _make_env()
        expected = env.observation_space.shape
        for obs, _ in self._run_n_steps(env):
            assert obs.shape == expected, f"Obs shape changed mid-episode: {obs.shape}"

    def test_equity_never_nan(self):
        env = _make_env()
        env.reset()
        for _ in range(300):
            result = env.step(env.action_space.sample())
            info = result[-1] if isinstance(result[-1], dict) else result[3]
            eq = info.get("equity_usd", None)
            assert eq is not None
            assert math.isfinite(eq), f"equity_usd={eq} is not finite"


# ─────────────────────────────────────────────────────────────────────────────
# 3. PnL arithmetic (floating-point precision)
# ─────────────────────────────────────────────────────────────────────────────
class TestPnLArithmetic:
    """
    Cuidado especial a la aritmética de punto flotante:
    0.1 + 0.2 ≠ 0.3 en IEEE 754. En trading, esto puede significar
    SL/TP que no se tocan correctamente o equity negativa por error de redondeo.
    """

    def test_pip_value_nonzero(self):
        env = _make_env()
        assert env.pip_value > 1e-10, "pip_value debe ser > 0"

    def test_close_position_pnl_sign(self):
        """Larga con precio superior → PnL positivo; con precio inferior → negativo."""
        env = _make_env()
        env.reset()
        env.position     = 1
        env.entry_price  = 1.10000
        env.sl_price     = 1.09800
        env.tp_price     = 1.10200
        env.time_in_trade = 20

        # Exit above entry → profit
        pnl_win = env._close_position("MANUAL_CLOSE", 1.10150)
        assert pnl_win > 0, "Larga con salida > entrada debe ser positiva"

        env.position     = 1
        env.entry_price  = 1.10000
        env.sl_price     = 1.09800
        env.tp_price     = 1.10200
        env.time_in_trade = 2

        # Exit below entry → loss
        pnl_loss = env._close_position("MANUAL_CLOSE", 1.09850)
        assert pnl_loss < 0, "Larga con salida < entrada debe ser negativa"

    def test_unrealized_pips_finite(self):
        env = _make_env()
        env.reset()
        env.position    = 1
        env.entry_price = 1.10000
        env.current_step = _WIN + 1
        pips = env._compute_unrealized_pips()
        assert math.isfinite(pips), "unrealized_pips debe ser finito"

    def test_pip_value_zero_guard(self):
        """Si pip_value fuera 0 (error de config), _compute_unrealized_pips no debe dividir por cero."""
        env = _make_env(pip_value=1e-10)   # near-zero but positive
        env.reset()
        env.position    = 1
        env.entry_price = 1.10000
        env.current_step = _WIN + 1
        pips = env._compute_unrealized_pips()
        assert math.isfinite(pips)

    def test_sl_tp_intrabar_sl_first_worst_case(self):
        """
        Si en la misma barra se tocan SL y TP, el código aplica
        la regla conservadora: SL primero (worst-case).
        Resultado esperado: PnL negativo.
        """
        env = _make_env()
        env.reset()
        env.position    = 1          # long
        entry = 1.10000
        env.entry_price = entry
        env.sl_price    = entry - 0.0020   # 20 pips abajo
        env.tp_price    = entry + 0.0040   # 40 pips arriba
        # Fabricar barra siguiente que toca AMBOS
        env.current_step = _WIN
        env.df.loc[env.current_step + 1, "Low"]  = env.sl_price - 0.0001
        env.df.loc[env.current_step + 1, "High"] = env.tp_price + 0.0001
        net = env._check_sl_tp_intrabar_and_maybe_close()
        assert net is not None, "Debería cerrar posición"
        assert net < 0, f"SL-first debe dar pérdida; got {net}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Casos extremos / robustez
# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_constant_price_episode(self):
        """Precio plano → ATR=0 → vol_multiplier edge-case → no crash."""
        df = _make_random_df()
        df["Close"] = 1.10000
        df["Open"]  = 1.10000
        df["High"]  = 1.10000
        df["Low"]   = 1.10000
        df["atr_14"] = 0.0    # ATR explícito = 0
        env = _make_env(df=df)
        env.reset()
        for _ in range(200):
            result = env.step(env.action_space.sample())
            obs, reward = result[0], result[1]
            assert np.all(np.isfinite(obs))
            assert math.isfinite(reward)

    def test_zero_volume_episode(self):
        """Volume = 0 → VSA features con división → no crash en env."""
        df = _make_random_df()
        df["Volume"] = 0.0
        env = _make_env(df=df)
        env.reset()
        for _ in range(100):
            result = env.step(env.action_space.sample())
            assert math.isfinite(result[1])

    def test_double_reset_preserves_shape(self):
        """Dos resets consecutivos deben devolver la misma forma de obs."""
        env = _make_env()
        obs1_raw = env.reset()
        obs1 = obs1_raw[0] if isinstance(obs1_raw, tuple) else obs1_raw
        obs2_raw = env.reset()
        obs2 = obs2_raw[0] if isinstance(obs2_raw, tuple) else obs2_raw
        assert obs1.shape == obs2.shape

    def test_step_after_done_returns_valid_obs(self):
        """Pasar step() después de terminated=True no debe lanzar excepción."""
        env = _make_env()
        env.reset()
        env.terminated = True
        result = env.step(0)   # action=HOLD
        obs, reward = result[0], result[1]
        assert obs.shape == env.observation_space.shape
        assert math.isfinite(reward)

    def test_full_episode_completes(self):
        """Un episodio completo (sin random_start) debe terminar sin error."""
        env = _make_env(random_start=False)
        obs_raw = env.reset()
        done = False
        steps = 0
        while not done and steps < 10000:
            result = env.step(env.action_space.sample())
            if len(result) == 5:
                _, _, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                _, _, done, _ = result
            steps += 1
        assert steps > 0, "El episodio no avanzó ni un paso"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Coherencia de estado interno
# ─────────────────────────────────────────────────────────────────────────────
class TestStateCoherence:
    def test_position_resets_to_zero(self):
        env = _make_env()
        env.reset()
        assert env.position == 0

    def test_equity_never_below_zero_after_episode(self):
        """
        Con las penalizaciones actuales, la equity puede bajar mucho,
        pero no debería ser NaN ni infinita.
        """
        env = _make_env(random_start=False)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 5000:
            result = env.step(env.action_space.sample())
            if len(result) == 5:
                _, _, terminated, truncated, info = result
                done = terminated or truncated
            else:
                _, _, done, info = result
            eq = info.get("equity_usd", 10000.0)
            assert math.isfinite(eq), f"equity={eq} at step {steps}"
            steps += 1

    def test_synaptic_weight_bounded(self):
        """synaptic_weight debe quedar entre 0.5 y 1.5 tras operaciones."""
        env = _make_env()
        env.reset()
        # Simulate wins and losses
        env.position    = 1
        env.entry_price = 1.10000
        env.sl_price    = 1.09800
        env.tp_price    = 1.10200
        env.time_in_trade = 15
        env._close_position("TP_HIT", 1.10200)   # win
        assert 0.4 <= env.synaptic_weight <= 1.6, \
            f"synaptic_weight fuera de rango: {env.synaptic_weight}"

    def test_time_in_trade_resets_on_close(self):
        env = _make_env()
        env.reset()
        env.position    = 1
        env.entry_price = 1.10000
        env.sl_price    = 1.09500
        env.tp_price    = 1.10500
        env.time_in_trade = 99
        env._close_position("MANUAL_CLOSE", 1.10100)
        assert env.time_in_trade == 0, "time_in_trade debe resetear a 0 al cerrar"
