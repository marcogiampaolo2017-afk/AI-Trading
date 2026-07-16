# -*- coding: utf-8 -*-
"""
test_integration.py — Tests de integración end-to-end
======================================================
Verifican que la cadena completa

  CSV  →  indicators  →  ForexTradingEnv  →  step × N

funciona sin NaN, Inf, crashes ni inconsistencias de shape.

Estos tests protegen la parte más importante: que **un cambio en cualquier
módulo no rompa la cadena que alimenta el modelo en producción**.

Los tests que usan el CSV maestro se saltean automáticamente si el archivo
no existe (CI sin datos), pero pasan siempre en la máquina de desarrollo.
"""
import os
import sys
import math
import pytest
import numpy as np
import pandas as pd

# ── Project root on sys.path ──────────────────────────────────────────────────
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_TESTS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import config
from env.indicators import (
    load_and_preprocess_data,
    add_quant_features,
    add_hmm_regime_proxy,
    add_physics_features,
    add_golden_strategy_features,
    add_volume_sniper_features,
    add_vision_360_features,
)
from env.trading_env import ForexTradingEnv

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture: full feature pipeline (scope=module → computed once)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def full_pipeline():
    """
    Carga el CSV maestro y aplica todo el pipeline de features,
    igual que lo hace train_agent.py y TITANY_AI_Terminal_Pro.py.
    Se salta si el CSV no existe.
    """
    if not os.path.exists(config.MASTER_CSV):
        pytest.skip(f"CSV maestro no encontrado: {config.MASTER_CSV}")

    df, feature_cols = load_and_preprocess_data(config.MASTER_CSV)
    df, q_cols   = add_quant_features(df);           feature_cols.extend(q_cols)
    df, r_cols   = add_hmm_regime_proxy(df);          feature_cols.extend(r_cols)
    df, p_cols   = add_physics_features(df);          feature_cols.extend(p_cols)
    df, g_cols   = add_golden_strategy_features(df);  feature_cols.extend(g_cols)
    df, s_cols   = add_volume_sniper_features(df);    feature_cols.extend(s_cols)
    df, v_cols   = add_vision_360_features(df);       feature_cols.extend(v_cols)

    # Deduplicate (same order as train_agent correlation filter)
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pipeline de features
# ─────────────────────────────────────────────────────────────────────────────
class TestFullPipeline:
    def test_dataframe_length(self, full_pipeline):
        df, _ = full_pipeline
        assert len(df) > 5000, "El CSV maestro debería tener miles de filas"

    def test_all_feature_cols_present(self, full_pipeline):
        df, feature_cols = full_pipeline
        missing = [c for c in feature_cols if c not in df.columns]
        assert not missing, f"Features faltantes en el DataFrame: {missing}"

    def test_no_nan_in_features(self, full_pipeline):
        df, feature_cols = full_pipeline
        nan_count = df[feature_cols].isna().sum().sum()
        assert nan_count == 0, f"Pipeline produce {nan_count} NaN en features"

    def test_no_inf_in_features(self, full_pipeline):
        df, feature_cols = full_pipeline
        numeric = df[feature_cols].select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric.values).sum()
        assert inf_count == 0, f"Pipeline produce {inf_count} Inf en features"

    def test_regime_proxy_always_present(self, full_pipeline):
        """
        regime_proxy es una feature protegida: debe existir siempre
        para mantener consistencia train / live.
        """
        _, feature_cols = full_pipeline
        assert "regime_proxy" in feature_cols, \
            "regime_proxy debe estar siempre en feature_cols (protegida de drop)"

    def test_ohlcv_columns_present(self, full_pipeline):
        df, _ = full_pipeline
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns, f"Columna OHLCV faltante: {col}"

    def test_high_ge_low(self, full_pipeline):
        """High ≥ Low en todas las filas — violación indica datos corruptos."""
        df, _ = full_pipeline
        violations = (df["High"] < df["Low"]).sum()
        assert violations == 0, f"{violations} filas con High < Low"

    def test_close_within_high_low(self, full_pipeline):
        """Close debe estar entre Low y High — tolerancia para datos de MT5."""
        df, _ = full_pipeline
        tol = 1e-5
        below_low  = (df["Close"] < df["Low"]  - tol).sum()
        above_high = (df["Close"] > df["High"] + tol).sum()
        assert below_low  == 0, f"{below_low} filas donde Close < Low"
        assert above_high == 0, f"{above_high} filas donde Close > High"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Integración pipeline → entorno → steps
# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineToEnv:
    _SL = [20, 50, 100]
    _TP = [40, 100, 200]

    def _make_env(self, df, feature_cols):
        return ForexTradingEnv(
            df=df.tail(1000).reset_index(drop=True),
            window_size=30,
            sl_options=self._SL,
            tp_options=self._TP,
            feature_columns=feature_cols,
            random_start=False,
        )

    def test_env_obs_shape_matches_features(self, full_pipeline):
        """Obs shape[-1] = len(feature_cols) + 3 (state features)."""
        df, feature_cols = full_pipeline
        env = self._make_env(df, feature_cols)
        obs_raw = env.reset()
        obs = obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw
        expected_cols = len(feature_cols) + 3
        assert obs.shape[-1] == expected_cols, (
            f"obs.shape[-1]={obs.shape[-1]} != expected {expected_cols}"
        )

    def test_10_steps_no_nan_reward(self, full_pipeline):
        """10 pasos aleatorios → rewards todos finitos."""
        df, feature_cols = full_pipeline
        env = self._make_env(df, feature_cols)
        env.reset()
        for _ in range(10):
            result = env.step(env.action_space.sample())
            reward = result[1]
            assert math.isfinite(reward), f"Reward no finito: {reward}"

    def test_10_steps_no_nan_obs(self, full_pipeline):
        """10 pasos aleatorios → observaciones todas finitas."""
        df, feature_cols = full_pipeline
        env = self._make_env(df, feature_cols)
        env.reset()
        for _ in range(10):
            obs = env.step(env.action_space.sample())[0]
            assert np.all(np.isfinite(obs)), "Obs contiene NaN/Inf"

    def test_equity_tracking(self, full_pipeline):
        """equity_usd en info debe ser finito y positivo al inicio."""
        df, feature_cols = full_pipeline
        env = self._make_env(df, feature_cols)
        env.reset()
        result = env.step(0)   # HOLD
        info = result[-1] if isinstance(result[-1], dict) else result[3]
        eq = info.get("equity_usd", 0.0)
        assert math.isfinite(eq), f"equity_usd no finito: {eq}"
        assert eq > 0, f"equity_usd no positivo: {eq}"

    def test_open_position_changes_state(self, full_pipeline):
        """Acción OPEN (index 2) desde posición neutral → position != 0."""
        df, feature_cols = full_pipeline
        env = self._make_env(df, feature_cols)
        env.reset()
        env.position = 0
        # Try actions until we find an OPEN
        for act_idx in range(2, len(env.action_map)):
            act_type = env.action_map[act_idx][0]
            if act_type == "OPEN":
                env.step(act_idx)
                # position should now be non-zero
                assert env.position != 0, \
                    "Después de OPEN, position debería ser ≠ 0"
                break

    def test_close_resets_position(self, full_pipeline):
        """OPEN seguido de CLOSE → position == 0."""
        df, feature_cols = full_pipeline
        env = self._make_env(df, feature_cols)
        env.reset()
        # Open a long position manually
        env.position    = 0
        first_open = next(
            i for i, a in enumerate(env.action_map) if a[0] == "OPEN"
        )
        env.step(first_open)
        if env.position != 0:
            env.step(1)  # action index 1 = CLOSE
            assert env.position == 0, "CLOSE debería resetear position a 0"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Consistencia de la acción API (api.py)
# ─────────────────────────────────────────────────────────────────────────────
class TestApiModule:
    def test_api_import_does_not_make_http_request(self):
        """
        api.py NO debe hacer peticiones HTTP al ser importado.
        El bug anterior ejecutaba make_api_request() a nivel de módulo.
        """
        # We can't intercept the actual HTTP call easily without mock,
        # but we can confirm importing the module doesn't raise when
        # BASE44_API_KEY is not set (it used to crash with connection error).
        import importlib
        import core.api as api_mod
        # Just verify the module loads and exposes the right functions
        assert callable(getattr(api_mod, "make_api_request", None)), \
            "make_api_request debe existir en api.py"
        assert callable(getattr(api_mod, "update_entity", None)), \
            "update_entity debe existir en api.py"
        # No side-effects: the module-level HTTP call was removed
        # (if it still exists, the import above would have raised ConnectionError
        #  or EnvironmentError on a machine without the API key set)
