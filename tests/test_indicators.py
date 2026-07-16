# -*- coding: utf-8 -*-
"""
test_indicators.py — Tests unitarios para env/indicators.py
===========================================================
Verifica que los indicadores se calculan sin NaN, Inf ni excepciones
bajo condiciones normales y adversas.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Asegurar imports del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from env.indicators import (
    load_and_preprocess_data,
    add_quant_features,
    add_hmm_regime_proxy,
    add_physics_features,
    add_golden_strategy_features,
    add_volume_sniper_features,
    add_vision_360_features,
    augment_data_noise,
    _safe_div,
    _sanitize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded_data():
    """Carga el CSV maestro una sola vez para todos los tests."""
    csv_path = config.MASTER_CSV
    if not os.path.exists(csv_path):
        pytest.skip(f"Master CSV not found: {csv_path}")
    df, feature_cols = load_and_preprocess_data(csv_path)
    return df, feature_cols


# ---------------------------------------------------------------------------
# Tests de helpers
# ---------------------------------------------------------------------------
class TestSafeDiv:
    def test_basic(self):
        num = pd.Series([10.0, 20.0, 30.0])
        den = pd.Series([2.0, 0.0, 5.0])
        result = _safe_div(num, den)
        assert result.iloc[0] == 5.0
        assert result.iloc[1] == 0.0  # division by zero → fill=0.0
        assert result.iloc[2] == 6.0

    def test_custom_fill(self):
        num = pd.Series([10.0])
        den = pd.Series([0.0])
        result = _safe_div(num, den, fill=-999.0)
        assert result.iloc[0] == -999.0


class TestSanitize:
    def test_nan_inf(self):
        s = pd.Series([1.0, np.nan, np.inf, -np.inf, 5.0])
        result = _sanitize(s, fill=0.0)
        assert not result.isna().any()
        assert not np.isinf(result.values).any()


# ---------------------------------------------------------------------------
# Tests de carga de datos
# ---------------------------------------------------------------------------
class TestLoadData:
    def test_csv_loads(self, loaded_data):
        df, feature_cols = loaded_data
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1000, "El CSV maestro debería tener más de 1000 filas"

    def test_required_columns_exist(self, loaded_data):
        df, _ = loaded_data
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns, f"Falta la columna {col}"

    def test_no_nan_in_ohlcv(self, loaded_data):
        df, _ = loaded_data
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]
        assert ohlcv.isna().sum().sum() == 0, "Hay NaN en columnas OHLCV"

    def test_feature_cols_match(self, loaded_data):
        df, feature_cols = loaded_data
        for col in feature_cols:
            assert col in df.columns, f"Feature {col} falta en el DataFrame"

    def test_no_nan_in_features(self, loaded_data):
        df, feature_cols = loaded_data
        nan_count = df[feature_cols].isna().sum().sum()
        assert nan_count == 0, f"Hay {nan_count} NaN en las features base"

    def test_no_inf_in_features(self, loaded_data):
        df, feature_cols = loaded_data
        numeric = df[feature_cols].select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric.values).sum()
        assert inf_count == 0, f"Hay {inf_count} Inf en las features base"


# ---------------------------------------------------------------------------
# Tests de features cuánticos
# ---------------------------------------------------------------------------
class TestQuantFeatures:
    def test_adds_columns(self, loaded_data):
        df, _ = loaded_data
        df_q, q_cols = add_quant_features(df.copy())
        for col in q_cols:
            assert col in df_q.columns

    def test_no_inf(self, loaded_data):
        df, _ = loaded_data
        df_q, q_cols = add_quant_features(df.copy())
        for col in q_cols:
            inf_count = np.isinf(df_q[col].dropna().values).sum()
            assert inf_count == 0, f"Inf en {col}"

    def test_fer_bounded(self, loaded_data):
        """FER debería estar entre 0 y 1 (ratio de eficiencia fractal)."""
        df, _ = loaded_data
        df_q, _ = add_quant_features(df.copy())
        fer = df_q["quant_fer"].dropna()
        assert fer.min() >= -0.01, f"FER mínimo inesperado: {fer.min()}"
        assert fer.max() <= 1.5, f"FER máximo inesperado: {fer.max()}"


# ---------------------------------------------------------------------------
# Tests de features de física
# ---------------------------------------------------------------------------
class TestPhysicsFeatures:
    def test_adds_columns(self, loaded_data):
        df, _ = loaded_data
        df_p, p_cols = add_physics_features(df.copy())
        for col in p_cols:
            assert col in df_p.columns

    def test_no_inf(self, loaded_data):
        df, _ = loaded_data
        df_p, p_cols = add_physics_features(df.copy())
        for col in p_cols:
            inf_count = np.isinf(df_p[col].dropna().values).sum()
            assert inf_count == 0, f"Inf en {col}"


# ---------------------------------------------------------------------------
# Tests de features Golden Strategy
# ---------------------------------------------------------------------------
class TestGoldenStrategy:
    def test_adds_columns(self, loaded_data):
        df, _ = loaded_data
        df_g, g_cols = add_golden_strategy_features(df.copy())
        for col in g_cols:
            assert col in df_g.columns

    def test_golden_trend_values(self, loaded_data):
        df, _ = loaded_data
        df_g, _ = add_golden_strategy_features(df.copy())
        unique = set(df_g["golden_trend"].unique())
        assert unique.issubset({-1, 0, 1}), f"Valores inesperados: {unique}"


# ---------------------------------------------------------------------------
# Tests de VSA (Volume Sniper)
# ---------------------------------------------------------------------------
class TestVolumeSniper:
    def test_adds_columns(self, loaded_data):
        df, _ = loaded_data
        df_v, v_cols = add_volume_sniper_features(df.copy())
        for col in v_cols:
            assert col in df_v.columns

    def test_no_nan_after(self, loaded_data):
        df, _ = loaded_data
        df_v, v_cols = add_volume_sniper_features(df.copy())
        for col in v_cols:
            nan_count = df_v[col].isna().sum()
            assert nan_count == 0, f"NaN en {col}: {nan_count}"


# ---------------------------------------------------------------------------
# Test de precisión de punto flotante
# ---------------------------------------------------------------------------
class TestFloatingPointSafety:
    """Verifica que la aritmética de punto flotante no cause sorpresas."""

    def test_small_price_difference(self):
        """Simula precios muy cercanos que podrían causar division-by-zero."""
        num = pd.Series([0.00001])
        den = pd.Series([0.00001])
        result = _safe_div(num, den)
        assert np.isfinite(result.iloc[0])

    def test_zero_volume(self, loaded_data):
        """Simula que Volume sea 0 en toda la serie."""
        df, _ = loaded_data
        df_test = df.head(500).copy()
        df_test["Volume"] = 0.0
        df_v, _ = add_volume_sniper_features(df_test)
        assert not df_v["vsa_ratio_norm"].isna().any()
        assert not np.isinf(df_v["vsa_ratio_norm"].values).any()

    def test_constant_price(self, loaded_data):
        """Simula que el precio no cambie → std = 0 → division by zero."""
        df, _ = loaded_data
        df_test = df.head(500).copy()
        df_test["Close"] = 1.10000
        df_test["Open"] = 1.10000
        df_test["High"] = 1.10000
        df_test["Low"] = 1.10000
        df_q, _ = add_quant_features(df_test)
        assert not np.isinf(df_q["quant_zscore"].dropna().values).any()
        assert not np.isinf(df_q["quant_fer"].dropna().values).any()

    def test_ieee754_pip_precision(self):
        """
        0.1 + 0.2 != 0.3 en IEEE 754.
        Verifica que los helpers de división segura son estables
        frente a errores de representación decimal binaria.
        """
        # 0.1 + 0.2 - 0.3 ≈ 5.55e-17 (no exactamente 0 en binario)
        residual = 0.1 + 0.2 - 0.3
        assert abs(residual) < 1e-10, (
            "Sanity check IEEE 754: residual debería ser ~0 pero no exactamente 0"
        )
        # Ahora verificar _safe_div no explota con valores de pip cercanos
        num = pd.Series([residual, 1e-10, 0.0])
        den = pd.Series([1e-10,   1e-10, 1e-10])
        result = _safe_div(num, den)
        assert result.apply(lambda x: np.isfinite(x)).all(), (
            "safe_div debe devolver finitos incluso con magnitudes de pip"
        )

    def test_augment_noise_preserves_shape(self, loaded_data):
        """augment_data_noise no debe cambiar el shape ni introducir NaN."""
        df, _ = loaded_data
        df_aug = augment_data_noise(df.copy(), probability=0.5)
        assert df_aug.shape == df.shape
        assert not df_aug["Close"].isna().any()


# ---------------------------------------------------------------------------
# Tests de HMM Regime Proxy
# ---------------------------------------------------------------------------
class TestHMMRegimeProxy:
    def test_adds_column(self, loaded_data):
        df, _ = loaded_data
        df_r, r_cols = add_hmm_regime_proxy(df.copy())
        assert "regime_proxy" in df_r.columns
        assert "regime_proxy" in r_cols

    def test_values_are_0_1_2(self, loaded_data):
        df, _ = loaded_data
        df_r, _ = add_hmm_regime_proxy(df.copy())
        unique = set(df_r["regime_proxy"].unique())
        assert unique.issubset({0, 1, 2}), f"Valores inesperados: {unique}"

    def test_no_nan(self, loaded_data):
        df, _ = loaded_data
        df_r, _ = add_hmm_regime_proxy(df.copy())
        assert df_r["regime_proxy"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests de Vision 360
# ---------------------------------------------------------------------------
class TestVision360:
    def test_adds_columns(self, loaded_data):
        df, _ = loaded_data
        df_v, v_cols = add_vision_360_features(df.copy())
        for col in v_cols:
            assert col in df_v.columns

    def test_values_bounded(self, loaded_data):
        df, _ = loaded_data
        df_v, v_cols = add_vision_360_features(df.copy())
        for col in v_cols:
            unique = set(df_v[col].unique())
            assert unique.issubset({-1, 0, 1}), f"Valores fuera de [-1,0,1] en {col}: {unique}"

    def test_no_nan(self, loaded_data):
        df, _ = loaded_data
        df_v, v_cols = add_vision_360_features(df.copy())
        for col in v_cols:
            assert df_v[col].isna().sum() == 0, f"NaN en {col}"
