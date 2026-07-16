# -*- coding: utf-8 -*-
"""
indicators.py — Motor de Features para TITANY AI
=================================================
Calcula los 26 indicadores técnicos + cuánticos que alimentan la red neuronal.
Protección robusta contra NaN, Inf y division-by-zero.
"""
import numpy as np
import pandas as pd
try:
    import pandas_ta as ta
    _HAS_TA = True
except ImportError:          # CI Ubuntu sin pandas-ta — solo afecta funciones que lo usan
    ta = None
    _HAS_TA = False


# ---------------------------------------------------------------------------
# Helpers de seguridad numérica
# ---------------------------------------------------------------------------
def _safe_div(numerator, denominator, fill=0.0):
    """División segura: reemplaza division-by-zero e Inf por *fill*."""
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(fill).replace([np.inf, -np.inf], fill)


def _sanitize(series_or_df, fill=0.0):
    """Reemplaza NaN e Inf en una Serie o DataFrame."""
    return series_or_df.fillna(fill).replace([np.inf, -np.inf], fill)


# ---------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO
# ---------------------------------------------------------------------------
def load_and_preprocess_data(csv_path: str):
    """
    CARGADOR DE DATOS:
    Lee archivos CSV de MetaTrader (historiales) y los prepara
    limpiando columnas y aplicando toda la matemática técnica.
    """
    df = pd.read_csv(csv_path, sep='\t')
    df.columns = df.columns.str.strip()

    if '<DATE>' in df.columns and '<TIME>' in df.columns:
        df['Gmt time'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    elif 'DATE' in df.columns and 'TIME' in df.columns:
        df['Gmt time'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])

    rename_map = {
        "<OPEN>": "Open", "OPEN": "Open",
        "<HIGH>": "High", "HIGH": "High",
        "<LOW>": "Low",   "LOW": "Low",
        "<CLOSE>": "Close", "CLOSE": "Close",
        "<TICKVOL>": "Volume", "TICKVOL": "Volume",
        "VOL": "Volume"
    }
    df.rename(columns=rename_map, inplace=True)
    df = df.set_index("Gmt time")
    df.sort_index(inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df, feature_cols = add_indicators(df)
    df.dropna(inplace=True)
    return df, feature_cols


# ---------------------------------------------------------------------------
# 2. INDICADORES BASE
# ---------------------------------------------------------------------------
def add_indicators(df):
    """
    LÓGICA TÉCNICA (Indicadores Base):
    Calcula RSI, MACD, ADX, ATR, Bandas de Bollinger y Medias Móviles.
    Son las 'señales clásicas' que todo trader institucional usa.
    """
    df["rsi_14"] = _sanitize(ta.rsi(df["Close"], length=14))
    df["rsi_50"] = _sanitize(ta.rsi(df["Close"], length=50))

    macd = ta.macd(df["Close"])
    if macd is not None:
        col_macd = [c for c in macd.columns if c.startswith("MACD_")][0]
        col_hist = [c for c in macd.columns if c.startswith("MACDh_") or c.startswith("MACDH_")][0]
        df["macd"] = _sanitize(macd[col_macd])
        df["macd_hist"] = _sanitize(macd[col_hist])
    else:
        df["macd"] = 0.0
        df["macd_hist"] = 0.0

    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None:
        col_adx = [c for c in adx.columns if c.startswith("ADX_")][0]
        df["adx_14"] = _sanitize(adx[col_adx])
    else:
        df["adx_14"] = 0.0

    df["atr_14"] = _sanitize(ta.atr(df["High"], df["Low"], df["Close"], length=14))

    bbands = ta.bbands(df["Close"], length=20, std=2)
    if bbands is not None:
        bbu_col = [c for c in bbands.columns if c.startswith("BBU")][0]
        bbl_col = [c for c in bbands.columns if c.startswith("BBL")][0]
        df["bb_upper_diff"] = _sanitize(bbands[bbu_col] - df["Close"])
        df["bb_lower_diff"] = _sanitize(df["Close"] - bbands[bbl_col])
    else:
        df["bb_upper_diff"] = 0.0
        df["bb_lower_diff"] = 0.0

    df["ma_20"] = _sanitize(ta.sma(df["Close"], length=20))
    df["ma_50"] = _sanitize(ta.sma(df["Close"], length=50))
    df["ma_200"] = _sanitize(ta.sma(df["Close"], length=200))
    df["ma_20_slope"] = _sanitize(df["ma_20"].diff())
    df["ma_50_slope"] = _sanitize(df["ma_50"].diff())

    df["close_ma20_diff"] = _safe_div(df["Close"] - df["ma_20"], df["ma_20"])
    df["close_ma50_diff"] = _safe_div(df["Close"] - df["ma_50"], df["ma_50"])
    df["ma_spread"] = _safe_div(df["ma_20"] - df["ma_50"], df["ma_50"])
    df["ma_spread_slope"] = _sanitize(df["ma_spread"].diff())

    df["vol_sma"] = _sanitize(ta.sma(df["Volume"], length=20))
    vol_std = df["Volume"].rolling(20).std().replace(0, 1)
    df["volume_zscore"] = _safe_div(df["Volume"] - df["vol_sma"], vol_std)
    df["pv_divergence"] = _safe_div(df["Close"].diff() * df["Volume"].diff(), df["Volume"])

    feature_cols = [
        "rsi_14", "rsi_50", "adx_14", "atr_14",
        "bb_upper_diff", "bb_lower_diff", "close_ma20_diff",
        "close_ma50_diff", "ma_spread", "ma_spread_slope",
        "volume_zscore", "pv_divergence"
    ]
    return df, feature_cols


# ---------------------------------------------------------------------------
# 3. FEATURES CUÁNTICOS
# ---------------------------------------------------------------------------
def add_quant_features(df):
    """
    Calcula indicadores cuantitativos avanzados (Fractales, VAM, OU)
    para actuar como 'Co-Piloto' matemático del modelo AI.
    """
    n = 20
    change = (df["Close"] - df["Close"].shift(n)).abs()
    path_length = (df["Close"] - df["Close"].shift(1)).abs().rolling(window=n).sum()
    df["quant_fer"] = _safe_div(change, path_length)

    volatility = df["Close"].rolling(window=n).std()
    returns = df["Close"].diff(n)
    sqrt_n_vol = (volatility * np.sqrt(n)).replace(0, np.nan)
    df["quant_vam"] = _sanitize(returns / sqrt_n_vol)

    rolling_mean = df["Close"].rolling(window=50).mean()
    rolling_std = df["Close"].rolling(window=50).std().replace(0, np.nan)
    df["quant_zscore"] = _sanitize((df["Close"] - rolling_mean) / rolling_std)

    rolling_mean_10 = df["Close"].rolling(window=10).mean()
    rolling_std_10 = df["Close"].rolling(window=10).std().replace(0, np.nan)
    df["quant_zscore_10"] = _sanitize((df["Close"] - rolling_mean_10) / rolling_std_10)

    window = 50
    log_returns = np.log(df["Close"] / df["Close"].shift(1))

    def calc_entropy(x):
        try:
            x_clean = x[np.isfinite(x)]
            if len(x_clean) < 5:
                return 0.5
            hist, _ = np.histogram(x_clean, bins=20, density=True)
            p_k = hist + 1e-10
            p_k = p_k / p_k.sum()
            ent = -np.sum(p_k * np.log(p_k))
            return float(min(1.0, ent / np.log(20)))
        except Exception:
            return 0.5

    df["quant_entropy"] = log_returns.rolling(window=window).apply(calc_entropy)
    df["quant_entropy"] = df["quant_entropy"].fillna(0.5)

    quant_cols = ["quant_fer", "quant_vam", "quant_zscore", "quant_entropy"]
    return df, quant_cols


# ---------------------------------------------------------------------------
# 4. PROXY DE RÉGIMEN (HMM)
# ---------------------------------------------------------------------------
def add_hmm_regime_proxy(df):
    """
    Simula un Hidden Markov Model para detectar 3 regímenes:
    0: Tranquilo, 1: Tendencia, 2: Crisis/Alta Vol.
    """
    vol = df["Close"].pct_change().rolling(20).std()
    mom = df["Close"].pct_change(20)
    regime = pd.Series(0, index=df.index)
    vol_mean = vol.mean()
    mom_std = mom.std()
    if mom_std > 0:
        regime[(mom.abs() > mom_std) & (vol < vol_mean)] = 1
    if vol_mean > 0:
        regime[vol > vol_mean * 2] = 2
    df["regime_proxy"] = regime
    return df, ["regime_proxy"]


# ---------------------------------------------------------------------------
# 5. FEATURES BASADOS EN FÍSICA
# ---------------------------------------------------------------------------
def add_physics_features(df):
    """
    Implementación de Maquinaria de Inferencia (Phase 6).
    Basado en Teoría de Fluidos (Navier-Stokes) y Geometría de la Información.
    """
    velocity = df["Close"].diff()
    acceleration = velocity.diff()
    vol_mean_50 = df["Volume"].rolling(50).mean().replace(0, 1)
    vol_rel = df["Volume"] / vol_mean_50
    df["phys_pressure"] = _sanitize((acceleration * vol_rel).rolling(10).mean())

    df["phys_viscosity"] = _safe_div(df["High"] - df["Low"], df["Volume"])
    df["phys_viscosity"] = _sanitize(df["phys_viscosity"].rolling(20).mean())

    mu_local = df["Close"].pct_change().rolling(10).mean()
    mu_global = df["Close"].pct_change().rolling(50).mean()
    std_global = df["Close"].pct_change().rolling(50).std().replace(0, np.nan)
    df["phys_fisher"] = _sanitize((mu_local - mu_global).abs() / std_global)

    abs_ret = df["Close"].pct_change().abs()
    df["phys_hawkes"] = _sanitize(abs_ret.ewm(halflife=5).mean())

    physics_cols = ["phys_pressure", "phys_viscosity", "phys_fisher", "phys_hawkes"]
    return df, physics_cols


# ---------------------------------------------------------------------------
# 6. GOLDEN STRATEGY (V10)
# ---------------------------------------------------------------------------
def add_golden_strategy_features(df):
    """
    Implementa la Lógica 'Golden Strategy' (V10):
    1. Tendencia EMA 200: Precio por encima/debajo de EMA200 sólida.
    2. MACD Cross: Cruce de la línea MACD sobre Signal en zona favorable.
    """
    if "ma_200" not in df.columns:
        df["ma_200"] = _sanitize(ta.sma(df["Close"], length=200))
    if "macd" not in df.columns:
        macd = ta.macd(df["Close"])
        col_macd = [c for c in macd.columns if c.startswith("MACD_")][0]
        col_sig = [c for c in macd.columns if c.startswith("MACDs_")][0]
        df["macd"] = _sanitize(macd[col_macd])
        df["macd_signal_line"] = _sanitize(macd[col_sig])
    else:
        if "macd_signal_line" not in df.columns:
            macd = ta.macd(df["Close"])
            col_sig = [c for c in macd.columns if c.startswith("MACDs_")][0]
            df["macd_signal_line"] = _sanitize(macd[col_sig])

    above_ema = (df["Open"] > df["ma_200"]) & (df["Close"] > df["ma_200"])
    below_ema = (df["Open"] < df["ma_200"]) & (df["Close"] < df["ma_200"])
    trend_up = above_ema.rolling(window=6).min().fillna(0).astype(int)
    trend_down = below_ema.rolling(window=6).min().fillna(0).astype(int)
    df["golden_trend"] = 0
    df.loc[trend_up == 1, "golden_trend"] = 1
    df.loc[trend_down == 1, "golden_trend"] = -1

    macd_prev = df["macd"].shift(1)
    sig_prev = df["macd_signal_line"].shift(1)
    cross_up = (macd_prev <= sig_prev) & (df["macd"] > df["macd_signal_line"])
    valid_long = cross_up & (df["macd"] < 0) & (df["macd_signal_line"] < 0)
    cross_down = (macd_prev >= sig_prev) & (df["macd"] < df["macd_signal_line"])
    valid_short = cross_down & (df["macd"] > 0) & (df["macd_signal_line"] > 0)
    df["golden_cross"] = 0
    df.loc[valid_long, "golden_cross"] = 1
    df.loc[valid_short, "golden_cross"] = -1

    df["golden_setup"] = 0
    df.loc[(df["golden_trend"] == 1) & (df["golden_cross"] == 1), "golden_setup"] = 1
    df.loc[(df["golden_trend"] == -1) & (df["golden_cross"] == -1), "golden_setup"] = -1

    return df, ["golden_trend", "golden_cross", "golden_setup"]


# ---------------------------------------------------------------------------
# 7. DATA AUGMENTATION
# ---------------------------------------------------------------------------
def augment_data_noise(df, probability=0.3):
    """
    Inyección de Ruido Sintético (Phase 5 - Punto 11)
    """
    mask = np.random.rand(len(df)) < probability
    n_masked = int(mask.sum())
    if n_masked > 0:
        noise = np.random.normal(0, 0.00005, size=n_masked)
        df.loc[mask, "Close"] += noise
    return df


# ---------------------------------------------------------------------------
# 8. VOLUME SNIPER (VSA)
# ---------------------------------------------------------------------------
def add_volume_sniper_features(df):
    """
    Volumen Spread Analysis (VSA). 'Lupa' para Modo Francotirador (V12).
    Busca huellas institucionales enormes analizando la cantidad de volumen vs tamaño de la vela.
    """
    spread = (df["High"] - df["Low"]).replace(0, 0.00001)
    vsa_ratio = df["Volume"] / spread
    vsa_mean = vsa_ratio.rolling(50).mean().replace(0, 1)
    df["vsa_ratio_norm"] = _sanitize(vsa_ratio / vsa_mean)
    df["volume_delta"] = _sanitize(df["Volume"].pct_change())

    avg_vol = df["Volume"].rolling(50).mean()
    avg_spr = spread.rolling(50).mean()
    df["climax_candle"] = ((df["Volume"] > avg_vol * 2.5) & (spread > avg_spr * 1.5)).astype(float)

    sniper_cols = ["vsa_ratio_norm", "volume_delta", "climax_candle"]
    df[sniper_cols] = df[sniper_cols].fillna(0)
    return df, sniper_cols


# ---------------------------------------------------------------------------
# 9. VISIÓN 360 (ICT + Price Action)
# ---------------------------------------------------------------------------
def add_vision_360_features(df):
    """
    VISIÓN 360 DE TRADER PROFESIONAL (ICT + PRICE ACTION)
    Actúa como sobre-escritura neuronal. Detecta patrones institucionales.
    """
    body = df["Close"] - df["Open"]
    abs_body = body.abs()
    full_range = df["High"] - df["Low"]
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]

    prev_body = body.shift(1)
    prev_abs_body = abs_body.shift(1)
    bullish_engulfing = (body > 0) & (prev_body < 0) & (abs_body > prev_abs_body)
    bearish_engulfing = (body < 0) & (prev_body > 0) & (abs_body > prev_abs_body)
    df["pa_engulfing"] = 0
    df.loc[bullish_engulfing, "pa_engulfing"] = 1
    df.loc[bearish_engulfing, "pa_engulfing"] = -1

    # Protección: abs_body puede ser 0 → usar max(abs_body, 1e-10)
    safe_abs_body = abs_body.replace(0, 1e-10)
    bullish_pinbar = (lower_wick >= 2 * safe_abs_body) & (upper_wick <= safe_abs_body)
    bearish_pinbar = (upper_wick >= 2 * safe_abs_body) & (lower_wick <= safe_abs_body)
    df["pa_pinbar"] = 0
    df.loc[bullish_pinbar, "pa_pinbar"] = 1
    df.loc[bearish_pinbar, "pa_pinbar"] = -1

    candle1_high = df["High"].shift(2)
    candle3_low = df["Low"]
    candle1_low = df["Low"].shift(2)
    candle3_high = df["High"]
    bullish_fvg = candle3_low > candle1_high
    bearish_fvg = candle3_high < candle1_low
    df["ict_fvg"] = 0
    df.loc[bullish_fvg, "ict_fvg"] = 1
    df.loc[bearish_fvg, "ict_fvg"] = -1

    rolling_20_high = df["High"].shift(1).rolling(20).max()
    rolling_20_low = df["Low"].shift(1).rolling(20).min()
    sweep_bottom = (df["Low"] < rolling_20_low) & (df["Close"] > rolling_20_low) & (bullish_pinbar)
    sweep_top = (df["High"] > rolling_20_high) & (df["Close"] < rolling_20_high) & (bearish_pinbar)
    df["ict_sweep"] = 0
    df.loc[sweep_bottom, "ict_sweep"] = 1
    df.loc[sweep_top, "ict_sweep"] = -1

    vision_cols = ["pa_engulfing", "pa_pinbar", "ict_fvg", "ict_sweep"]
    df[vision_cols] = df[vision_cols].fillna(0).astype(int)
    return df, vision_cols
