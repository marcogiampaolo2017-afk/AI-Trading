import pandas as pd
import pandas_ta as ta

def load_and_preprocess_data(csv_path: str):
    """
    CARGADOR DE DATOS:
    Lee archivos CSV de MetaTrader (historiales) y los prepara
    limpiando columnas y aplicando toda la matemática técnica.
    """
    # 1. Leemos el CSV
    df = pd.read_csv(csv_path, sep='\t')

    # 2. Limpieza de nombres de columnas
    df.columns = df.columns.str.strip()

    # 3. Combinar FECHA y HORA manualmente (Evita warnings)
    if '<DATE>' in df.columns and '<TIME>' in df.columns:
        df['Gmt time'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    elif 'DATE' in df.columns and 'TIME' in df.columns:
        df['Gmt time'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    
    # 4. Renombrar columnas al estándar
    rename_map = {
        "<OPEN>": "Open", "OPEN": "Open",
        "<HIGH>": "High", "HIGH": "High",
        "<LOW>": "Low",   "LOW": "Low",
        "<CLOSE>": "Close", "CLOSE": "Close",
        "<TICKVOL>": "Volume", "TICKVOL": "Volume",
        "VOL": "Volume"
    }
    df.rename(columns=rename_map, inplace=True)

    # Configurar índice
    df = df.set_index("Gmt time")
    df.sort_index(inplace=True)

    # Asegurar numéricos
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Calcular Indicadores
    df, feature_cols = add_indicators(df)
    
    # Limpieza final
    df.dropna(inplace=True)

    return df, feature_cols

def add_indicators(df):
    """
    LÓGICA TÉCNICA (Indicadores Base):
    Calcula RSI, MACD, ADX, ATR, Bandas de Bollinger y Medias Móviles.
    Son las 'señales clásicas' que todo trader institucional usa.
    """
    
    # --- RSI ---
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["rsi_50"] = ta.rsi(df["Close"], length=50) # PROXY H4 MOMENTUM
    
    # --- MACD ---
    macd = ta.macd(df["Close"])
    if macd is not None:
        # Buscamos columnas dinámicamente para evitar KeyError
        # MACD suele tener 3 columnas: MACD, Histogram, Signal
        col_macd = [c for c in macd.columns if c.startswith("MACD_")][0]
        col_hist = [c for c in macd.columns if c.startswith("MACDh_") or c.startswith("MACDH_")][0]
        
        df["macd"] = macd[col_macd]
        df["macd_hist"] = macd[col_hist]
    else:
        df["macd"] = 0.0
        df["macd_hist"] = 0.0

    # --- ADX ---
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None:
        # Busca la columna que empieza con ADX_
        col_adx = [c for c in adx.columns if c.startswith("ADX_")][0]
        df["adx_14"] = adx[col_adx]
    else:
        df["adx_14"] = 0.0

    # --- ATR ---
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # --- Bollinger Bands ---
    bbands = ta.bbands(df["Close"], length=20, std=2)
    if bbands is not None:
        # CORRECCIÓN ROBUSTA: No adivinamos el nombre, lo buscamos.
        # Buscamos la columna que empiece por BBU (Upper) y BBL (Lower)
        bbu_col = [c for c in bbands.columns if c.startswith("BBU")][0]
        bbl_col = [c for c in bbands.columns if c.startswith("BBL")][0]
        
        df["bb_upper_diff"] = bbands[bbu_col] - df["Close"]
        df["bb_lower_diff"] = df["Close"] - bbands[bbl_col]
    else:
         df["bb_upper_diff"] = 0.0
         df["bb_lower_diff"] = 0.0

    # --- Medias Móviles ---
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)
    df["ma_200"] = ta.sma(df["Close"], length=200) # PROXY H4 TREND

    # AÑADE ESTAS LÍNEAS PARA CALCULAR LA PENDIENTE:
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()
    
    # Evitar división por cero
    df["close_ma20_diff"] = (df["Close"] - df["ma_20"]) / df["ma_20"].replace(0, 1)
    df["close_ma50_diff"] = (df["Close"] - df["ma_50"]) / df["ma_50"].replace(0, 1)
    
    df["ma_spread"] = (df["ma_20"] - df["ma_50"]) / df["ma_50"].replace(0, 1)
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # --- SENTIMENT & VOLUME PROXIES (Pillar 8) ---
    df["vol_sma"] = ta.sma(df["Volume"], length=20)
    df["volume_zscore"] = (df["Volume"] - df["vol_sma"]) / df["Volume"].rolling(20).std().replace(0, 1)
    # Divergencia Precio/Volumen
    df["pv_divergence"] = (df["Close"].diff() * df["Volume"].diff()) / df["Volume"].replace(0, 1)

    # Filtrar solo lo más potente (Dual-Positive Focus)
    feature_cols = [
        "rsi_14", "rsi_50", "adx_14", "atr_14", 
        "bb_upper_diff", "bb_lower_diff", "close_ma20_diff", 
        "close_ma50_diff", "ma_spread", "ma_spread_slope",
        "volume_zscore", "pv_divergence"
    ]

    return df, feature_cols

def add_quant_features(df):
    """
    Calcula indicadores cuantitativos avanzados (Fractales, VAM, OU)
    para actuar como 'Co-Piloto' matemático del modelo AI.
    """
    import numpy as np
    
    # 1. Fractal Efficiency Ratio (FER) - Chaos Theory
    # Mide si el mercado es tendencia (cerca de 1) o caótico/lateral (cerca de 0)
    n = 20
    change = (df["Close"] - df["Close"].shift(n)).abs()
    path_length = (df["Close"] - df["Close"].shift(1)).abs().rolling(window=n).sum()
    df["quant_fer"] = change / path_length.replace(0, 1)

    # 2. Volatility Adjusted Momentum (VAM)
    # Impulso ajustado por riesgo (basado en Sharpe Ratio local)
    # VAM = (P_t - P_{t-n}) / (sigma * sqrt(n))
    volatility = df["Close"].rolling(window=n).std()
    returns = df["Close"].diff(n)
    df["quant_vam"] = returns / (volatility * np.sqrt(n)).replace(0, 1)

    # 3. Ornstein-Uhlenbeck (Mean Reversion) Z-Score (ventana larga 50 barras)
    # Mide qué tan lejos ("estirado") está el precio de su media estadística
    # dX_t = theta(mu - X_t)dt + sigma*dW_t
    rolling_mean = df["Close"].rolling(window=50).mean()
    rolling_std = df["Close"].rolling(window=50).std()
    df["quant_zscore"] = (df["Close"] - rolling_mean) / rolling_std.replace(0, 1)

    # 3b. Z-Score Rápido (ventana corta 10 barras)
    # Detecta movimientos visibles a corto plazo (rebotes, oscilaciones)
    # Útil para el Motor de Reversión a la Media en timeframes H1
    rolling_mean_10 = df["Close"].rolling(window=10).mean()
    rolling_std_10 = df["Close"].rolling(window=10).std()
    df["quant_zscore_10"] = (df["Close"] - rolling_mean_10) / rolling_std_10.replace(0, 1)
    df["quant_zscore_10"] = df["quant_zscore_10"].fillna(0.0)

    # 4. Shannon Entropy (Market Uncertainty)
    # Mide el "desorden" o imprevisibilidad. Usamos rolling log-returns.
    window = 50
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    
    def calc_entropy(x):
        try:
            hist, _ = np.histogram(x, bins=20, density=True)
            p_k = hist + 1e-10
            p_k = p_k / p_k.sum()
            ent = -np.sum(p_k * np.log(p_k))
            return min(1.0, ent / np.log(20)) # Normalize
        except:
            return 0.5
            
    df["quant_entropy"] = log_returns.rolling(window=window).apply(calc_entropy)
    df["quant_entropy"] = df["quant_entropy"].fillna(0.5)

    quant_cols = ["quant_fer", "quant_vam", "quant_zscore", "quant_entropy"]
    return df, quant_cols

def add_hmm_regime_proxy(df):
    """
    Simula un Hidden Markov Model para detectar 3 regímenes:
    0: Tranquilo, 1: Tendencia, 2: Crisis/Alta Vol.
    """
    vol = df["Close"].pct_change().rolling(20).std()
    mom = df["Close"].pct_change(20)
    
    regime = pd.Series(0, index=df.index)
    # Regimen 1: Alta inercia (Tendencia)
    regime[(mom.abs() > mom.std()) & (vol < vol.mean())] = 1
    # Regimen 2: Caos (Crisis)
    regime[vol > vol.mean() * 2] = 2
    
    df["regime_proxy"] = regime
    return df, ["regime_proxy"]

def add_physics_features(df):
    """
    Implementación de Maquinaria de Inferencia (Phase 6).
    Basado en Teoría de Fluidos (Navier-Stokes) y Geometría de la Información.
    """
    import numpy as np
    
    # 1. Navier-Stokes: Market Pressure (Aceleración vs Flujo)
    # Presión = Aceleración del precio x Volumen Relativo
    velocity = df["Close"].diff()
    acceleration = velocity.diff()
    vol_rel = df["Volume"] / df["Volume"].rolling(50).mean().replace(0, 1)
    df["phys_pressure"] = (acceleration * vol_rel).rolling(10).mean()
    
    # 2. Market Viscosity (Fricción Transaccional)
    # Alta volatilidad con bajo volumen = Alta fricción (Liquidez delgada)
    df["phys_viscosity"] = (df["High"] - df["Low"]) / df["Volume"].replace(0, 1)
    df["phys_viscosity"] = df["phys_viscosity"].rolling(20).mean()

    # 3. Fisher-Rao Metric (Cambio de Régimen Informacional)
    # Distancia entre la distribución local (10) y la global (50)
    mu_local = df["Close"].pct_change().rolling(10).mean()
    mu_global = df["Close"].pct_change().rolling(50).mean()
    std_global = df["Close"].pct_change().rolling(50).std().replace(0, 1)
    df["phys_fisher"] = (mu_local - mu_global).abs() / std_global

    # 4. Hawkes Intensity (Cascada de Volatilidad)
    # EWMA de retornos absolutos (Proxy de auto-excitación)
    abs_ret = df["Close"].pct_change().abs()
    df["phys_hawkes"] = abs_ret.ewm(halflife=5).mean()

    physics_cols = ["phys_pressure", "phys_viscosity", "phys_fisher", "phys_hawkes"]
    return df, physics_cols

def add_golden_strategy_features(df):
    """
    Implementa la Lógica 'Golden Strategy' (V10):
    1. Tendencia EMA 200: Precio por encima/debajo de EMA200 sólida.
    2. MACD Cross: Cruce de la línea MACD sobre Signal en zona favorable.
    """
    if "ma_200" not in df.columns:
        df["ma_200"] = ta.sma(df["Close"], length=200)

    # Asegurar MACD
    if "macd" not in df.columns:
        macd = ta.macd(df["Close"])
        col_macd = [c for c in macd.columns if c.startswith("MACD_")][0]
        col_sig = [c for c in macd.columns if c.startswith("MACDs_")][0]
        df["macd"] = macd[col_macd]
        df["macd_signal_line"] = macd[col_sig]
    else:
        # Intentamos obtener la signal line si no existe columna explícita
        if "macd_signal_line" not in df.columns:
             macd = ta.macd(df["Close"])
             col_sig = [c for c in macd.columns if c.startswith("MACDs_")][0]
             df["macd_signal_line"] = macd[col_sig]

    # --- 1. SEÑAL DE TENDENCIA (EMA 200) ---
    above_ema = (df["Open"] > df["ma_200"]) & (df["Close"] > df["ma_200"])
    below_ema = (df["Open"] < df["ma_200"]) & (df["Close"] < df["ma_200"])
    
    # Check consistency of last 6 candles
    trend_up = above_ema.rolling(window=6).min().fillna(0).astype(int)
    trend_down = below_ema.rolling(window=6).min().fillna(0).astype(int)
    
    df["golden_trend"] = 0
    df.loc[trend_up == 1, "golden_trend"] = 1
    df.loc[trend_down == 1, "golden_trend"] = -1

    # --- 2. SEÑAL DE CRUCE MACD (Setup) ---
    macd_prev = df["macd"].shift(1)
    sig_prev = df["macd_signal_line"].shift(1)
    
    # Cruce Alcista (Buy): Cross Up BELOW ZERO
    cross_up = (macd_prev <= sig_prev) & (df["macd"] > df["macd_signal_line"])
    valid_long = cross_up & (df["macd"] < 0) & (df["macd_signal_line"] < 0)
    
    # Cruce Bajista (Sell): Cross Down ABOVE ZERO
    cross_down = (macd_prev >= sig_prev) & (df["macd"] < df["macd_signal_line"])
    valid_short = cross_down & (df["macd"] > 0) & (df["macd_signal_line"] > 0)
    
    df["golden_cross"] = 0
    df.loc[valid_long, "golden_cross"] = 1
    df.loc[valid_short, "golden_cross"] = -1
    
    # --- 3. SETUP COMBINADO (V10 GOLDEN TRIGGER) ---
    df["golden_setup"] = 0
    df.loc[(df["golden_trend"] == 1) & (df["golden_cross"] == 1), "golden_setup"] = 1
    df.loc[(df["golden_trend"] == -1) & (df["golden_cross"] == -1), "golden_setup"] = -1

    return df, ["golden_trend", "golden_cross", "golden_setup"]

def augment_data_noise(df, probability=0.3):
    """
    Inyeccion de Ruido Sintetico (Phase 5 - Punto 11)
    """
    import numpy as np
    mask = np.random.rand(len(df)) < probability
    n_masked = int(mask.sum())
    if n_masked > 0:
        noise = np.random.normal(0, 0.00005, size=n_masked) # ±0.5 pips
        df.loc[mask, "Close"] += noise
    return df

def add_volume_sniper_features(df):
    """
    Volumen Spread Analysis (VSA). "Lupa" para Modo Francotirador (V12).
    Busca huellas institucionales enormes analizando la cantidad de volumen vs tamaño de la vela.
    """
    import numpy as np

    # 1. Spread (tamaño) de la vela H1
    spread = (df["High"] - df["Low"]).replace(0, 0.00001)

    # 2. VSA Ratio (Volumen / Spread)
    # Mucho volumen pero poco movimiento = Gran jugador acumulando posición ocultamente.
    vsa_ratio = df["Volume"] / spread
    df["vsa_ratio_norm"] = vsa_ratio / vsa_ratio.rolling(50).mean().replace(0, 1)

    # 3. Delta Volumen (Acelerador de Presión)
    df["volume_delta"] = df["Volume"].pct_change()

    # 4. Vela Clímax (Detector de agotamiento del mercado)
    avg_vol = df["Volume"].rolling(50).mean()
    avg_spr = spread.rolling(50).mean()
    df["climax_candle"] = ((df["Volume"] > avg_vol * 2.5) & (spread > avg_spr * 1.5)).astype(float)

    sniper_cols = ["vsa_ratio_norm", "volume_delta", "climax_candle"]
    df[sniper_cols] = df[sniper_cols].fillna(0)

    return df, sniper_cols