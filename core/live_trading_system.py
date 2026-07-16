import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import sys

# ── Ensure project root is on the path so relative imports work from any CWD ──
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_CORE_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import indicators
from env.indicators import add_volume_sniper_features
from env.trading_env import ForexTradingEnv
import config  # shared constants — MUST match training

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
# ── Credentials: read from environment variables; fall back to legacy constants
# (set MT5_SERVER / MT5_ACCOUNT_ID / MT5_PASSWORD in your OS environment or a .env file)
SERVER     = os.environ.get("MT5_SERVER",     "Ava-Real 1-MT5")
_account_id_raw = os.environ.get("MT5_ACCOUNT_ID")
_password_raw   = os.environ.get("MT5_PASSWORD")
if not _account_id_raw or not _password_raw:
    raise EnvironmentError(
        "MT5 credentials not found. "
        "Set MT5_ACCOUNT_ID and MT5_PASSWORD as environment variables "
        "before running live_trading_system.py."
    )
ACCOUNT_ID = int(_account_id_raw)
PASSWORD   = _password_raw
# SHOWSTOPPER FIX: use the SAME SL/TP/WIN as training (from config).
# A mismatch here creates a different action-space size → wrong predictions.
SL_OPTS  = config.SL_OPTIONS   # [20, 50, 100]
TP_OPTS  = config.TP_OPTIONS   # [40, 100, 200]
WIN_SIZE = config.WIN_SIZE     # 30
MAGIC_NUMBER = 123456
def connect_to_mt5(login, password, server):
    """
    Se encarga de inicializar el terminal de MT5 en Windows y hacer el login.
    Si el mercado está abierto y las credenciales son correctas, nos da acceso.
    """
    if not mt5.initialize():
        print("❌ Error al inicializar MetaTrader 5.")
        return False
    authorized = mt5.login(login, password=password, server=server)
    if authorized:
        print(f"✅ Conectado con éxito a la cuenta {login} en el servidor {server}")
        acc_info = mt5.account_info()
        print(f"💰 Balance: {acc_info.balance} | Equity: {acc_info.equity} | Profit: {acc_info.profit}")
        return True
    else:
        print(f"❌ Fallo de autenticación: {mt5.last_error()}")
        return False
class LiveTradingAgent:
    """
    Esta clase actúa como el 'Cerebro'. Su función es:
    1. Cargar el modelo .zip y detectar cuántas neuronas de entrada necesita.
    2. Configurar el sistema de normalización (.pkl) para que las escalas coincidan.
    3. Seleccionar los indicadores correctos (V5, V7 o V10) automáticamente.
    """
    def __init__(self, primary_model_path, normalize_path):
        self.model = None
        self.v_env = None
        self.required_features = 0
        self.feature_cols = []
        print(f"🔍 Analizando arquitectura de {primary_model_path}...")
        try:
            temp_model = RecurrentPPO.load(primary_model_path)
            self.required_features = temp_model.observation_space.shape[-1]
            print(f"🧩 Arquitectura detectada: {self.required_features} neuronas.")
        except Exception as e:
            print(f"⚠️ Error al analizar {primary_model_path}: {e}")
            self.required_features = 20                    
        self.setup_environment(normalize_path)
        try:
            self.model = RecurrentPPO.load(primary_model_path, env=self.v_env)
            print("💎 Cerebro del bot cargado y listo.")
        except Exception as e:
            print(f"❌ Error crítico al acoplar modelo: {e}")
            sys.exit(1)
        self.action_map = self._build_action_map()
    def setup_environment(self, normalize_path):
        """Asigna los indicadores correctos para sumar el total de neuronas detectadas."""
        base   = ["rsi_14", "rsi_50", "adx_14", "atr_14", "bb_upper_diff", "bb_lower_diff",
                  "close_ma20_diff", "close_ma50_diff", "ma_spread", "ma_spread_slope",
                  "volume_zscore", "pv_divergence"]
        quant  = ["quant_fer", "quant_vam", "quant_zscore", "quant_entropy"]
        regime = ["regime_proxy"]
        phys   = ["phys_pressure", "phys_viscosity", "phys_fisher", "phys_hawkes"]
        golden = ["golden_trend", "golden_cross", "golden_setup"]
        sniper = ["vsa_ratio_norm", "volume_delta", "climax_candle"]  # V12 Sniper features
        state_f = 3  # [position, time_norm, unrealized_scaled] appended in env

        # feature count = obs_shape[-1] - state_f
        n_feat = self.required_features - state_f
        if n_feat == 27:
            # V12 Sniper: base(12)+quant(4)+regime(1)+phys(4)+golden(3)+sniper(3)
            self.feature_cols = base + quant + regime + phys + golden + sniper
            print("📝 Modo: V12 Sniper Detectado (27 features).")
        elif n_feat == 26:
            self.feature_cols = base + quant + regime + phys + golden[:2] + sniper
            print("📝 Modo: V11 Sniper-Parcial Detectado.")
        elif n_feat == 24:
            self.feature_cols = base + quant + regime + phys + golden[:3]
            print("📝 Modo: V10 Golden Detectado.")
        elif n_feat == 21:
            self.feature_cols = base + quant + regime + phys
            print("📝 Modo: V7 Survivor Detectado.")
        elif n_feat == 17:
            self.feature_cols = base + quant + regime
            print("📝 Modo: V5 Ultimate Detectado.")
        else:
            # Fallback: best guess matching all known feature groups
            self.feature_cols = base + quant + regime + phys + golden + sniper
            print(f"⚠️ Arquitectura con {n_feat} features desconocida. Usando V12 completo.")
        def make_dummy():
            # Build a properly-named dummy DataFrame so ForexTradingEnv validates correctly
            dummy_df = pd.DataFrame(
                np.zeros((100, len(self.feature_cols))),
                columns=self.feature_cols
            )
            return ForexTradingEnv(
                dummy_df,
                window_size=WIN_SIZE,
                sl_options=SL_OPTS,
                tp_options=TP_OPTS,
                feature_columns=self.feature_cols,
            )
        self.v_env = DummyVecEnv([make_dummy])
        if os.path.exists(normalize_path):
            try:
                self.v_env = VecNormalize.load(normalize_path, self.v_env)
                self.v_env.training = False
                self.v_env.norm_reward = False
                print(f"🧠 Escalas de normalización aplicadas ({normalize_path}).")
            except Exception as e:
                print(f"⚠️ El normalizador no coincide con esta arquitectura: {e}")
                print("Probando carga sin normalización (Peligro: Escalas inconsistentes)")
    def _build_action_map(self):
        """Build action map from explicit (sl,tp) pairs — must match training."""
        actions = [("HOLD", 0, 0, 0), ("CLOSE", 0, 0, 0)]
        for d in [0, 1]:
            for sl, tp in config.SL_TP_PAIRS:   # same pairs as training
                actions.append(("OPEN", d, float(sl), float(tp)))
        return actions
    def prepare_observation(self, df_rates, n_pos):
        """
        TRANSFORMACIÓN DE DATOS:
        Aquí convertimos las velas crudas de MT5 en una 'Observación' que la IA entiende.
        Se aplican indicadores, se calculan quants, física y estrategia Golden.
        Finalmente, se inyecta el estado de tu cuenta (si hay posición abierta).
        """
        df = df_rates.copy()
        df, _ = indicators.add_indicators(df)
        df, _ = indicators.add_quant_features(df)
        df, _ = indicators.add_hmm_regime_proxy(df)
        df, _ = indicators.add_physics_features(df)
        df, _ = indicators.add_golden_strategy_features(df)
        # SHOWSTOPPER FIX: V12 Sniper needs these 3 columns (vsa_ratio_norm,
        # volume_delta, climax_candle). Without them the observation is short
        # by 3 features and the model crashes or produces garbage predictions.
        df, _ = add_volume_sniper_features(df)
        obs_b = df[self.feature_cols].tail(WIN_SIZE).to_numpy()
        pos_val = float(n_pos)                                    
        state_v = np.tile(np.array([pos_val, 0.0, 0.0]), (WIN_SIZE, 1))
        obs = np.hstack([obs_b, state_v]).astype(np.float32)
        obs_ready = obs[np.newaxis, ...]                  
        if hasattr(self.v_env, 'normalize_obs'):
            return self.v_env.normalize_obs(obs_ready)
        return obs_ready
def execute_trade(action_tuple):
    """
    TRADUCTOR: Convierte las decisiones de la IA (0, 1, 2...) en órdenes
    reales de COMPRA, VENTA o CIERRE en MetaTrader 5 con SL y TP.
    """
    act_type, direction, sl_pips, tp_pips = action_tuple
    if act_type == "HOLD":
        return
    if act_type == "CLOSE":
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions and len(positions) > 0:
            for p in positions:
                tick = mt5.symbol_info_tick(SYMBOL)
                order_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": p.volume,
                    "type": order_type,
                    "position": p.ticket,
                    "price": price,
                    "magic": MAGIC_NUMBER,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                mt5.order_send(request)
            print("🔒 Posiciones cerradas por orden de la IA.")
        return
    if act_type == "OPEN":
        if len(mt5.positions_get(symbol=SYMBOL)) > 0:
            return
        tick = mt5.symbol_info_tick(SYMBOL)
        point = mt5.symbol_info(SYMBOL).point
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid
        sl_dist = sl_pips * 10 * point                    
        tp_dist = tp_pips * 10 * point
        sl = price - sl_dist if direction == 1 else price + sl_dist
        tp = price + tp_dist if direction == 1 else price - tp_dist
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": 0.01,                                  
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": MAGIC_NUMBER,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🚀 Orden {act_type} enviada: SL={sl_pips} TP={tp_pips}")
        else:
            print(f"❌ Error al enviar orden: {result.comment}")
def main():
    """
    MOTOR PRINCIPAL:
    1. Se conecta a la cuenta.
    2. Busca el mejor modelo disponible en tu carpeta.
    3. Inicia un ciclo infinito que analiza el mercado cada 10 segundos.
    """
    print("--- TITANY RL LIVE MASTER GUIDE ---")
    if not connect_to_mt5(ACCOUNT_ID, PASSWORD, SERVER):
        return
    # SHOWSTOPPER FIX: include the V12 Sniper model that train_agent.py saves.
    # Paths are relative to the project root (parent of /core/).
    models_dir = os.path.join(_ROOT_DIR, "models")
    model_paths = [
        os.path.join(models_dir, "model_eurusd_titany_v12_sniper.zip"),  # NEW — V12
        os.path.join(models_dir, "best_models", "best_model.zip"),
        os.path.join(models_dir, "best_model.zip"),
    ]
    selected_model = None
    for path in model_paths:
        if os.path.exists(path):
            selected_model = path
            break
    if not selected_model:
        print("❌ No se encontró ningún modelo (.zip) en el directorio.")
        print(f"   Buscado en: {models_dir}")
        return
    normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
    agent = LiveTradingAgent(selected_model, normalize_path)
    if not mt5.symbol_select(SYMBOL, True):
        print(f"❌ Símbolo {SYMBOL} no encontrado o no disponible.")
        return
    print(f"🚀 Iniciando bucle de mercado para {SYMBOL}...")
    try:
        while True:
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 300)
            if rates is None:
                err = mt5.last_error()
                if err[0] == mt5.RES_ERROR_PARAMS:                                   
                     print(f"⏳ Esperando datos (Mercado Cerrado o sin conexión). Apertura: 23:00. ({err})")
                else:
                    print(f"⚠️ Error MT5 al pedir datos: {err}. Reintentando...")
                time.sleep(30)
                continue
            df = pd.DataFrame(rates)
            df.rename(columns={'time': 'Gmt time', 'open': 'Open', 'high': 'High', 
                              'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            n_pos = len(mt5.positions_get(symbol=SYMBOL))
            obs = agent.prepare_observation(df, n_pos)
            action, _ = agent.model.predict(obs, deterministic=True)
            action_info = agent.action_map[int(action.item())]
            execute_trade(action_info)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido manualmente por el usuario.")
    finally:
        mt5.shutdown()
if __name__ == "__main__":
    main()
