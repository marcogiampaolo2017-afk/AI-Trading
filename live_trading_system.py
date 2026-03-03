import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import sys
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Importación del motor de indicadores del proyecto
import indicators

# =================================================================
# 1. AJUSTES DEL BROKER Y CUENTA (AvaTrade / MetaTrader 5)
# =================================================================
# Aquí definimos el símbolo a operar, el marco temporal y el servidor.
# Es vital que el servidor coincida exactamente con lo que ves en tu MT5.
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
SERVER = "Ava-Real 1-MT5"

# LOGIN CREDENTIALS (REPLACE WITH YOURS)
ACCOUNT_ID = 89856601  # Ejemplo del usuario
PASSWORD = "1973Investi@$52"

# TRADING OPTIONS (Must match training/GUI settings)
SL_OPTS = [10, 20, 30, 50, 80, 100]
TP_OPTS = [10, 20, 30, 50, 80, 100]
WIN_SIZE = 30
MAGIC_NUMBER = 123456

# =================================================================
# 2. SISTEMA DE ENLACE CON METATRADER 5
# =================================================================
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

# =================================================================
# 3. NÚCLEO NEURONAL ADAPTATIVO (Carga Inteligente)
# =================================================================
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
        
        # 1. Intentar cargar el modelo primario para detectar arquitectura
        print(f"🔍 Analizando arquitectura de {primary_model_path}...")
        try:
            temp_model = RecurrentPPO.load(primary_model_path)
            self.required_features = temp_model.observation_space.shape[-1]
            print(f"🧩 Arquitectura detectada: {self.required_features} neuronas.")
        except Exception as e:
            print(f"⚠️ Error al analizar {primary_model_path}: {e}")
            self.required_features = 20 # Fallback standard
            
        # 2. Configurar entorno y normalizador
        self.setup_environment(normalize_path)
        
        # 3. Cargar el modelo definitivo con el entorno ya listo
        try:
            self.model = RecurrentPPO.load(primary_model_path, env=self.v_env)
            print("💎 Cerebro del bot cargado y listo.")
        except Exception as e:
            print(f"❌ Error crítico al acoplar modelo: {e}")
            sys.exit(1)

        self.action_map = self._build_action_map()

    def setup_environment(self, normalize_path):
        """Asigna los indicadores correctos para sumar el total de neuronas detectadas."""
        # --- DEFINICIÓN DE MENÚ DE INDICADORES ---
        base = ["rsi_14", "rsi_50", "adx_14", "atr_14", "bb_upper_diff", "bb_lower_diff", 
                "close_ma20_diff", "close_ma50_diff", "ma_spread", "ma_spread_slope", "volume_zscore", "pv_divergence"]
        quant = ["quant_fer", "quant_vam", "quant_zscore", "quant_entropy"]
        regime = ["regime_proxy"]
        phys = ["phys_pressure", "phys_viscosity", "phys_fisher", "phys_hawkes"]
        golden = ["golden_trend", "golden_cross", "golden_setup"]
        
        state_f = 3 # Position, Time, Unrealized
        
        # Selección dinámica
        if self.required_features == 26:
            # V10 Golden: Base(12) + Quant(4) + Regime(1) + Phys(4) + Golden(2 de 3) + State(3) = 26
            self.feature_cols = base + quant + regime + phys + golden[:2]
            print("📝 Modo: V10 Golden Detectado.")
        elif self.required_features == 24:
            # V7 Survivor: Base(12) + Quant(4) + Regime(1) + Phys(4) + State(3) = 24
            self.feature_cols = base + quant + regime + phys
            print("📝 Modo: V7 Survivor Detectado.")
        elif self.required_features == 20:
            # V5 Ultimate: Base(12) + Quant(4) + Regime(1) + State(3) = 20
            self.feature_cols = base + quant + regime
            print("📝 Modo: V5 Ultimate Detectado.")
        else:
            # Fallback seguro
            self.feature_cols = base + quant + regime
            print(f"⚠️ Arquitectura {self.required_features} desconocida. Usando Base+Quant.")

        # Recrear DummyEnv
        def make_dummy():
            from trading_env import ForexTradingEnv
            return ForexTradingEnv(pd.DataFrame(np.zeros((100, len(self.feature_cols)))), 
                                 window_size=WIN_SIZE, sl_options=SL_OPTS, tp_options=TP_OPTS)
        
        self.v_env = DummyVecEnv([make_dummy])
        
        # Cargar Normalización
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
        """Action map remains identical."""
        actions = [("HOLD", 0, 0, 0), ("CLOSE", 0, 0, 0)]
        for d in [0, 1]:
            for sl in SL_OPTS:
                for tp in TP_OPTS:
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
        
        # 1. Aplicar indicadores
        df, _ = indicators.add_indicators(df)
        df, _ = indicators.add_quant_features(df)
        df, _ = indicators.add_hmm_regime_proxy(df)
        df, _ = indicators.add_physics_features(df)
        df, _ = indicators.add_golden_strategy_features(df)
        
        # 2. Construir ventana de observación usando las columnas seleccionadas
        obs_b = df[self.feature_cols].tail(WIN_SIZE).to_numpy()
        
        # 3. Inyectar estado del bot (Position en [-1, 0, 1], Time, Unrealized)
        # Adaptado para coincidir con la escala del entrenamiento
        pos_val = float(n_pos) # Simplificado para el Master Guide
        state_v = np.tile(np.array([pos_val, 0.0, 0.0]), (WIN_SIZE, 1))
        
        obs = np.hstack([obs_b, state_v]).astype(np.float32)
        obs_ready = obs[np.newaxis, ...] # Dimensión Batch
        
        # 4. Normalizar
        if hasattr(self.v_env, 'normalize_obs'):
            return self.v_env.normalize_obs(obs_ready)
        return obs_ready

# =================================================================
# 4. EJECUCIÓN DE ÓRDENES (Interface MT5)
# =================================================================
def execute_trade(action_tuple):
    """
    TRADUCTOR: Convierte las decisiones de la IA (0, 1, 2...) en órdenes
    reales de COMPRA, VENTA o CIERRE en MetaTrader 5 con SL y TP.
    """
    act_type, direction, sl_pips, tp_pips = action_tuple
    
    if act_type == "HOLD":
        return
        
    if act_type == "CLOSE":
        # Cerrar todas las posiciones del símbolo
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
        # Evitar sobre-operar si ya estamos dentro
        if len(mt5.positions_get(symbol=SYMBOL)) > 0:
            return

        tick = mt5.symbol_info_tick(SYMBOL)
        point = mt5.symbol_info(SYMBOL).point
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid
        
        # Calcular SL/TP
        sl_dist = sl_pips * 10 * point # 1 pip = 10 points
        tp_dist = tp_pips * 10 * point
        
        sl = price - sl_dist if direction == 1 else price + sl_dist
        tp = price + tp_dist if direction == 1 else price - tp_dist
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": 0.01, # Lote fijo mínimo para seguridad
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

# =================================================================
# 5. BUCLE DE MERCADO EN TIEMPO REAL
# =================================================================
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

    # Prioridad: 1. V10 Golden (Máxima precisión + Normalizador coincidente)
    model_paths = [
        "best_models/best_model.zip",
        "model_eurusd_titany_v10_golden.zip",
        "model_eurusd_titany_v5_ultimate.zip"
    ]
    
    selected_model = None
    for path in model_paths:
        if os.path.exists(path):
            selected_model = path
            break
            
    if not selected_model:
        print("❌ No se encontró ningún modelo (.zip) en el directorio.")
        return
        
    normalize_path = "vec_normalize.pkl"
    agent = LiveTradingAgent(selected_model, normalize_path)
    
    # 0. Verificar símbolo
    if not mt5.symbol_select(SYMBOL, True):
        print(f"❌ Símbolo {SYMBOL} no encontrado o no disponible.")
        return
    
    print(f"🚀 Iniciando bucle de mercado para {SYMBOL}...")
    try:
        while True:
            # 1. Obtener datos históricos recientes (300 velas para dar margen a MA200)
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 300)
            if rates is None:
                err = mt5.last_error()
                # Si es domingo y no hay datos, es probablemente por el mercado cerrado
                if err[0] == mt5.RES_ERROR_PARAMS: # O similar si el mercado está off
                     print(f"⏳ Esperando datos (Mercado Cerrado o sin conexión). Apertura: 23:00. ({err})")
                else:
                    print(f"⚠️ Error MT5 al pedir datos: {err}. Reintentando...")
                time.sleep(30)
                continue
                
            df = pd.DataFrame(rates)
            df.rename(columns={'time': 'Gmt time', 'open': 'Open', 'high': 'High', 
                              'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            
            # 2. Obtener estado de posiciones
            n_pos = len(mt5.positions_get(symbol=SYMBOL))
            
            # 3. Preparar observación y predecir
            obs = agent.prepare_observation(df, n_pos)
            action, _ = agent.model.predict(obs, deterministic=True)
            
            # 4. Traducir acción
            action_info = agent.action_map[int(action.item())]
            execute_trade(action_info)
            
            # 5. Esperar al siguiente ciclo (Mercado H1)
            # El bot duerme 10 segundos para no saturar el enlace IPC
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido manualmente por el usuario.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
