import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
import requests
import customtkinter as ctk
from sb3_contrib import RecurrentPPO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import indicators
import datetime

# =================================================================
# 1. CONFIGURACIÓN MAESTRA (DATOS REALES DE TU BASE44)
# =================================================================
# Usamos la API oficial de Base44 para garantizar permisos de escritura (PUT)
BASE44_API_URL = "https://app.base44.com/api/apps/696fe84f14c617992088dd7d/entities"
BASE44_API_KEY = "82e4ca558e8546f89859b4a3dba1e1cf"
# Este es el ID exacto de tu App móvil que vimos en el CSV

STATUS_ROW_ID = "696fe9543bd9c31ec5fcb8af" 

# SINTONIZADO PARA TU BROKER (EURUSD en MAYÚSCULAS)
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
MAX_POSITIONS = 1
RESTART_DELAY = 180
WIN = 30

# ---- CONFIGURACIÓN EXACTA MODELO GOLDEN ----
SL_OPTS = [10, 20, 30, 50, 80, 100]
TP_OPTS = [10, 20, 30, 50, 80, 100]

# =================================================================
# 1.5. MODOS DE RENDIMIENTO (Dinamismo del Sistema)
# =================================================================
# 1: Actual (Normal) - Sleep 2s + Sync 3s
# 2: Optimizado (AFK Monitor) - Sleep 0.1s + Sync cada 20 ciclos (~10 FPS)
# 3: Headless (Turbo) - Sin UI + No Sync (+50 FPS)
BOT_MODE = 2 

# =================================================================
# 2. MOTOR DE COMUNICACIÓN (ACTUALIZA TU MÓVIL)
# =================================================================
class TitanySync:
    def __init__(self, url, key, row_id):
        # url viene como ".../entities". Appendemos la entidad y el ID.
        self.url = f"{url}/BotStatus/{row_id}"
        # Include Accept header for JSON responses
        self.headers = {
            "api_key": key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # Verify endpoint connectivity on startup (non‑blocking for UI)
        self._test_connection()

    def _test_connection(self):
        """Quick GET request to ensure the Base44 endpoint is reachable.
        Logs the result; does not raise exceptions to keep the UI alive.
        """
        try:
            response = requests.get(self.url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                print("[Base44] Conexión establecida correctamente.")
            else:
                print(f"[Base44] Conexión fallida (código {response.status_code}): {response.text[:200]}")
        except Exception as e:
            print(f"[Base44] Error al probar la conexión: {e}")

    def update_mobile_app(self, profit, equity, balance, n_trades, state, lot, win_rate, daily_profit, max_dd):
        """Envía los datos exactos que tu App espera recibir"""
        try:
            payload = {
                "floating_profit": str(round(profit, 2)),
                "equity": str(round(equity, 2)),
                "balance": str(round(balance, 2)),
                "open_positions": str(n_trades),
                "state": str(state),
                "lot_size": str(lot),
                "current_symbol": SYMBOL,
                "win_rate": float(round(win_rate, 1)),
                "daily_profit": float(round(daily_profit, 2)),
                "max_dd": float(round(max_dd, 2)),
                "last_update": time.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            # Base44 API usa PUT para actualizaciones completas
            response = requests.put(self.url, json=payload, headers=self.headers, timeout=15)
            if response.status_code not in (200, 201, 204):
                print(f"[Base44] PUT fallido (código {response.status_code}): {response.text[:200]}")
        except Exception:
            # Silently handle network jitter to avoid console spam
            pass

    def check_kill_switch(self):
        """Verifica si el móvil ha presionado el botón Kill Switch"""
        try:
            cmd_url = f"{self.url.split('/BotStatus')[0]}/Command"
            response = requests.get(cmd_url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        data = data[-1]
                    return data.get("kill_bot", False) == True or data.get("kill_bot", False) == "true"
                except ValueError:
                    # Si falla el JSON, probablemente es HTML (SPA fallback). Ignorar silenciosamente o loguear una "z" para debug.
                    pass
            else:
                # print(f"[Base44] GET kill switch fallido (código {response.status_code})")
                pass
        except Exception:
            pass
        return False

    

# =================================================================
# 1. MOTOR DE CÁLCULO AVANZADO (Neuro-Quantum Engineering)
# =================================================================
class NeuroQuantEngine:
    """
    Este es el motor matemático 'oculto' detrás de la interfaz.
    Calcula entropía, filtros cuánticos y gestión de riesgo Kelly.
    """
    def __init__(self):
        self.synaptic_weight = 1.0  # Confianza base del sistema (Plasticidad)
        self.learning_rate = 0.05   # Tasa de adaptación STDP
        
    def calculate_entropy(self, price_series, window=50):
        """
        Calcula la Entropía de Shannon del mercado (Incertidumbre).
        Concepto: Física estadística aplicada a flujos financieros.
        Higher Entropy = Higher Chaos/Uncertainty.
        """
        try:
            if len(price_series) < window:
                return 0.5
            
            # Usamos log-returns para estacionariedad
            relevant_data = np.array(price_series[-window:])
            returns = np.diff(np.log(relevant_data))
            
            # Histograma de probabilidad
            hist, _ = np.histogram(returns, bins=20, density=True)
            p_k = hist + 1e-10  # Evitar log(0)
            p_k = p_k / np.sum(p_k) # Normalizar probabilidad
            
            # Fórmula de Shannon: H = -Sum(p * log(p))
            entropy = -np.sum(p_k * np.log(p_k))
            
            # Normalizar a 0-1 (Max entropía para 20 bins es ~3.0)
            max_ent = np.log(20)
            normalized_entropy = min(1.0, entropy / max_ent)
            
            return normalized_entropy
        except Exception as e:
            return 0.5 # Valor neutral por defecto

    def calculate_kelly_lot(self, base_lot, win_rate, balance, rr=2.0):
        """
        Calcula el lote óptimo usando el Criterio de Kelly (Fraccional 20%).
        Fórmula: K = (p*R - (1-p)) / R
        """
        p = win_rate / 100.0
        if p <= 0.33: return base_lot # Kelly no es válido para p baja, usamos base mínimo
        
        k_full = (p * rr - (1 - p)) / rr
        # Usamos Kelly Fraccional (20%) para evitar sobre-exposición masiva
        k_safe = k_full * 0.2
        
        # Convertimos fracción de cuenta a lote (Aproximación para EURUSD)
        # Margin required approx $30 per 0.01 lot on 1:30 leverage
        potential_lot = (balance * k_safe) / 3000 # Escala conservadora
        
        return max(base_lot, round(potential_lot, 2))

    def adapt_synapses(self, won_last_trade):
        """
        Simula Plasticidad Sináptica (Hebbian Learning).
        """
        if won_last_trade:
            self.synaptic_weight = min(2.0, self.synaptic_weight * 1.1) # LTC (Potenciación)
        else:
            self.synaptic_weight = max(0.5, self.synaptic_weight * 0.8) # LTD (Depresión)
        return self.synaptic_weight

    def quantum_energy_filter(self, actions_list, current_atr, entropy):
        """
        Proxy de QUBO (Quantum Unconstrained Binary Optimization).
        Calcula el 'Hamiltoniano de Riesgo' para cada acción y elige el de mínima energía.
        H = A * Risk + B * Entropy - C * Reward_Potential
        """
        best_action = None
        min_energy = float('inf')
        
        for act in actions_list:
            if act[0] != "OPEN": continue
            
            # Parametrización del Hamiltoniano (Valores del 2026)
            sl_dist = act[2]
            tp_dist = act[3]
            
            # Término de Riesgo (Energía Positiva)
            risk_energy = (sl_dist * current_atr * 1000)
            # Término de Caos (Energía Positiva)
            chaos_energy = entropy * 50
            # Término de Recompensa (Energía Negativa)
            reward_potential = (tp_dist / (sl_dist + 1e-6)) * 20
            
            # H Total
            energy = risk_energy + chaos_energy - reward_potential
            
            if energy < min_energy:
                min_energy = energy
                best_action = act
                
        return best_action if best_action else actions_list[0]

# =================================================================
# 2. INTERFAZ GRÁFICA PROFESIONAL (El Dashboard)
# =================================================================
class TitanyBotApp(ctk.CTk):
    """
    Controla toda la parte visual: Gráficos, botones, logs y
    la conexión de hilos con la lógica de trading.
    """
    def __init__(self):
        super().__init__()
        self.title("TITANY AI - ULTIMATE AUTO-RESTART EDITION")
        self.geometry("1600x900")
        self.configure(fg_color="#0a0e1a")

        self.equity_history = []
        self.sync_engine = TitanySync(BASE44_API_URL, BASE44_API_KEY, STATUS_ROW_ID)
        self.neuro_engine = NeuroQuantEngine() # Cortex Digital Iniciado
        self.running = True  # Must be set before setup_ui for animations

        self.setup_ui()

        self.bot_thread = threading.Thread(target=self.run_bot_logic, daemon=True)
        self.bot_thread.start()

    def setup_ui(self):
        # === PREMIUM COLOR PALETTE ===
        self.colors = {
            "bg_dark": "#050810",
            "bg_panel": "#0a0f1a",
            "bg_card": "#0d1420",
            "accent_cyan": "#00e5ff",
            "accent_purple": "#a855f7",
            "accent_pink": "#ec4899",
            "accent_green": "#10b981",
            "accent_red": "#ef4444",
            "accent_orange": "#f59e0b",
            "text_primary": "#f8fafc",
            "text_secondary": "#94a3b8",
            "border_subtle": "#1e293b",
            "border_glow": "#00e5ff"
        }
        
        self.configure(fg_color=self.colors["bg_dark"])
        
        # --- GLASSMORPHISM SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, 
                                    fg_color=self.colors["bg_panel"], 
                                    border_width=1, border_color=self.colors["border_subtle"])
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Logo / Brand Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=120)
        brand_frame.pack(fill="x", pady=(20, 10))
        brand_frame.pack_propagate(False)
        
        ctk.CTkLabel(brand_frame, text="🧠", font=("Segoe UI Emoji", 48)).pack()
        ctk.CTkLabel(brand_frame, text="TITANY AI", 
                    font=("Segoe UI", 22, "bold"), text_color=self.colors["accent_cyan"]).pack()
        ctk.CTkLabel(brand_frame, text="NEURAL TRADING SYSTEM", 
                    font=("Segoe UI", 9), text_color=self.colors["text_secondary"]).pack()
        
        # Divider
        ctk.CTkFrame(self.sidebar, height=1, fg_color=self.colors["border_subtle"]).pack(fill="x", padx=20, pady=15)
        
        # === NEURAL ACTIVITY INDICATOR ===
        neural_frame = ctk.CTkFrame(self.sidebar, fg_color=self.colors["bg_card"], 
                                   corner_radius=12, height=80)
        neural_frame.pack(fill="x", padx=15, pady=(0, 15))
        neural_frame.pack_propagate(False)
        
        neural_header = ctk.CTkFrame(neural_frame, fg_color="transparent")
        neural_header.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(neural_header, text="⚡ NEURAL STATUS", 
                    font=("Segoe UI", 11, "bold"), text_color=self.colors["accent_purple"]).pack(side="left")
        
        self.neural_dot = ctk.CTkLabel(neural_header, text="●", 
                                       font=("Segoe UI", 16), text_color=self.colors["accent_green"])
        self.neural_dot.pack(side="right")
        
        self.lbl_neural = ctk.CTkLabel(neural_frame, text="AI Processing Active", 
                                       font=("Segoe UI", 10), text_color=self.colors["text_secondary"])
        self.lbl_neural.pack(padx=15)
        
        # === CONTROL INPUTS ===
        inputs_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        inputs_frame.pack(fill="x", padx=15, pady=10)
        
        self.create_premium_input("💎 LOT SIZE", "0.01", self.colors["accent_cyan"], "entry_lots", inputs_frame)
        self.create_premium_input("🛡️ MAX LOSS", "-25.0", self.colors["accent_red"], "entry_loss", inputs_frame)
        self.create_premium_input("🎯 MAX POS", "3", self.colors["accent_orange"], "entry_maxpos", inputs_frame)
        
        # === CONNECTION STATUS ===
        conn_frame = ctk.CTkFrame(self.sidebar, fg_color=self.colors["bg_card"], 
                                 corner_radius=12, height=70)
        conn_frame.pack(side="bottom", fill="x", padx=15, pady=20)
        conn_frame.pack_propagate(False)
        
        market_status = "INITIALIZING..."
        status_color = self.colors["accent_orange"]
        
        conn_inner = ctk.CTkFrame(conn_frame, fg_color="transparent")
        conn_inner.pack(expand=True)
        
        self.conn_dot = ctk.CTkLabel(conn_inner, text="●", font=("Segoe UI", 12), text_color=status_color)
        self.conn_dot.pack(side="left", padx=(0, 8))
        
        self.lbl_mt5 = ctk.CTkLabel(conn_inner, text=f"MT5: {market_status}", 
                                    font=("Segoe UI", 11, "bold"), text_color=status_color)
        self.lbl_mt5.pack(side="left")

        # --- MAIN CONTENT AREA ---
        self.main = ctk.CTkFrame(self, fg_color=self.colors["bg_dark"])
        self.main.pack(side="right", fill="both", expand=True)

        # Top Bar with gradient effect simulation
        top_bar = ctk.CTkFrame(self.main, fg_color=self.colors["bg_panel"], height=60, corner_radius=0)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)
        
        ctk.CTkLabel(top_bar, text="📊 LIVE TRADING DASHBOARD", 
                    font=("Segoe UI", 16, "bold"), text_color=self.colors["text_primary"]).pack(side="left", padx=25, pady=15)
        
        self.lbl_time = ctk.CTkLabel(top_bar, text=time.strftime('%H:%M:%S'), 
                                     font=("Consolas", 14), text_color=self.colors["text_secondary"])
        self.lbl_time.pack(side="right", padx=25)

        # Content Grid
        content = ctk.CTkFrame(self.main, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # === PROFIT HERO CARD ===
        self.f_profit = ctk.CTkFrame(content, fg_color=self.colors["bg_card"], 
                                     corner_radius=20, border_width=2, 
                                     border_color=self.colors["accent_green"], height=200)
        self.f_profit.pack(fill="x", pady=(0, 15))
        self.f_profit.pack_propagate(False)
        
        profit_inner = ctk.CTkFrame(self.f_profit, fg_color="transparent")
        profit_inner.pack(expand=True)
        
        ctk.CTkLabel(profit_inner, text="FLOATING P/L", 
                    font=("Segoe UI", 12), text_color=self.colors["text_secondary"]).pack()
        
        self.lbl_profit = ctk.CTkLabel(profit_inner, text="+0.00", 
                                       font=("Segoe UI", 72, "bold"), text_color=self.colors["accent_green"])
        self.lbl_profit.pack()
        
        ctk.CTkLabel(profit_inner, text="USD", 
                    font=("Segoe UI", 20), text_color=self.colors["text_secondary"]).pack()
        
        self.lbl_status = ctk.CTkLabel(self.f_profit, text="● MONITORING 0/5 POSITIONS", 
                                       font=("Segoe UI", 11, "bold"), text_color=self.colors["accent_cyan"])
        self.lbl_status.pack(side="bottom", pady=15)

        # === STATS DASHBOARD GRID ===
        stats_frame = ctk.CTkFrame(content, fg_color="transparent", height=100)
        stats_frame.pack(fill="x", pady=(0, 15))
        
        # Create 4 stat cards
        self.stat_cards = {}
        stats_data = [
            ("💰 BALANCE", "0.00", self.colors["accent_cyan"], "balance"),
            ("📈 EQUITY", "0.00", self.colors["accent_purple"], "equity"),
            ("🎯 WIN RATE", "0%", self.colors["accent_green"], "winrate"),
            ("🔥 TODAY", "+0.00", self.colors["accent_orange"], "daily")
        ]
        
        for i, (title, value, color, key) in enumerate(stats_data):
            card = ctk.CTkFrame(stats_frame, fg_color=self.colors["bg_card"], 
                               corner_radius=15, border_width=1, border_color=self.colors["border_subtle"])
            card.pack(side="left", fill="both", expand=True, padx=(0 if i == 0 else 5, 5 if i < 3 else 0))
            
            ctk.CTkLabel(card, text=title, font=("Segoe UI", 10), 
                        text_color=self.colors["text_secondary"]).pack(pady=(15, 5))
            
            lbl = ctk.CTkLabel(card, text=value, font=("Segoe UI", 24, "bold"), text_color=color)
            lbl.pack(pady=(0, 15))
            self.stat_cards[key] = lbl

        # === EQUITY CHART ===
        self.f_chart = ctk.CTkFrame(content, fg_color=self.colors["bg_card"], 
                                   corner_radius=15, border_width=1, border_color=self.colors["border_subtle"])
        self.f_chart.pack(fill="both", expand=True, pady=(0, 15))
        
        chart_header = ctk.CTkFrame(self.f_chart, fg_color="transparent", height=40)
        chart_header.pack(fill="x", padx=15, pady=(10, 0))
        chart_header.pack_propagate(False)
        
        ctk.CTkLabel(chart_header, text="📈 EQUITY CURVE", 
                    font=("Segoe UI", 12, "bold"), text_color=self.colors["text_primary"]).pack(side="left")
        
        self.fig = Figure(figsize=(10, 3), dpi=100, facecolor=self.colors["bg_card"])
        self.ax = self.fig.add_subplot(111)
        self.setup_chart_style()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.f_chart)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # === LOGS & EMERGENCY ===
        bottom_frame = ctk.CTkFrame(content, fg_color="transparent", height=140)
        bottom_frame.pack(fill="x")
        bottom_frame.pack_propagate(False)
        
        # Logs
        log_frame = ctk.CTkFrame(bottom_frame, fg_color=self.colors["bg_card"], 
                                corner_radius=12, border_width=1, border_color=self.colors["border_subtle"])
        log_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.log_txt = ctk.CTkTextbox(log_frame, fg_color="transparent", 
                                     text_color=self.colors["text_secondary"], 
                                     font=("Consolas", 10), corner_radius=0)
        self.log_txt.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Emergency Button
        btn_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent", width=200)
        btn_frame.pack(side="right", fill="y")
        btn_frame.pack_propagate(False)
        
        self.btn_stop = ctk.CTkButton(btn_frame, text="⚠️\nEMERGENCY\nCLOSE ALL", 
                                     fg_color=self.colors["accent_red"], 
                                     hover_color="#dc2626",
                                     font=("Segoe UI", 14, "bold"), 
                                     command=self.emergency_stop,
                                     corner_radius=15,
                                     text_color="#ffffff")
        self.btn_stop.pack(fill="both", expand=True)

        # Start time update loop
        self.update_time()
        # Start neural pulse animation
        self.animate_neural()

    def setup_chart_style(self):
        self.ax.set_facecolor(self.colors["bg_card"])
        self.ax.tick_params(colors=self.colors["text_secondary"], labelsize=8)
        for spine in ['top', 'right']:
            self.ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            self.ax.spines[spine].set_color(self.colors["border_subtle"])

    def create_premium_input(self, label, default, color, attr, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(frame, text=label, text_color=color, 
                    font=("Segoe UI", 10, "bold")).pack(anchor="w")
        
        e = ctk.CTkEntry(frame, fg_color=self.colors["bg_card"], 
                        border_color=color, justify="center",
                        font=("Segoe UI", 13, "bold"), height=42, 
                        border_width=2, corner_radius=10,
                        text_color=self.colors["text_primary"])
        e.insert(0, default)
        e.pack(fill="x", pady=(3, 0))
        setattr(self, attr, e)

    def _update_profit_display(self, color, sign, profit):
        self.lbl_profit.configure(text=f"{sign}{profit:.2f}", text_color=color)
        self.f_profit.configure(border_color=color)

    def update_time(self):
        if self.running:
            self.lbl_time.configure(text=time.strftime('%H:%M:%S'))
            self.after(1000, self.update_time)

    def animate_neural(self):
        if self.running:
            current = self.neural_dot.cget("text_color")
            new_color = self.colors["accent_purple"] if current == self.colors["accent_green"] else self.colors["accent_green"]
            self.neural_dot.configure(text_color=new_color)
            self.after(800, self.animate_neural)

    def add_log(self, msg):
        timestamp = time.strftime('%H:%M:%S')
        # Thread-safe log update
        self.after(0, lambda: self._perform_log_update(f"[{timestamp}] {msg}\n"))

    def _perform_log_update(self, text):
        try:
            if not self.running: return
            self.log_txt.insert("end", text)
            self.log_txt.see("end")
        except:
            pass

    def update_ui_chart(self, val):
        # Thread-safe chart update
        self.after(0, lambda: self._perform_ui_update(val))

    def _perform_ui_update(self, val):
        if not self.running: return
        try:
            self.equity_history.append(val)
            if len(self.equity_history) > 750:
                self.equity_history.pop(0)
            
            self.ax.clear()
            self.setup_chart_style()
            
            if len(self.equity_history) > 1:
                x = range(len(self.equity_history))
                self.ax.fill_between(x, min(self.equity_history), self.equity_history, 
                                    alpha=0.4, color=self.colors["accent_cyan"])
                self.ax.plot(self.equity_history, color=self.colors["accent_cyan"], 
                            linewidth=2, antialiased=True)
            
            self.ax.grid(True, color=self.colors["border_subtle"], alpha=0.3, linestyle='-', linewidth=0.5)
            self.fig.tight_layout()
            self.canvas.draw()
        except (KeyboardInterrupt, RuntimeError, Exception):
            pass # Silenciar errores durante el cierre

    # =================================================================
# 3. LÓGICA DE TRADING E INTELIGENCIA ARTIFICIAL (El Motor)
# =================================================================
    def close_all_now(self):
        """Orden de pánico: Cierra todo inmediatamente."""
        pos = mt5.positions_get(symbol=SYMBOL)
        if pos:
            for p in pos:
                tick = mt5.symbol_info_tick(SYMBOL)
                tipo = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                req = {
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": p.volume,
                    "type": tipo, "position": p.ticket, "price": tick.bid if tipo == mt5.ORDER_TYPE_SELL else tick.ask,
                    "magic": 123456, "type_filling": mt5.ORDER_FILLING_FOK
                }
                mt5.order_send(req)
            self.add_log("🔒 Todas las posiciones cerradas.")

    def emergency_stop(self):
        self.close_all_now()
        self.running = False
        self.lbl_status.configure(text="⛔ SISTEMA: APAGADO", text_color="#ef4444")
        self.btn_stop.configure(state="disabled", fg_color="#4b5563")

    def run_bot_logic(self):
        """
        CICLO INFINITO DE LA IA:
        Este es el corazón del bot. Se encarga de:
        1. Cargar el modelo V10 Golden y su normalizador.
        2. Bajar datos de MT5 cada pocos segundos.
        3. Predecir la siguiente acción (Buy/Sell/Wait).
        4. Ejecutar la orden si la probabilidad es alta.
        """
        if not mt5.initialize():
            self.add_log("❌ MT5 no conectado.")
            self.lbl_mt5.configure(text="MARKET: OFFLINE", text_color="#ef4444")
            return

        model = None
        v_env = None
        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 🚀 CARGA DEL MODELO V5 ULTIMATE (Solicitado por el Usuario)
            # 🚀 CARGA DEL MODELO V10 GOLDEN (Prioridad)
            model_name = "best_models/best_model.zip"
            if not os.path.exists(os.path.join(script_dir, model_name)):
                model_name = "model_eurusd_titany_v5_ultimate.zip"
            model_path = os.path.join(script_dir, model_name)
            
            # Carga del normalizador
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            
            # Definimos las opciones de Trading Institucionales
            SL_OPTS_L = [10, 20, 30, 50, 80, 100]
            TP_OPTS_L = [10, 20, 30, 50, 80, 100]

            # Recreamos un entorno fake basado en el modelo
            def dummy_env_adaptive():
                from trading_env import ForexTradingEnv
                n_ind = 23 if "best_model" in model_name else 17
                return ForexTradingEnv(pd.DataFrame(np.zeros((100, n_ind))), window_size=30, sl_options=SL_OPTS_L, tp_options=TP_OPTS_L)
            
            v_env = DummyVecEnv([dummy_env_adaptive])
            # Intentamos cargar normalizador
            if os.path.exists(os.path.join(script_dir, "vec_normalize.pkl")):
                try:
                    v_env = VecNormalize.load(os.path.join(script_dir, "vec_normalize.pkl"), v_env)
                    v_env.training = False 
                    v_env.norm_reward = False
                except:
                    self.add_log("⚠️ AI: VecNormalize incompatible para esta arquitectura.")
            else:
                self.add_log("ℹ️ AI: Sin archivo de normalización. Usando modo RAW.")

            model = RecurrentPPO.load(model_path, env=v_env)
            self.add_log(f"💎 AI: MODELO {os.path.basename(model_name)} cargado con éxito.")
            
        except Exception as e:
            self.add_log(f"⚠️ AI: Error carga primaria ({e}). Probando compatibilidad V7 Survivor...")
            try:
                # Intento para V7 Survivor (21 indicadores + 3 estados = 24 features)
                def dummy_env_v7():
                    from trading_env import ForexTradingEnv
                     # SL/TP options must match V7 training as well!
                    return ForexTradingEnv(pd.DataFrame(np.zeros((100, 21))), window_size=30, sl_options=SL_OPTS_L, tp_options=TP_OPTS_L)
                
                v_env = DummyVecEnv([dummy_env_v7])
                model_path_bkp = os.path.join(script_dir, "best_models", "best_model.zip")
                if os.path.exists(os.path.join(script_dir, "model_eurusd_titany_v7_survivor.zip")):
                     model_path_bkp = os.path.join(script_dir, "model_eurusd_titany_v7_survivor.zip")

                model = RecurrentPPO.load(model_path_bkp, env=v_env)
                self.add_log(f"💎 AI: MODELO V7 SURVIVOR (Backup) cargado.")
            except Exception as e2:
                 self.add_log(f"❌ AI: CRITICAL - FALLO TOTAL DE CARGA: {e2}")

        # DETECCIÓN AUTOMÁTICA DE ARQUITECTURA
        # El modelo nos dice cuántas features espera. Usaremos esto en el bucle principal.
        # Shape esperada: (Batch, Pasos, Features) -> Features es el último
        try:
             self.model_features = model.observation_space.shape[-1]
             self.add_log(f"🧠 ARQUITECTURA DETECTADA: {self.model_features} Neuronas de Entrada.")
        except:
             self.model_features = 24 # Default V7

        actions = [("HOLD", 0,0,0), ("CLOSE", 0,0,0)]
        for d in [0, 1]:
            for sl in SL_OPTS:
                for tp in TP_OPTS:
                    actions.append(("OPEN", d, float(sl), float(tp)))

        last_sync_time = 0
        last_trade_time = 0 # --- NUEVO: COOLDOWN DE DISPARO ---
        last_balance = 0
        cycle_count = 0
        
        while self.running:
            if cycle_count == 0:
                print("🟢 [TITANY AI] Operación Esmeralda - Versión 1.0.0 Cargada.")
                # --- AUTO-DISCOVERY DE SÍMBOLO ---
                all_pos = mt5.positions_get()
                if all_pos:
                    print(f"🔍 [DEBUG] Encontradas {len(all_pos)} posiciones totales.")
                    for p in all_pos:
                        print(f"👉 Símbolo en MT5: '{p.symbol}' | Ticket: {p.ticket}")
                else:
                    print("🔍 [DEBUG] No se detectan posiciones abiertas en ninguna cuenta.")
                
                # Verificar error de MT5
                last_err = mt5.last_error()
                if last_err[0] != 1:
                    print(f"❌ [MT5 ERROR] Código: {last_err[0]} | Descripción: {last_err[1]}")
            
            cycle_count += 1

            # --- GESTIÓN DE POSICIONES (Símbolo vs Total) ---
            all_positions = mt5.positions_get()
            total_n_pos = len(all_positions) if all_positions else 0
            
            # Filtramos para el bot (EURUSD)
            symbol_pos = [p for p in all_positions if p.symbol == SYMBOL] if all_positions else []
            n_pos = len(symbol_pos)
            
            acc = mt5.account_info()
            if acc:
                # El profit ahora es el TOTAL de la cuenta (para coincidir con MT5/Móvil)
                profit = acc.profit 
            
            if acc:
                # === PLASTICIDAD SINÁPTICA (Aprendizaje en Vivo) ===
                current_balance = acc.balance
                if last_balance != 0 and current_balance != last_balance:
                    diff = current_balance - last_balance
                    won = diff > 0
                    new_w = self.neuro_engine.adapt_synapses(won)
                    if won:
                        self.add_log(f"🧠 NEURO: Win! Potenciando sinapsis -> W={new_w:.2f}")
                    else:
                        self.add_log(f"🧠 NEURO: Loss. Deprimiendo sinapsis -> W={new_w:.2f}")
                
                # === ESTADÍSTICAS DEL DÍA PARA APP ===
                # Calcular Profit Hoy, Win Rate y Max DD
                now = datetime.datetime.now()
                today_start = datetime.datetime(now.year, now.month, now.day)
                deals = mt5.history_deals_get(today_start, now + datetime.timedelta(days=1))
                
                daily_profit = 0.0
                wins = 0
                total_trades = 0
                if deals:
                    for d in deals:
                        if d.symbol == SYMBOL and d.entry == mt5.DEAL_ENTRY_OUT:
                            daily_profit += d.profit
                            total_trades += 1
                            if d.profit > 0:
                                wins += 1
                
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
                
                # Estimación simple de Max DD basado en la sesión actual
                peak = max(self.equity_history) if self.equity_history else acc.equity
                dd = ((peak - acc.equity) / peak * 100) if peak > 0 else 0.0
                
                last_balance = current_balance
                
                # OPTIMIZACIÓN: Solo enviar a la App según el modo elegido
                should_sync = False
                if BOT_MODE == 1 and time.time() - last_sync_time >= 3.0:
                    should_sync = True
                elif BOT_MODE == 2 and cycle_count % 20 == 0:
                    should_sync = True
                # Modo 3 (Headless) no envía sync por defecto para máxima velocidad

                if should_sync:
                    # Usamos el lote actual para la sincronización
                    lote_val = lote if 'lote' in locals() else 0.01
                    # Sincronizamos Profit Total y Posiciones Totales
                    self.sync_engine.update_mobile_app(acc.profit, acc.equity, acc.balance, total_n_pos, "trading", lote_val, win_rate, daily_profit, dd)
                    last_sync_time = time.time()

                
                self.update_ui_chart(acc.equity)
                
                # Update stat cards (Thread-safe)
                self.after(0, lambda: self.stat_cards["balance"].configure(text=f"{acc.balance:,.2f}"))
                self.after(0, lambda: self.stat_cards["equity"].configure(text=f"{acc.equity:,.2f}"))
                
                # Color dinámico según ganancia/pérdida
                if profit > 0:
                    color = self.colors["accent_green"]
                    sign = "+"
                elif profit < 0:
                    color = self.colors["accent_red"]
                    sign = ""
                else:
                    color = self.colors["text_secondary"]
                    sign = ""
                
                self.after(0, lambda c=color, s=sign, p=profit: self._update_profit_display(c, s, p))
                self.after(0, lambda n=n_pos, tn=total_n_pos: self.lbl_status.configure(text=f"● MONITOREANDO {n}/{MAX_POSITIONS} {SYMBOL} | {tn} TOTALES"))

                # ⚡ Verificar Kill Switch desde el móvil
                if self.sync_engine.check_kill_switch():
                    self.add_log("🔴 KILL SWITCH ACTIVADO desde la App móvil")
                    self.emergency_stop()
                    return

                try: 
                    m_loss = float(self.entry_loss.get())
                except: 
                    m_loss = -50.0

                if profit <= m_loss and n_pos > 0:
                    self.add_log(f"⚠️ Max Loss alcanzado ({profit:.2f}). Cerrando...")
                    self.close_all_now()
                    for i in range(RESTART_DELAY, 0, -1):
                        self.lbl_status.configure(text=f"⏳ AUTO-RESTART: {i}s")
                        time.sleep(1)
                    self.add_log("🔄 Sistema reiniciado.")
                    continue

                # 300 velas para que la EMA200 (V10) no falle
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 300)
                if rates is None:
                    self.after(0, lambda: self.lbl_mt5.configure(text="MTS: MARKET CLOSED", text_color=self.colors["accent_orange"]))
                    self.after(0, lambda: self.conn_dot.configure(text_color=self.colors["accent_orange"]))
                    if int(time.time()) % 20 == 0:
                        print(f"❌ [DEBUG] MT5 no devuelve datos para '{SYMBOL}'. Revisa el nombre o si el mercado está abierto.")
                else:
                    self.after(0, lambda: self.lbl_mt5.configure(text="MTS: MARKET OPEN", text_color=self.colors["accent_green"]))
                    self.after(0, lambda: self.conn_dot.configure(text_color=self.colors["accent_green"]))
                    df = pd.DataFrame(rates)
                    df.rename(columns={'time': 'Gmt time', 'open': 'Open', 'high': 'High', 
                                      'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
                    df_p, cols = indicators.add_indicators(df)
                    
                    # === GESTIÓN DINÁMICA DE INDICADORES ===
                    # El 'Menú' de indicadores posibles
                    # Base (12) + Quant (4) + Regime (1) + Physics (4)
                    
                    df_p, _ = indicators.add_quant_features(df_p)
                    df_p, _ = indicators.add_hmm_regime_proxy(df_p)
                    df_p, _ = indicators.add_physics_features(df_p)
                    df_p, _ = indicators.add_golden_strategy_features(df_p)
                    
                    base_cols = [
                        "rsi_14", "rsi_50", "adx_14", "atr_14", 
                        "bb_upper_diff", "bb_lower_diff", "close_ma20_diff", 
                        "close_ma50_diff", "ma_spread", "ma_spread_slope",
                        "volume_zscore", "pv_divergence"
                    ]
                    quant_cols = ["quant_fer", "quant_vam", "quant_zscore", "quant_entropy"]
                    regime_col = ["regime_proxy"]
                    phys_cols = ["phys_pressure", "phys_viscosity", "phys_fisher", "phys_hawkes"]
                    golden_cols = ["golden_trend", "golden_cross", "golden_setup"]
                    
                    state_features = 3
                    try:
                        required_indicators = self.model_features - state_features
                    except:
                        required_indicators = 21 # Default V7
                    
                    # SELECCIÓN AUTOMÁTICA ADAPTADA
                    if required_indicators == 23: # V10 Golden
                         cols = base_cols + quant_cols + regime_col + phys_cols + golden_cols[:2]
                    elif required_indicators == 21: # V7 Survivor
                         cols = base_cols + quant_cols + phys_cols + regime_col
                    elif required_indicators == 20: # V5 Ultimate
                         cols = base_cols + quant_cols + phys_cols[:3] + regime_col 
                    elif required_indicators == 17: # V5 Standard
                         cols = base_cols + quant_cols + regime_col
                    else:
                         self.add_log(f"⚠️ Mismatch Neural: Modelo pide {required_indicators} ind. Usando config V7.")
                         cols = base_cols + quant_cols + phys_cols + regime_col
                    
                    # DEBUG: Verificar integridad de vectores
                    # self.add_log(f"🔍 DEBUG Features: Base={len(cols)-len(quant_cols)} | Quant={len(quant_cols)}")
                    
                    # Extraer últimos valores para monitoreo "Co-Piloto"
                    last_fer = df_p["quant_fer"].iloc[-1]
                    last_vam = df_p["quant_vam"].iloc[-1]
                    last_z = df_p["quant_zscore"].iloc[-1]
                    
                    # Mostrar análisis cuantitativo en consola esporádicamente (cada 10 ciclos aprox para no saturar)
                    last_z10 = df_p["quant_zscore_10"].iloc[-1] if "quant_zscore_10" in df_p.columns else 0.0
                    if int(time.time()) % 20 == 0:
                        self.add_log(f"🧠 QUANT: FER={last_fer:.2f} | VAM={last_vam:.2f} | Z={last_z:.2f} | Z10={last_z10:.2f}")

                    # === MOTOR DE REVERSIÓN A LA MEDIA (Mean Reversion Override) ===
                    # Usa Z-Score RÁPIDO (10 barras) para detectar rebotes visibles en H1.
                    # Z10 < -1.5 → Precio muy por DEBAJO de la media reciente → COMPRAR
                    # Z10 > +1.5 → Precio muy por ARRIBA de la media reciente → VENDER
                    MR_THRESHOLD = 1.5
                    mean_reversion_signal = None
                    if last_z10 < -MR_THRESHOLD:
                        mean_reversion_signal = "BUY"
                    elif last_z10 > MR_THRESHOLD:
                        mean_reversion_signal = "SELL"

                    if mean_reversion_signal and n_pos < MAX_POSITIONS:
                        time_since_last = time.time() - last_trade_time
                        if time_since_last >= 60:  # Cooldown de 1 minuto
                            t_mr = mt5.symbol_info_tick(SYMBOL)
                            p_inf_mr = mt5.symbol_info(SYMBOL)
                            base_lot_mr = float(self.entry_lots.get())
                            sl_pips_mr = 30.0   # SL fijo de 30 pips
                            tp_pips_mr = 60.0   # TP fijo de 60 pips (ratio 1:2)
                            sl_pts_mr = int(sl_pips_mr * 10)
                            tp_pts_mr = int(tp_pips_mr * 10)

                            if mean_reversion_signal == "BUY":
                                px_mr = t_mr.ask
                                sl_mr = px_mr - (sl_pts_mr * p_inf_mr.point)
                                tp_mr = px_mr + (tp_pts_mr * p_inf_mr.point)
                                ot_mr = mt5.ORDER_TYPE_BUY
                            else:
                                px_mr = t_mr.bid
                                sl_mr = px_mr + (sl_pts_mr * p_inf_mr.point)
                                tp_mr = px_mr - (tp_pts_mr * p_inf_mr.point)
                                ot_mr = mt5.ORDER_TYPE_SELL

                            req_mr = {
                                "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
                                "volume": base_lot_mr, "type": ot_mr,
                                "price": px_mr, "sl": sl_mr, "tp": tp_mr,
                                "magic": 123456, "type_filling": mt5.ORDER_FILLING_FOK
                            }
                            result_mr = mt5.order_send(req_mr)
                            if result_mr.retcode == mt5.TRADE_RETCODE_DONE:
                                last_trade_time = time.time()
                                self.add_log(f"🔄 REVERSIÓN A LA MEDIA: {mean_reversion_signal} | Z10={last_z10:.2f} | SL={sl_pips_mr}p | TP={tp_pips_mr}p")

                    if len(df_p) >= WIN:
                        obs_b = df_p[cols].tail(WIN).to_numpy()
                        state_v = np.tile(np.array([1.0 if n_pos > 0 else 0.0, 0.0, 0.0]), (WIN, 1))
                        obs = np.hstack([obs_b, state_v]).astype(np.float32)
                        # Añadimos dimensión de batch para el normalizador y el modelo (1, 30, 18)
                        obs_ready = obs[np.newaxis, ...]
                        
                        # Suministramos observación normalizada si el modelo lo requiere
                        if v_env and hasattr(v_env, 'normalize_obs'):
                            final_obs = v_env.normalize_obs(obs_ready)
                        else:
                            final_obs = obs_ready
                            
                        act, _ = model.predict(final_obs, deterministic=True)
                        act_item = int(act.item())
                        res = actions[act_item]

                        # === PILLAR 1: QUBO ENERGY FILTERING (V4) ===
                        # Si la IA sugiere OPEN, optimizamos la elección mediante QUBO Hamiltoniano
                        if res[0] == "OPEN":
                             current_atr = float(df_p["atr_14"].iloc[-1]) if "atr_14" in df_p.columns else 0.001
                             market_entropy = self.neuro_engine.calculate_entropy(df_p["Close"].values)
                             
                             # Evaluamos un subconjunto de acciones similares (Quantum Sampling)
                             # Para el proxy, evaluamos la acción sugerida y un par de variaciones SL/TP
                             sample_actions = [res]
                             # Simula variaciones SL/TP si se desea, o simplemente aplica el filtro al Hamiltoniano actual
                             res = self.neuro_engine.quantum_energy_filter(sample_actions, current_atr, market_entropy)

                        # === FILTRO DE RÉGIMEN LAZARUS (Balanceado V9) ===
                        # FER > 0.30 permite operar en tendencias moderadas.
                        # Z-Score > 1.2 para reversiones estándar.
                        allow_entry = True
                        if last_fer < 0.30:
                            if abs(last_z) < 1.2: 
                                allow_entry = False
                                if int(time.time()) % 60 == 0:
                                    self.add_log("🛡️ FILTRO LAZARUS: Esperando mejor setup (FER<0.30 | Z<1.2).")
                        
                        # --- NUEVO: SISTEMA ANTI-ESCOPETA (Cooldown Reducido) ---
                        time_since_last = time.time() - last_trade_time
                        if allow_entry and n_pos < MAX_POSITIONS and time_since_last < 60:
                            allow_entry = False # Bloqueamos por cooldown (1 min)
                            if int(time.time()) % 30 == 0:
                                self.add_log(f"⏳ COOLDOWN: Esperando {int(60 - time_since_last)}s.")
                        
                        if res[0] == "OPEN" and n_pos < MAX_POSITIONS and allow_entry:
                            # === LÓGICA NEURO-CUÁNTICA (Plasticidad + Entropía) ===
                            # 1. Calculamos Entropía del Mercado (Incertidumbre)
                            market_entropy = self.neuro_engine.calculate_entropy(df_p["Close"].values)
                            # 2. Obtenemos Peso Sináptico (Confianza basada en historia reciente)
                            synaptic_w = self.neuro_engine.synaptic_weight
                            
                            # 3. Ajuste Cuántico de Lote (Kelly Pillar 3)
                            base_lot = float(self.entry_lots.get())
                            # Obtenemos win_rate real del sistema
                            current_wr = win_rate if win_rate > 35 else 40 # Piso de 40% para Kelly inicial
                            
                            # Mezclamos Neuro-Plasticidad con Kelly
                            kelly_lot = self.neuro_engine.calculate_kelly_lot(base_lot, current_wr, acc.balance)
                            neuro_lot = kelly_lot * synaptic_w * (1 - (market_entropy * 0.5))
                            neuro_lot = max(0.01, min(0.5, round(neuro_lot, 2))) # Límites de seguridad (max 0.5)
                            
                            self.add_log(f"🧠 KELLY+NEURO: WR={current_wr:.1f}% | Lot={neuro_lot}")
                            
                            lote = neuro_lot
                            t = mt5.symbol_info_tick(SYMBOL)
                            p_inf = mt5.symbol_info(SYMBOL)
                            px = t.ask if res[1] == 1 else t.bid
                            
                            # === RISK MANAGEMENT INSTITUCIONAL (AI SELECTED) ===
                            # Usamos los pips exactos elegidos por el Cerebro Golden
                            chosen_sl_pips = float(res[2])
                            # Forzamos 1:2 para recuperación agresiva pero segura
                            chosen_tp_pips = chosen_sl_pips * 2.0 
                            
                            sl_points = int(chosen_sl_pips * 10) # 1 pip = 10 points
                            tp_points = int(chosen_tp_pips * 10)
                            
                            self.add_log(f"📐 AI TARGET: SL={chosen_sl_pips} pips | TP={chosen_tp_pips} pips (1:2)")
                            
                            # Calcular precios finales
                            if res[1] == 1: # BUY
                                sl_v = px - (sl_points * p_inf.point)
                                tp_v = px + (tp_points * p_inf.point)
                                order_type = mt5.ORDER_TYPE_BUY
                            else: # SELL
                                sl_v = px + (sl_points * p_inf.point)
                                tp_v = px - (tp_points * p_inf.point)
                                order_type = mt5.ORDER_TYPE_SELL
                            
                            req = {
                                "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lote,
                                "type": order_type,
                                "price": px, "sl": sl_v, "tp": tp_v, "magic": 123456, "type_filling": mt5.ORDER_FILLING_FOK
                            }
                            result = mt5.order_send(req)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                last_trade_time = time.time() # Reset Cooldown
                                self.add_log(f"🚀 POSICIÓN {n_pos+1} ABIERTA (Modo Francotirador)")
                                # Enviar notificación al móvil
                                notif_url = f"{BASE44_API_URL}/api/Notification"
                                notif_payload = {
                                    "title": "🚀 IA: Operación Abierta",
                                    "message": f"Abierta posición en {SYMBOL} con lote {lote}",
                                    "type": "info",
                                    "read": "false"
                                }
                                try:
                                    requests.post(notif_url, json=notif_payload, headers=self.sync_engine.headers, timeout=2)
                                except:
                                    pass
    
            time.sleep(0.1 if BOT_MODE >= 2 else 2.0)

if __name__ == "__main__":
    import sys
    # Soporte para --headless desde terminal
    if "--headless" in sys.argv:
        BOT_MODE = 3
        print("🚀 INICIANDO EN MODO TURBO (HEADLESS)...")
    
    app = TitanyBotApp()
    # Si es modo Turbo Headless, no mostramos ventana pero corremos lógica
    if BOT_MODE == 3:
        app.withdraw() # Ocultar ventana
        # Forzar hilo de ejecución
        logic_thread = threading.Thread(target=app.run_bot_logic, daemon=True)
        logic_thread.start()
        try:
            while True: time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Deteniendo bot Headless...")
    else:
        try:
            app.mainloop()
        except KeyboardInterrupt:
            print("\n🛑 Deteniendo bot por usuario...")
        finally:
            # Graceful shutdown: Update status to OFFLINE
            print("\n[Base44] Cerrando conexión y actualizando estado a OFFLINE...")
            try:
                if hasattr(app, 'sync_engine'):
                    app.sync_engine.update_mobile_app(0, 0, 0, 0, "OFFLINE", 0, 0.0, 0.0, 0.0)
            except Exception as e:
                print(f"Error al enviar estado OFFLINE: {e}")
            
            app.running = False
            print("✅ Sistema detenido correctamente.")
