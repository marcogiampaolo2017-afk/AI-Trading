import sys
import os
import time
import threading
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from env import indicators
    from core.genetic_engine import GeneticPredictor
except ImportError:
    pass                                              
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5                                 
FUTURE_CANDLES = 15
PAST_CANDLES = 30
N_SIMULATIONS = 300                       
class MultiverseUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🕷️ TITANY MULTIVERSE - QUANTUM COLLAPSE")
        self.geometry("1200x800")
        self.configure(fg_color="#000000")             
        self.running = True
        if not mt5.initialize():
            print("Fallo al conectar con MT5")
        
        self.genetic_engine = GeneticPredictor()
        self.golden_path = None
        self.prediction_locked = False
        self.lock_price = 0
        self.checkpoints = []
        
        self.setup_ui()
        self.start_multiverse_engine()
    def setup_ui(self):
        header = ctk.CTkFrame(self, fg_color="#0a0a0a", height=60, corner_radius=0)
        header.pack(fill="x", side="top")
        ctk.CTkLabel(header, text="🌌 THE MULTIVERSE: PROBABILITY COLLAPSE", 
                    font=("Consolas", 24, "bold"), text_color="#00e5ff").pack(pady=15)
        self.stats_label = ctk.CTkLabel(header, text="Z-SCORE: -- | ATR: -- | UNIVERSOS: 300", 
                                        font=("Consolas", 12), text_color="#a855f7")
        self.stats_label.pack(pady=0)
        self.plot_frame = ctk.CTkFrame(self, fg_color="#050505", corner_radius=10)
        self.plot_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.fig = Figure(figsize=(10, 6), facecolor='#050505')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#050505')
        self.ax.tick_params(colors="#475569", labelsize=8)
        for spine in ['top', 'right']:
            self.ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            self.ax.spines[spine].set_color("#1e293b")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    def start_multiverse_engine(self):
        self.thread = threading.Thread(target=self.multiverse_loop, daemon=True)
        self.thread.start()
    def calculate_current_state(self):
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, PAST_CANDLES + 50)
        if rates is None: return None, None, None
        df = pd.DataFrame(rates)
        df.rename(columns={'time': 'Gmt time', 'open': 'Open', 'high': 'High', 
                          'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        df['atr'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        atr = df['atr'].iloc[-1]
        ma50 = df['Close'].rolling(50).mean()
        std50 = df['Close'].rolling(50).std()
        z_score = (df['Close'].iloc[-1] - ma50.iloc[-1]) / (std50.iloc[-1] + 1e-8)
        closes = df['Close'].tail(PAST_CANDLES).values
        return closes, atr, z_score
    def multiverse_loop(self):
        while self.running:
            try:
                closes, atr, z = self.calculate_current_state()
                if closes is not None:
                    self.after(0, self.render_multiverse, closes, atr, z)
            except Exception as e:
                print(f"Error multiverso: {e}")
            time.sleep(1.0)                                                          
    def render_multiverse(self, closes, atr, z):
        self.ax.clear()
        self.ax.set_facecolor('#050505')
        
        x_past = np.arange(-PAST_CANDLES + 1, 1)
        self.ax.plot(x_past, closes, color="#ef4444", linewidth=2.5, zorder=5, label="Sagrada Línea Temporal")
        
        last_price = closes[-1]
        x_future = np.arange(0, FUTURE_CANDLES + 1)
        
        # 🧬 Obtener SESGO GENÉTICO para filtrar universos
        genetic_bias = self.genetic_engine.get_signal()
        drift = -z * (atr * 0.1)
        
        # PERSISTENCIA: Si ya tenemos una línea dorada y el precio no se ha desviado > 70% ATR
        if self.golden_path is not None and self.prediction_locked:
            deviation = abs(last_price - self.lock_price)
            if deviation < (atr * 0.7):
                # Mantener la predicción anterior visualmente (desplazada al precio actual)
                offset = last_price - self.golden_path[0]
                visual_path = self.golden_path + offset
                self.ax.plot(x_future, visual_path, color="#FFD700", linewidth=4.0, zorder=7, label="PROFECÍA DORADA (BLOQUEADA)")
            else:
                self.prediction_locked = False
                self.golden_path = None

        paths_with_weights = []
        for i in range(N_SIMULATIONS):
            # Simulamos con mayor dispersión para buscar realidades genéticas
            random_steps = np.random.normal(0, atr * 0.5, FUTURE_CANDLES)
            steps = np.insert(drift + random_steps, 0, 0)
            path = last_price + np.cumsum(steps)
            
            # Cálculo de congruencia (Gaussian Mean + Genetic Alignment)
            end_price = path[-1]
            dist_mean = (end_price - (last_price - (z * atr)))
            weight_gaussian = np.exp(-0.5 * (dist_mean / (atr * 2))**2)
            
            # Alineación con Genética: Si el sesgo es alcista, premiar universos alcistas
            direction = 1 if (end_price > last_price) else -1
            gen_bonus = 1.5 if (direction == (1 if genetic_bias > 0 else -1)) else 0.5
            
            final_weight = weight_gaussian * gen_bonus
            paths_with_weights.append((path, final_weight))

        # Ordenar por peso para encontrar el Top 3 Dorado
        paths_with_weights.sort(key=lambda x: x[1], reverse=True)
        top_3 = paths_with_weights[:3]
        others = paths_with_weights[3:]

        # Dibujar universos descartados (Cyan tenue)
        for p, w in others:
            alpha = max(0.01, min(0.15, w * 0.2))
            self.ax.plot(x_future, p, color="#00e5ff", alpha=alpha, linewidth=0.8)

        # Dibujar y Promediar el TOP 3 DORADO
        if len(top_3) > 0:
            golden_sum = np.zeros_like(top_3[0][0])
            for p, w in top_3:
                self.ax.plot(x_future, p, color="#FFD700", alpha=0.6, linewidth=1.5, zorder=6)
                golden_sum += p
            
            # La Trayectoria Maestra (Promedio del Top 3)
            master_golden = golden_sum / 3.0
            self.ax.plot(x_future, master_golden, color="#FFD700", linewidth=4.5, zorder=8, label="LÍNEA DORADA (TOP 3)")
            
            if not self.prediction_locked:
                self.golden_path = master_golden
                self.lock_price = last_price
                self.prediction_locked = True
                # Extraer Checkpoints (Puntos de Inflexión)
                idx_max = np.argmax(master_golden)
                idx_min = np.argmin(master_golden)
                self.checkpoints = [
                    {"step": idx_max, "price": master_golden[idx_max], "type": "H"},
                    {"step": idx_min, "price": master_golden[idx_min], "type": "L"}
                ]

        self.stats_label.configure(text=f"Z-SCORE: {z:+.2f} | GENETIC BIAS: {genetic_bias:+.2f} | UNIVERSOS DORADOS ACTIVOS")
        self.ax.legend(loc="upper left", facecolor="#0a0a0a", edgecolor="#1e293b", labelcolor="#f8fafc", fontsize=9)
        self.canvas.draw()
    def on_closing(self):
        self.running = False
        mt5.shutdown()
        self.destroy()
if __name__ == "__main__":
    app = MultiverseUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
