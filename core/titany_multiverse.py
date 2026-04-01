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
        drift = -z * (atr * 0.1) 
        best_path = None
        best_weight = -1
        paths = []
        for i in range(N_SIMULATIONS):
            random_steps = np.random.normal(0, atr * 0.4, FUTURE_CANDLES)
            steps = np.insert(drift + random_steps, 0, 0)
            path = last_price + np.cumsum(steps)
            end_price = path[-1]
            distance_from_mean = (end_price - (last_price - (z * atr)))
            weight = np.exp(-0.5 * (distance_from_mean / (atr * 2))**2)
            alpha = max(0.01, min(0.4, weight * 0.5))
            self.ax.plot(x_future, path, color="#00e5ff", alpha=alpha, linewidth=1.0)
            paths.append(path)
            if weight > best_weight:
                best_weight = weight
                best_path = path
        if best_path is not None:
            self.ax.plot(x_future, best_path, color="#ffffff", linewidth=3.5, alpha=0.9, zorder=6, label="Línea Evolutiva Prime")
            self.ax.scatter([FUTURE_CANDLES], [best_path[-1]], color="#ffffff", s=50, zorder=7)
            self.ax.scatter([FUTURE_CANDLES], [best_path[-1]], color="#00e5ff", s=200, alpha=0.4, zorder=6)
        self.stats_label.configure(text=f"Z-SCORE: {z:+.2f} | ATR: {atr:.5f} | UNIVERSOS SUPERVIVIENTES: {N_SIMULATIONS}")
        self.ax.legend(loc="upper left", facecolor="#0a0a0a", edgecolor="#1e293b", labelcolor="#f8fafc")
        self.canvas.draw()
    def on_closing(self):
        self.running = False
        mt5.shutdown()
        self.destroy()
if __name__ == "__main__":
    app = MultiverseUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
