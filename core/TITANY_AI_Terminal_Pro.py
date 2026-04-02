import sys
import os
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
import datetime
import subprocess
from PIL import Image                                                
import queue                 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import indicators
import json
from tkinter import messagebox
class AcademyWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("🎓 TITANY AI: ACADEMIA DE ENTRENAMIENTO INTENSIVO")
        self.geometry("1100x850")
        self.colors = parent.colors
        self.configure(fg_color=self.colors["bg_panel"])
        self.training_process = None
        self.log_queue = queue.Queue()
        self._setup_ui()
        self._check_log_queue()
        self._update_visuals_loop()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    def _setup_ui(self):
        top_bar = ctk.CTkFrame(self, fg_color=self.colors["bg_panel"], height=60, corner_radius=0)
        top_bar.pack(fill="x")
        ctk.CTkLabel(top_bar, text="🎓 AFK ACADEMY: ENTRENAMIENTO INTENSIVO", 
                    font=("Segoe UI", 16, "bold"), text_color=self.colors["text_primary"]).pack(side="left", padx=25)
        self.btn_run = ctk.CTkButton(top_bar, text="🚀 EMPEZAR ENTRENAMIENTO", 
                                    fg_color=self.colors["accent_green"], hover_color="#27ae60",
                                    command=self._start_training)
        self.btn_run.pack(side="right", padx=25)
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        stats_frame = ctk.CTkFrame(content, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(0, 15))
        self.academy_stats = {}
        stats_cfg = [
            ("EXPERIENCIA ACUMULADA", "0", self.colors["accent_cyan"]),
            ("VUELTAS DE EXAMEN", "0", self.colors["accent_purple"]),
            ("ERROR DE APRENDIZAJE", "---", self.colors["accent_orange"]),
            ("VELOCIDAD DE ESTUDIO", "0", "orange")
        ]
        for name, val, color in stats_cfg:
            card = ctk.CTkFrame(stats_frame, fg_color=self.colors["bg_card"], corner_radius=10)
            card.pack(side="left", fill="both", expand=True, padx=5)
            ctk.CTkLabel(card, text=name, font=("Segoe UI", 10, "bold"), text_color="white").pack(pady=(10, 0))
            self.academy_stats[name] = ctk.CTkLabel(card, text=val, font=("Segoe UI", 24, "bold"), text_color=color)
            self.academy_stats[name].pack(pady=(5, 15))
        charts_frame = ctk.CTkFrame(content, fg_color="transparent")
        charts_frame.pack(fill="x", pady=(0, 15))
        self.fig_train = Figure(figsize=(5, 3), dpi=90, facecolor=self.colors["bg_card"])
        self.ax_train = self.fig_train.add_subplot(111)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=charts_frame)
        self.canvas_train.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5)
        self.fig_equity_afk = Figure(figsize=(5, 3), dpi=90, facecolor=self.colors["bg_card"])
        self.ax_equity_afk = self.fig_equity_afk.add_subplot(111)
        self.canvas_equity_afk = FigureCanvasTkAgg(self.fig_equity_afk, master=charts_frame)
        self.canvas_equity_afk.get_tk_widget().pack(side="right", fill="both", expand=True, padx=5)
        log_frame = ctk.CTkFrame(content, fg_color=self.colors["bg_card"], corner_radius=15)
        log_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(log_frame, text="TERMINAL OUTPUT (STABLE-BASELINES-3)", 
                    font=("Segoe UI", 12, "bold"), text_color=self.colors["text_secondary"]).pack(pady=10)
        self.academy_log = ctk.CTkTextbox(log_frame, fg_color="#0a0f12", text_color="#2ecc71", 
                                        font=("Consolas", 12), border_width=1, border_color="#1c2a33")
        self.academy_log.pack(fill="both", expand=True, padx=15, pady=15)
    def _start_training(self):
        if self.training_process:
            if messagebox.askyesno("Confirmación", "¿Deseas detener el entrenamiento actual?"):
                self._stop_training()
            return
        self.academy_log.configure(state="normal")
        self.academy_log.delete("1.0", "end")
        self.academy_log.insert("end", "[SISTEMA] Iniciando sesión: Cargando Motor Cuántico y Dataset (Espere un momento...)\n")
        self.academy_log.configure(state="disabled")
        self.btn_run.configure(text="⌛ ENTRENANDO...", fg_color="#e67e22")
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(root_dir, "research", "train_agent.py")
        venv_python = sys.executable
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        self.training_process = subprocess.Popen(
            [venv_python, script_path, "--headless"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, creationflags=subprocess.CREATE_NO_WINDOW
        )
        threading.Thread(target=self._stream_logs, daemon=True).start()
    def _stream_logs(self):
        if not self.training_process or not self.training_process.stdout:
            return
        for line in iter(self.training_process.stdout.readline, ''):
            self.log_queue.put(line)
        try:
            self.training_process.stdout.close()
        except:
            pass
    def _check_log_queue(self):
        while not self.log_queue.empty():
            line = self.log_queue.get()
            if "Gym has been unmaintained" in line:
                if getattr(self, "_gym_warning_shown", False): continue
                self._gym_warning_shown = True
            elif "Please upgrade to Gymnasium" in line or "Users of this version" in line or "See the migration guide" in line:
                continue
            is_table_line = "|" in line or "----------------" in line
            current_time = time.time()
            if not hasattr(self, "_last_table_time"): self._last_table_time = 0
            if not hasattr(self, "_last_table_session"): self._last_table_session = 0
            if not hasattr(self, "_suppress_current_table"): self._suppress_current_table = False
            show_line = True
            if is_table_line:
                if (current_time - self._last_table_time) > 2.0:
                    if (current_time - self._last_table_session) < 20.0:
                        self._suppress_current_table = True
                    else:
                        self._suppress_current_table = False
                        self._last_table_session = current_time
                self._last_table_time = current_time
                if self._suppress_current_table:
                    show_line = False
            if show_line:
                self.academy_log.configure(state="normal")
                self.academy_log.insert("end", line)
                if self.academy_log.index("end-1c") != "1.0":
                    self.academy_log.see("end")
                self.academy_log.configure(state="disabled")
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    k, v = parts[1].lower(), parts[2]
                    try:
                        if "total_timesteps" in k: 
                            self.academy_stats["EXPERIENCIA ACUMULADA"].configure(text=f"{int(v):,}")
                        elif "iterations" in k:
                            self.academy_stats["VUELTAS DE EXAMEN"].configure(text=v)
                        elif "fps" in k: 
                            self.academy_stats["VELOCIDAD DE ESTUDIO"].configure(text=v)
                        elif "loss" in k: 
                            self.academy_stats["ERROR DE APRENDIZAJE"].configure(text=v[:6])
                    except:
                        pass
        self.after(100, self._check_log_queue)
    def _update_visuals_loop(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        npz_path = os.path.join(root_dir, "logs", "evaluations.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                ts, res = data["timesteps"], data["results"]
                m_rew = res.mean(axis=1)
                self.ax_train.clear()
                self.ax_train.set_facecolor(self.colors["bg_card"])
                self.ax_train.plot(ts, m_rew, color=self.colors["accent_cyan"], marker="o", markersize=4)
                self.ax_train.set_title("CURVA DE REWARD", color=self.colors["accent_cyan"], fontsize=9)
                self.ax_equity_afk.clear()
                self.ax_equity_afk.set_facecolor(self.colors["bg_card"])
                self.ax_equity_afk.plot(ts, np.cumsum(m_rew), color=self.colors["accent_green"], marker="o", markersize=4)
                self.ax_equity_afk.set_title("EQUITY SIMULADO", color=self.colors["accent_green"], fontsize=9)
                for ax in [self.ax_train, self.ax_equity_afk]:
                    ax.tick_params(colors="#556677", labelsize=7)
                self.canvas_train.draw()
                self.canvas_equity_afk.draw()
            except: pass
        self.after(15000, self._update_visuals_loop)
    def _stop_training(self):
        if self.training_process:
            # Create a flag file to tell the training process to stop and save cleanly
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            flag_path = os.path.join(root_dir, "stop_training.flag")
            try:
                with open(flag_path, "w") as f:
                    f.write("STOP")
                self.academy_log.configure(state="normal")
                self.academy_log.insert("end", "\n🛑 Señal de stop enviada. Esperando guardado de seguridad (puede tardar unos segundos)...\n")
                self.academy_log.configure(state="disabled")
            except: pass
            
            # Start a fallback timer to forcefully kill if it hangs
            def force_kill():
                if self.training_process:
                    self.training_process.terminate()
                    self.training_process = None
                    self.btn_run.configure(text="🚀 EMPEZAR ENTRENAMIENTO", fg_color=self.colors["accent_green"])
                    self.academy_log.configure(state="normal")
                    self.academy_log.insert("end", "💀 Proceso finalizado de golpe.\n")
                    self.academy_log.configure(state="disabled")
            self.after(20000, force_kill)  # Le damos 20 segundos para guardar el modelo pesado
    def _on_closing(self):
        if self.training_process:
            if messagebox.askyesno("Academia Activa", "El entrenamiento sigue en curso. ¿Deseas detenerlo antes de salir?"):
                self._stop_training()
        self.destroy()
BASE44_API_URL = "https://app.base44.com/api/apps/696fe84f14c617992088dd7d/entities"
BASE44_API_KEY = "82e4ca558e8546f89859b4a3dba1e1cf"
STATUS_ROW_ID = "696fe9543bd9c31ec5fcb8af" 
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
MAX_POSITIONS = 1
RESTART_DELAY = 180
WIN = 30
INTEGRITY_TARGET = 100.0                                              
INTEGRITY_MIN_PROFIT = 0.15  # Ganancia mínima sobre el target para activar el cierre (evita falsos disparos por el spread)
SL_OPTS = [10, 20, 30, 50, 80, 100]
TP_OPTS = [10, 20, 30, 50, 80, 100]
BOT_MODE = 2 
class TitanySync:
    def __init__(self, url, key, row_id):
        self.url = f"{url}/BotStatus/{row_id}"
        self.headers = {
            "api_key": key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
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
    def update_mobile_app(self, profit, equity, balance, n_trades, state, lot, win_rate, daily_profit, max_dd, weekly_profit=0.0):
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
                "weekly_profit": float(round(weekly_profit, 2)),
                "max_dd": float(round(max_dd, 2)),
                "last_update": time.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            response = requests.put(self.url, json=payload, headers=self.headers, timeout=15)
            if response.status_code not in (200, 201, 204):
                print(f"[Base44] PUT fallido (código {response.status_code}): {response.text[:200]}")
        except Exception:
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
                    pass
            else:
                pass
        except Exception:
            pass
        return False
class NeuroQuantEngine:
    """
    Este es el motor matemático 'oculto' detrás de la interfaz.
    Calcula entropía, filtros cuánticos y gestión de riesgo Kelly.
    """
    def __init__(self):
        self.synaptic_weight = 1.0                                            
        self.learning_rate = 0.05                            
        self.last_trade_time = 0.0                        
        self.memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_memory")
        self.memory_file = os.path.join(self.memory_dir, "synaptic_memory.json")
        self._load_memory()
    def _load_memory(self):
        """Carga la adaptación pasada desde el disco."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.synaptic_weight = data.get("synaptic_weight", 1.0)
                    self.last_trade_time = data.get("last_trade_time", 0.0)
                    print(f"🧠 MEMORIA: AGA-MORA cargó confianza W={self.synaptic_weight:.2f} y Cooldown previo.")
        except:
            self.synaptic_weight = 1.0
    def _save_memory(self):
        """Guarda la adaptación actual para no olvidar jamas."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump({
                    "synaptic_weight": self.synaptic_weight, 
                    "last_trade_time": self.last_trade_time,
                    "last_update": str(datetime.datetime.now())
                }, f)
        except:
            pass
    def calculate_entropy(self, price_series, window=50):
        """
        Calcula la Entropía de Shannon del mercado (Incertidumbre).
        Concepto: Física estadística aplicada a flujos financieros.
        Higher Entropy = Higher Chaos/Uncertainty.
        """
        try:
            if len(price_series) < window:
                return 0.5
            relevant_data = np.array(price_series[-window:])
            returns = np.diff(np.log(relevant_data))
            hist, _ = np.histogram(returns, bins=20, density=True)
            p_k = hist + 1e-10                 
            p_k = p_k / np.sum(p_k)                          
            entropy = -np.sum(p_k * np.log(p_k))
            max_ent = np.log(20)
            normalized_entropy = min(1.0, entropy / max_ent)
            return normalized_entropy
        except Exception as e:
            return 0.5                            
    def calculate_kelly_lot(self, base_lot, win_rate, balance, rr=2.0):
        """
        Calcula el lote óptimo usando el Criterio de Kelly (Fraccional 20%).
        Fórmula: K = (p*R - (1-p)) / R
        """
        p = win_rate / 100.0
        if p <= 0.33: return base_lot                                                     
        k_full = (p * rr - (1 - p)) / rr
        k_safe = k_full * 0.2
        potential_lot = (balance * k_safe) / 3000                      
        return max(base_lot, round(potential_lot, 2))
    def adapt_synapses(self, won_last_trade):
        """
        Simula Plasticidad Sináptica (Hebbian Learning).
        """
        if won_last_trade:
            self.synaptic_weight = min(2.0, self.synaptic_weight * 1.1)                     
        else:
            self.synaptic_weight = max(0.5, self.synaptic_weight * 0.8)                  
        self._save_memory()                                   
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
            sl_dist = act[2]
            tp_dist = act[3]
            risk_energy = (sl_dist * current_atr * 1000)
            chaos_energy = entropy * 50
            reward_potential = (tp_dist / (sl_dist + 1e-6)) * 20
            if sl_dist == 0:                                      
                energy = float('inf')
            else:
                energy = risk_energy + chaos_energy - reward_potential
            if energy < min_energy:
                min_energy = energy
                best_action = act
        return best_action if best_action else actions_list[0]
class TitanyBotApp(ctk.CTk):
    """
    Controla toda la parte visual: Gráficos, botones, logs y
    la conexión de hilos con la lógica de trading.
    """
    def __init__(self):
        super().__init__()
        self.title("AGA-MORA AI - NEURAL ADAPTIVE SYSTEM")
        self.geometry("1600x900")
        self.configure(fg_color="#0a0e1a")
        self.equity_history = []
        self._load_equity_history()                                 
        self.sync_engine = TitanySync(BASE44_API_URL, BASE44_API_KEY, STATUS_ROW_ID)
        self.neuro_engine = NeuroQuantEngine()                          
        try:
            from genetic_engine import GeneticPredictor
            self.genetic_ai = GeneticPredictor()
        except Exception as e:
            print(f"⚠️ Error al inyectar Genoma Evolutivo: {e}")
            self.genetic_ai = None
        try:
            from titany_multiverse import MultiverseUI
            self.add_log("🕷️ MULTIVERSO CUÁNTICO: Instanciado en background.")
        except Exception as e:
            self.add_log(f"⚠️ Error en motor multiverso: {e}")
        self.shadow_status = "Esperando ticks..."
        try:
            import sys as _sys
            _core_dir = os.path.dirname(os.path.abspath(__file__))
            if _core_dir not in _sys.path:
                _sys.path.insert(0, _core_dir)
            import titany_continuous_trainer as _shadow_mod
            self.shadow_trainer_thread = threading.Thread(
                target=_shadow_mod.build_shadow_trainer,
                daemon=True,
                name="ShadowTrainer"
            )
            self.shadow_trainer_thread.start()
            self.add_log("🌑 SHADOW TRAINER: Aprendizaje perpetuo ACTIVO.")
        except Exception as e:
            self.add_log(f"⚠️ Shadow Trainer: {e}")
        self.last_terminal_sync = "📡 [AGA-MORA] SYNC: Esperando datos..."
        self.last_terminal_quant = "🧠 [AGA-MORA] QUANT: Esperando datos..."
        self.running = True                                              
        self.setup_ui()
        self.bot_thread = threading.Thread(target=self.run_bot_logic, daemon=True)
        self.bot_thread.start()
        self.log_queue = queue.Queue()
        self.training_process = None                                          
        self._check_log_queue()
        self.update_time()
        self.animate_neural()
    def _load_equity_history(self):
        """Carga el gráfico histórico para verlo el sábado."""
        try:
            memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_memory")
            history_path = os.path.join(memory_dir, "equity_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.equity_history = json.load(f)
                    self.equity_history = self.equity_history[-1000:]
        except:
            self.equity_history = []
    def _save_equity_history(self):
        """Guarda el gráfico actual para la posteridad."""
        try:
            memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_memory")
            history_path = os.path.join(memory_dir, "equity_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.equity_history, f)
        except:
            pass
    def setup_ui(self):
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
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, 
                                    fg_color=self.colors["bg_panel"], 
                                    border_width=1, border_color=self.colors["border_subtle"])
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=120)
        brand_frame.pack(fill="x", pady=(20, 10))
        brand_frame.pack_propagate(False)
        ctk.CTkLabel(brand_frame, text="🛡️", font=("Segoe UI Emoji", 48)).pack()
        ctk.CTkLabel(brand_frame, text="AGA-MORA AI", 
                    font=("Segoe UI", 22, "bold"), text_color=self.colors["accent_cyan"]).pack()
        ctk.CTkLabel(brand_frame, text="NEURAL ADAPTIVE SYSTEM", 
                    font=("Segoe UI", 9), text_color=self.colors["text_secondary"]).pack()
        ctk.CTkFrame(self.sidebar, height=1, fg_color=self.colors["border_subtle"]).pack(fill="x", padx=20, pady=15)
        self.btn_academy = ctk.CTkButton(self.sidebar, text="🎓 ACADEMIA AFK", 
                                        fg_color=self.colors["bg_card"],
                                        hover_color=self.colors["accent_purple"],
                                        text_color="white",
                                        font=("Segoe UI", 12, "bold"),
                                        command=self._open_academy_window)
        self.btn_academy.pack(fill="x", padx=20, pady=(0, 15))
        self.academy_window = None
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
        ctk.CTkFrame(self.sidebar, height=1, fg_color=self.colors["border_subtle"]).pack(fill="x", padx=20, pady=(0,10))
        shadow_frame = ctk.CTkFrame(self.sidebar, fg_color=self.colors["bg_card"],
                                   corner_radius=12, height=80)
        shadow_frame.pack(fill="x", padx=15, pady=(0, 10))
        shadow_frame.pack_propagate(False)
        shadow_header = ctk.CTkFrame(shadow_frame, fg_color="transparent")
        shadow_header.pack(fill="x", padx=15, pady=(10,0))
        ctk.CTkLabel(shadow_header, text="🌑 SHADOW TRAINER",
                    font=("Segoe UI", 11, "bold"), text_color="#a78bfa").pack(side="left")
        self.shadow_dot = ctk.CTkLabel(shadow_header, text="●",
                                       font=("Segoe UI", 16), text_color=self.colors["accent_green"])
        self.shadow_dot.pack(side="right")
        self.lbl_shadow = ctk.CTkLabel(shadow_frame, text="Acumulando experiencia...",
                                       font=("Segoe UI", 10), text_color=self.colors["text_secondary"])
        self.lbl_shadow.pack(padx=15)
        self._update_shadow_status()
        inputs_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        inputs_frame.pack(fill="x", padx=15, pady=10)
        self.create_premium_input("💎 LOT SIZE", "0.01", self.colors["accent_cyan"], "entry_lots", inputs_frame)
        self.create_premium_input("🛡️ MAX LOSS", "-10.0", self.colors["accent_red"], "entry_loss", inputs_frame)
        self.create_premium_input("🎯 MAX POS", "1", self.colors["accent_orange"], "entry_maxpos", inputs_frame)
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
        self.main_container = ctk.CTkFrame(self, fg_color=self.colors["bg_dark"])
        self.main_container.pack(side="right", fill="both", expand=True)
        self.live_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.live_frame.pack(fill="both", expand=True)
        self._setup_live_dashboard()
    def _open_academy_window(self):
        """Abre la academia en una ventana flotante independiente (Adiós al Lag)."""
        if hasattr(self, "academy_win") and self.academy_win.winfo_exists():
            self.academy_win.focus()
        else:
            self.academy_win = AcademyWindow(self)
    def _setup_live_dashboard(self):
        parent = self.live_frame
        top_bar = ctk.CTkFrame(parent, fg_color=self.colors["bg_panel"], height=60, corner_radius=0)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)
        ctk.CTkLabel(top_bar, text="📊 LIVE TRADING DASHBOARD", 
                    font=("Segoe UI", 16, "bold"), text_color=self.colors["text_primary"]).pack(side="left", padx=25, pady=15)
        self.lbl_telemetry_top = ctk.CTkLabel(top_bar, text="", 
                                         font=("Consolas", 14, "bold"), text_color=self.colors["accent_cyan"])
        self.lbl_telemetry_top.pack(side="left", padx=20)
        self.lbl_time = ctk.CTkLabel(top_bar, text=time.strftime('%H:%M:%S'), 
                                     font=("Consolas", 14), text_color=self.colors["text_secondary"])
        self.lbl_time.pack(side="right", padx=25)
        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
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
        ctk.CTkLabel(profit_inner, text="EUR", 
                    font=("Segoe UI", 20), text_color=self.colors["text_secondary"]).pack()
        self.lbl_status = ctk.CTkLabel(self.f_profit, text="● MONITOREANDO SISTEMA...", 
                                       font=("Segoe UI", 11, "bold"), text_color=self.colors["accent_cyan"])
        self.lbl_status.pack(side="bottom", pady=15)
        stats_frame = ctk.CTkFrame(content, fg_color="transparent", height=100)
        stats_frame.pack(fill="x", pady=(0, 15))
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
            ctk.CTkLabel(card, text=title, font=("Segoe UI", 10), text_color=self.colors["text_secondary"]).pack(pady=(15, 5))
            lbl = ctk.CTkLabel(card, text=value, font=("Segoe UI", 24, "bold"), text_color=color)
            lbl.pack(pady=(0, 15))
            self.stat_cards[key] = lbl
        self.tabs = ctk.CTkTabview(content, fg_color=self.colors["bg_card"], 
                                   corner_radius=15, border_width=1, border_color=self.colors["border_subtle"])
        self.tabs.pack(fill="both", expand=True, pady=(0, 15))
        tab_equity = self.tabs.add("📈 EQUITY CURVE")
        tab_multi = self.tabs.add("🌌 QUANTUM MULTIVERSE")
        self.fig = Figure(figsize=(10, 3), dpi=100, facecolor=self.colors["bg_card"])
        self.ax = self.fig.add_subplot(111)
        self.setup_chart_style()
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_equity)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.fig_mv = Figure(figsize=(10, 3), dpi=100, facecolor=self.colors["bg_card"])
        self.ax_mv = self.fig_mv.add_subplot(111)
        self.ax_mv.set_facecolor(self.colors["bg_card"])
        self.ax_mv.tick_params(colors=self.colors["text_secondary"], labelsize=8)
        for spine in ['top', 'right']: self.ax_mv.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']: self.ax_mv.spines[spine].set_color(self.colors["border_subtle"])
        self.canvas_mv = FigureCanvasTkAgg(self.fig_mv, master=tab_multi)
        self.canvas_mv.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        bottom_frame = ctk.CTkFrame(content, fg_color="transparent", height=140)
        bottom_frame.pack(fill="x")
        bottom_frame.pack_propagate(False)
        log_frame = ctk.CTkFrame(bottom_frame, fg_color=self.colors["bg_card"], corner_radius=12, border_width=1)
        log_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.log_txt = ctk.CTkTextbox(log_frame, fg_color="transparent", text_color=self.colors["text_secondary"], font=("Consolas", 10))
        self.log_txt.pack(fill="both", expand=True, padx=10, pady=10)
        btn_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent", width=200)
        btn_frame.pack(side="right", fill="y")
        self.btn_stop = ctk.CTkButton(btn_frame, text="⚠️\nEMERGENCY\nCLOSE ALL", fg_color=self.colors["accent_red"], command=self.emergency_stop, corner_radius=15)
        self.btn_stop.pack(fill="both", expand=True)
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
        """Pequeña animación de estatus y actualización de tiempo de forma segura."""
        if not hasattr(self, "lbl_shadow") or not hasattr(self, "shadow_dot"):
            return
        timestamp = time.strftime('%H:%M:%S')
        try:
            self.after(0, lambda t=timestamp: self.lbl_shadow.configure(text=f"V13 activo | Última mejora: {t}"))
            current_color = self.shadow_dot.cget("text_color")
            next_color = "#00ff88" if current_color == self.colors["accent_green"] else self.colors["accent_green"]
            self.after(0, lambda c=next_color: self.shadow_dot.configure(text_color=c))
            if hasattr(self, "neural_dot"):
                curr_n = self.neural_dot.cget("text_color")
                next_n = self.colors["accent_purple"] if curr_n == self.colors["accent_green"] else self.colors["accent_green"]
                self.after(0, lambda n=next_n: self.neural_dot.configure(text_color=n))
        except: pass
        if self.running:
            self.after(2000, self.animate_neural)
    def update_ui_chart(self, val):
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
            pass                                      
    def _log_market_experience(self, final_obs, action_index, df_p):
        """ Guarda el estado del mercado (Experience Replay) para el Shadow Trainer """
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_dir = os.path.join(root_dir, "brain_memory")
            os.makedirs(memory_dir, exist_ok=True)
            exp_file = os.path.join(memory_dir, "market_experience.csv")
            ts = int(time.time())
            close_price = df_p["Close"].iloc[-1]
            obs_flat = final_obs.flatten().tolist()
            obs_str = "|".join([f"{x:.4f}" for x in obs_flat])
            row = f"{ts},{action_index},{close_price},{obs_str}\n"
            with open(exp_file, "a", encoding="utf-8") as f:
                f.write(row)
        except Exception:
            pass
    def _update_shadow_status(self):
        """Actualiza el indicador del Shadow Trainer en la sidebar cada 10s."""
        try:
            if hasattr(self, 'lbl_shadow'):
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                exp_file = os.path.join(root_dir, "brain_memory", "market_experience.csv")
                new_model = os.path.join(root_dir, "models", "model_eurusd_titany_v13_live.zip")
                if os.path.exists(new_model):
                    import datetime as _dt
                    mtime = os.path.getmtime(new_model)
                    last_train = _dt.datetime.fromtimestamp(mtime).strftime("%H:%M")
                    self.lbl_shadow.configure(text=f"V13 activo - Ultima mejora: {last_train}")
                    self.shadow_dot.configure(text_color="#00e5ff")
                elif os.path.exists(exp_file):
                    try:
                        with open(exp_file, "r") as f:
                            n = sum(1 for _ in f)
                        self.lbl_shadow.configure(text=f"Acumulando: {n}/100 ticks")
                        self.shadow_dot.configure(text_color="#f59e0b")
                    except:
                        pass
                else:
                    self.lbl_shadow.configure(text="Esperando datos del mercado...")
                    self.shadow_dot.configure(text_color="#6b7280")
        except Exception:
            pass
        self.after(10000, self._update_shadow_status)
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
            self.after(0, lambda: self.lbl_mt5.configure(text="MARKET: OFFLINE", text_color="#ef4444"))
            return
        model = None
        v_env = None
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(root_dir, "models")
            best_m_path = os.path.join(models_dir, "best_model.zip")
            model_name = "model_eurusd_titany_v12_sniper.zip"
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(best_m_path):
                model_path = best_m_path
                self.add_log("🎓 AI: Cargando mejor modelo detectado en la ACADEMIA.")
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            SL_OPTS_L = [10, 20, 30, 50, 80, 100]
            TP_OPTS_L = [10, 20, 30, 50, 80, 100]
            if model_path == best_m_path:
                SL_OPTS_L = [20, 50, 100]
                TP_OPTS_L = [40, 100, 200]
            def dummy_env_adaptive():
                from env.trading_env import ForexTradingEnv
                n_ind = 26 if "v12_sniper" in model_name else (23 if "v10" in model_name else 21)
                return ForexTradingEnv(pd.DataFrame(np.zeros((100, n_ind))), window_size=30, sl_options=SL_OPTS_L, tp_options=TP_OPTS_L)
            v_env = DummyVecEnv([dummy_env_adaptive])
            norm_path = os.path.join(models_dir, "vec_normalize.pkl")
            if os.path.exists(norm_path):
                try:
                    v_env = VecNormalize.load(norm_path, v_env)
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
                def dummy_env_v7():
                    from env.trading_env import ForexTradingEnv
                    return ForexTradingEnv(pd.DataFrame(np.zeros((100, 21))), window_size=30, sl_options=SL_OPTS_L, tp_options=TP_OPTS_L)
                v_env = DummyVecEnv([dummy_env_v7])
                model_path_bkp = os.path.join(models_dir, "best_models", "best_model.zip")
                if os.path.exists(os.path.join(models_dir, "model_eurusd_titany_v7_survivor.zip")):
                     model_path_bkp = os.path.join(models_dir, "model_eurusd_titany_v7_survivor.zip")
                model = RecurrentPPO.load(model_path_bkp, env=v_env)
                self.add_log(f"💎 AI: MODELO V7 SURVIVOR (Backup) cargado.")
            except Exception as e2:
                 self.add_log(f"❌ AI: CRITICAL - FALLO TOTAL DE CARGA: {e2}")
        try:
             self.model_features = model.observation_space.shape[-1]
             self.add_log(f"🧠 ARQUITECTURA DETECTADA: {self.model_features} Neuronas de Entrada.")
        except:
             self.model_features = 24             
        actions = [("HOLD", 0,0,0), ("CLOSE", 0,0,0)]
        for d in [0, 1]:
            for sl in SL_OPTS_L:
                for tp in TP_OPTS_L:
                    actions.append(("OPEN", d, float(sl), float(tp)))
        last_sync_time = 0
        last_trade_time = self.neuro_engine.last_trade_time                     
        last_balance = 0
        cycle_count = 0
        self.meta_diaria_alcanzada = False  # 🛡️ FIX: flag para bloquear entradas tras alcanzar la meta diaria
        lstm_states = None
        is_warmed_up = False
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        memory_dir = os.path.join(root_dir, "brain_memory")
        lstm_path = os.path.join(memory_dir, "lstm_memory_v12.pkl")
        try:
            if os.path.exists(lstm_path):
                import pickle
                with open(lstm_path, 'rb') as f:
                    lstm_states = pickle.load(f)
                is_warmed_up = True
                self.add_log("🧠 NEURO: Estados LSTM cargados. Memoria instantánea activa.")
            else:
                self.add_log("🧠 Iniciando Módulo de Consciencia Histórica (Lectura Profunda 500 candelas)...")
        except:
            self.add_log("⚠️ Error cargando LSTM. Iniciando warm-up estándar...")
        new_model_name = "best_model.zip"
        new_model_path = os.path.join(models_dir, new_model_name)
        last_model_mtime = 0
        if os.path.exists(new_model_path):
            last_model_mtime = os.path.getmtime(new_model_path)
        elif os.path.exists(os.path.join(models_dir, "model_eurusd_titany_v13_live.zip")):
             new_model_path = os.path.join(models_dir, "model_eurusd_titany_v13_live.zip")
             last_model_mtime = os.path.getmtime(new_model_path)
        while self.running:
            if cycle_count == 0:
                print("🟢 [AGA-MORA AI] Operación Esmeralda - Sistema de Grado Especial Iniciado.")
                if getattr(self, "genetic_ai", None) and getattr(self.genetic_ai, "compiled_func", None):
                    self.add_log(f"🧬 GENÉTICA: Cerebro evolutivo de DEAP ({len(self.genetic_ai.model)} nodos) ONLINE.")
                else:
                    self.add_log("⚠️ GENÉTICA: El ADN evolutivo está inactivo o falló la carga.")
                all_pos = mt5.positions_get()
                if all_pos:
                    print(f"🔍 [DEBUG] Encontradas {len(all_pos)} posiciones totales.")
                    for p in all_pos:
                        print(f"👉 Símbolo en MT5: '{p.symbol}' | Ticket: {p.ticket}")
                else:
                    print("🔍 [DEBUG] No se detectan posiciones abiertas en ninguna cuenta.")
                last_err = mt5.last_error()
                if last_err[0] != 1:
                    print(f"❌ [MT5 ERROR] Código: {last_err[0]} | Descripción: {last_err[1]}")
            cycle_count += 1
            if cycle_count % 30 == 0:
                if os.path.exists(new_model_path):
                    current_mtime = os.path.getmtime(new_model_path)
                    if current_mtime > last_model_mtime:
                        self.add_log("🔄 Shadow Trainer evolucionó la red. Inyectando V13_Live en caliente...")
                        try:
                            model = RecurrentPPO.load(new_model_path, env=v_env)
                            last_model_mtime = current_mtime
                            self.add_log("✅ Neuro-Inyección exitosa. Operando con máxima experiencia.")
                        except Exception as e:
                            self.add_log(f"⚠️ Rechazo de Neuro-Inyección V13: {e}")
            all_positions = mt5.positions_get()
            total_n_pos = len(all_positions) if all_positions else 0
            symbol_pos = [p for p in all_positions if p.symbol == SYMBOL] if all_positions else []
            n_pos = len(symbol_pos)
            acc = mt5.account_info()
            if acc:
                profit = acc.profit 
            if acc:
                current_balance = acc.balance
                if last_balance != 0 and current_balance != last_balance:
                    diff = current_balance - last_balance
                    won = diff > 0
                    new_w = self.neuro_engine.adapt_synapses(won)
                    if won:
                        self.add_log(f"🧠 NEURO: Win! Potenciando sinapsis -> W={new_w:.2f}")
                    else:
                        self.add_log(f"🧠 NEURO: Loss. Deprimiendo sinapsis -> W={new_w:.2f}")
                now = datetime.datetime.now()
                today_start = datetime.datetime(now.year, now.month, now.day)
                deals_today = mt5.history_deals_get(today_start, now + datetime.timedelta(days=1))
                realized_today = sum([d.profit for d in deals_today if d.entry == mt5.DEAL_ENTRY_OUT]) if deals_today else 0.0
                daily_profit = realized_today + acc.profit                                             
                weekday = now.weekday()          
                week_start = today_start - datetime.timedelta(days=weekday)
                deals_week = mt5.history_deals_get(week_start, now + datetime.timedelta(days=1))
                weekly_profit = sum([d.profit for d in deals_week if d.entry == mt5.DEAL_ENTRY_OUT]) if deals_week else 0.0
                wins = sum([1 for d in deals_today if d.entry == mt5.DEAL_ENTRY_OUT and d.profit > 0]) if deals_today else 0
                total_trades_today = sum([1 for d in deals_today if d.entry == mt5.DEAL_ENTRY_OUT]) if deals_today else 0
                win_rate = (wins / total_trades_today * 100) if total_trades_today > 0 else 0.0
                peak = max(self.equity_history) if self.equity_history else acc.equity
                dd = ((peak - acc.equity) / peak * 100) if peak > 0 else 0.0
                last_balance = current_balance
                should_sync = False
                if BOT_MODE == 1 and time.time() - last_sync_time >= 3.0:
                    should_sync = True
                elif BOT_MODE == 2 and cycle_count % 20 == 0:
                    should_sync = True
                if should_sync:
                    lote_val = lote if 'lote' in locals() else 0.01
                    self.sync_engine.update_mobile_app(acc.profit, acc.equity, acc.balance, total_n_pos, "trading", lote_val, win_rate, daily_profit, dd, weekly_profit)
                    last_sync_time = time.time()
                    stats_str = f"Día: {daily_profit:+.2f}€ | Sem: {weekly_profit:+.2f}€"
                    self.last_terminal_sync = f"[{time.strftime('%H:%M:%S')}] 📡 [AGA-MORA] SYNC OK | {stats_str} | MT5: {SYMBOL}"
                    self.add_log(f"📡 [AGA-MORA] SYNC OK | {stats_str} | MT5: {SYMBOL}")
                # 🛡️ FIX BUG OVERTRADING: Verificar meta diaria ya alcanzada
                if self.meta_diaria_alcanzada:
                    pass  # No abrir nada más hoy
                elif hasattr(self, 'last_equity_internal'):
                    # CONDICIÓN CORREGIDA: requiere ganancia mínima real (> spread + INTEGRITY_MIN_PROFIT)
                    # Antes: acc.equity >= INTEGRITY_TARGET (se disparaba con el spread de 1-2 pips)
                    # Ahora: acc.equity >= INTEGRITY_TARGET + INTEGRITY_MIN_PROFIT (ganancia real confirmada)
                    real_profit_on_trade = acc.equity - INTEGRITY_TARGET
                    if (self.last_equity_internal < INTEGRITY_TARGET and 
                            acc.equity >= (INTEGRITY_TARGET + INTEGRITY_MIN_PROFIT)):
                        self.add_log(f"🛡️ INTEGRIDAD: Ganancia real de +{real_profit_on_trade:.2f}€ confirmada. Asegurando capital...")
                        positions = mt5.positions_get(symbol=SYMBOL)
                        if positions is not None and len(positions) > 0:
                            self.add_log("⚔️ SECURE PROTOCOL: Ejecutando Cierre Forzado para bloquear ganancias.")
                            for pos in positions:
                                tick = mt5.symbol_info_tick(SYMBOL)
                                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": SYMBOL,
                                    "volume": pos.volume,
                                    "type": close_type,
                                    "position": pos.ticket,
                                    "price": close_price,
                                    "deviation": 20,
                                    "magic": 123456,
                                    "comment": "Integrity Rescue",
                                    "type_time": mt5.ORDER_TIME_GTC,               
                                    "type_filling": mt5.ORDER_FILLING_FOK,
                                }
                                res_cl = mt5.order_send(request)
                                if res_cl.retcode != mt5.TRADE_RETCODE_DONE:
                                    self.add_log(f"⚠️ Error al cerrar ticket {pos.ticket}: {res_cl.retcode}")
                                else:
                                    self.add_log(f"✅ Ticket CERRADO. Meta de {INTEGRITY_TARGET}€ Completada. (+{real_profit_on_trade:.2f}€)")
                            # 🔒 FIX: Activar flag para bloquear TODAS las entradas futuras de esta sesión
                            self.meta_diaria_alcanzada = True
                            self.add_log("🔒 SESIÓN COMPLETADA: Meta diaria asegurada. Bot en modo SOLO MONITOREO hasta mañana.")
                            self.after(0, lambda: self.lbl_status.configure(
                                text="🏆 META DIARIA ALCANZADA - Solo monitoreando",
                                text_color="#f59e0b"
                            ))
                        try:
                            notif_payload = {
                                "title": "🛡️ ¡Meta Diaria Completada!",
                                "message": f"Portafolio en {acc.equity:.2f}€ (+{real_profit_on_trade:.2f}€). Bot en pausa segura.",
                                "type": "success"
                            }
                            requests.post(f"{BASE44_API_URL}/api/Notification", json=notif_payload, headers=self.sync_engine.headers, timeout=2)
                        except: pass
                self.last_equity_internal = acc.equity
                self.update_ui_chart(acc.equity)
                self._save_equity_history()                             
                self.after(0, lambda: self.stat_cards["balance"].configure(text=f"{acc.balance:,.2f}"))
                self.after(0, lambda: self.stat_cards["equity"].configure(text=f"{acc.equity:,.2f}"))
                self.after(0, lambda: self.stat_cards["winrate"].configure(text=f"{win_rate:.0f}%"))
                
                # Color dinámico para TODAY
                day_col = self.colors["accent_green"] if daily_profit > 0 else (self.colors["accent_red"] if daily_profit < 0 else self.colors["accent_orange"])
                self.after(0, lambda c=day_col, d=daily_profit: self.stat_cards["daily"].configure(text=f"{d:+.2f}", text_color=c))
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
                        self.after(0, lambda v=i: self.lbl_status.configure(text=f"⏳ AUTO-RESTART: {v}s"))
                        time.sleep(1)
                    self.add_log("🔄 Sistema reiniciado.")
                    continue
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 500)
                if rates is None:
                    self.after(0, lambda: self.lbl_mt5.configure(text="MTS: MARKET CLOSED", text_color=self.colors["accent_orange"]))
                    self.after(0, lambda: self.conn_dot.configure(text_color=self.colors["accent_orange"]))
                else:
                    self.after(0, lambda: self.lbl_mt5.configure(text="MTS: MARKET OPEN", text_color=self.colors["accent_green"]))
                    self.after(0, lambda: self.conn_dot.configure(text_color=self.colors["accent_green"]))
                    df = pd.DataFrame(rates)
                    df.rename(columns={'time': 'Gmt time', 'open': 'Open', 'high': 'High', 
                                      'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
                    df_p, cols = indicators.add_indicators(df)
                    last_fer, last_vam, last_z, last_z10, last_vsa = 0.0, 0.0, 0.0, 0.0, 1.0
                    df_p, _ = indicators.add_quant_features(df_p)
                    df_p, _ = indicators.add_hmm_regime_proxy(df_p)
                    df_p, _ = indicators.add_physics_features(df_p)
                    df_p, _ = indicators.add_golden_strategy_features(df_p)
                    df_p, _ = indicators.add_volume_sniper_features(df_p)
                    df_p, _ = indicators.add_vision_360_features(df_p)                                 
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
                    sniper_cols = ["vsa_ratio_norm", "volume_delta", "climax_candle"]
                    state_features = 3
                    try:
                        required_indicators = self.model_features - state_features
                    except:
                        required_indicators = 21             
                    if required_indicators == 26:                  
                         cols = base_cols + quant_cols + phys_cols + golden_cols + sniper_cols
                    elif required_indicators == 23:             
                         cols = base_cols + quant_cols + regime_col + phys_cols + golden_cols[:2]
                    elif required_indicators == 21:              
                         cols = base_cols + quant_cols + phys_cols + regime_col
                    elif required_indicators == 20:              
                         cols = base_cols + quant_cols + phys_cols[:3] + regime_col 
                    elif required_indicators == 17:              
                         cols = base_cols + quant_cols + regime_col
                    else:
                         self.add_log(f"⚠️ Mismatch Neural: Modelo pide {required_indicators} ind. Usando config V7.")
                         cols = base_cols + quant_cols + phys_cols + regime_col
                    last_fer = df_p["quant_fer"].iloc[-1]
                    last_vam = df_p["quant_vam"].iloc[-1]
                    last_z = df_p["quant_zscore"].iloc[-1]
                    last_z10 = df_p["quant_zscore_10"].iloc[-1] if "quant_zscore_10" in df_p.columns else 0.0
                    if len(df_p) >= WIN:
                        obs_b = df_p[cols].tail(WIN).to_numpy()
                        state_v = np.tile(np.array([1.0 if n_pos > 0 else 0.0, 0.0, 0.0]), (WIN, 1))
                        obs = np.hstack([obs_b, state_v]).astype(np.float32)
                        obs_ready = obs[np.newaxis, ...]
                        if v_env and hasattr(v_env, 'normalize_obs'):
                            final_obs = v_env.normalize_obs(obs_ready)
                        else:
                            final_obs = obs_ready
                        if not is_warmed_up:
                            self.add_log("📥 Cargando memoria de mercado (LSTM Context)...")
                            for i in range(len(df_p) - 100, len(df_p) - WIN):
                                window_past = df_p[cols].iloc[i-WIN:i].to_numpy()
                                state_past = np.zeros((WIN, 3))
                                obs_past_full = np.hstack([window_past, state_past]).astype(np.float32)
                                obs_past = obs_past_full[np.newaxis, ...]
                                if v_env and hasattr(v_env, 'normalize_obs'):
                                    obs_past = v_env.normalize_obs(obs_past)
                                _, lstm_states = model.predict(obs_past, state=lstm_states, deterministic=True)
                            is_warmed_up = True
                            self.add_log("✅ Consciencia de mercado COMPLETADA. Sniper en posición.")
                        act, lstm_states = model.predict(final_obs, state=lstm_states, deterministic=True)
                        if cycle_count % 50 == 0:                                               
                            try:
                                import pickle
                                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                memory_dir = os.path.join(root_dir, "brain_memory")
                                lstm_save_path = os.path.join(memory_dir, "lstm_memory_v12.pkl")
                                with open(lstm_save_path, 'wb') as f:
                                    pickle.dump(lstm_states, f)
                            except: pass
                        act_item = int(act.item())
                        res = actions[act_item]
                        self.after(0, lambda obs=final_obs, act=act_item, df=df_p.tail(1): 
                                   self._log_market_experience(obs, act, df))
                        if res[0] == "OPEN":
                         # ══════════════════════════════════════════════════════════
                         # 🌊 PASO 0: CONTEXTO MACRO H4 — Se calcula PRIMERO
                         # ══════════════════════════════════════════════════════════
                         h4_bull = False  # EMA50 > EMA200 +15p → macro alcista
                         h4_bear = False  # EMA50 < EMA200 -15p → macro bajista
                         h4_gap  = 0.0   # pips entre EMA50 y EMA200 (+ = alcista)
                         try:
                             _r4 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 220)
                             if _r4 is not None and len(_r4) >= 200:
                                 _c4   = pd.DataFrame(_r4)["close"]
                                 _e50  = _c4.ewm(span=50,  adjust=False).mean().iloc[-1]
                                 _e200 = _c4.ewm(span=200, adjust=False).mean().iloc[-1]
                                 h4_gap  = (_e50 - _e200) / 0.0001
                                 h4_bull = h4_gap  >  15
                                 h4_bear = h4_gap  < -15
                         except:
                             pass
                         allow_entry = True
                         if last_fer < 0.30:
                             if abs(last_z) < 1.2:
                                 allow_entry = False
                                 if int(time.time()) % 60 == 0:
                                     self.add_log("🛡️ FILTRO LAZARUS: Esperando mejor setup (FER<0.30 | Z<1.2).")
                         vsa_confirm = False
                         if "vsa_ratio_norm" in df_p.columns:
                             last_vsa = df_p["vsa_ratio_norm"].iloc[-1]
                             if last_vsa > 1.1:
                                 vsa_confirm = True
                             elif abs(h4_gap) > 40:  # Tendencia macro muy fuerte: umbral VSA reducido
                                 vsa_confirm = last_vsa > 0.40
                         if res[0] == "OPEN" and not vsa_confirm:
                             allow_entry = False
                             if int(time.time()) % 60 == 0:
                                 self.add_log(f"🕵️ VSA FILTER: Volumen débil ({last_vsa:.2f}) | H4={h4_gap:.0f}p")
                         # ─── REVERSIÓN / MOMENTUM (ajustado al contexto H4) ──────────
                         if res[0] == "OPEN" and allow_entry:
                             if res[1] == 1:   # === BUY ===
                                 if h4_bear and last_z10 >= 0.0:  # Macro bajista: bloquear BUY salvo reversión profunda
                                     allow_entry = False
                                     if int(time.time()) % 60 == 0:
                                         self.add_log(f"📉 MACRO-BEAR: BUY bloqueado (Z10={last_z10:.2f}, H4={h4_gap:.0f}p)")
                                 elif last_z10 > 3.5:
                                     allow_entry = False
                                     if int(time.time()) % 60 == 0:
                                         self.add_log(f"📉 REVERSIÓN: BUY en techo extremo (Z10={last_z10:.2f})")
                             else:   # === SELL ===
                                 if h4_bull:
                                     if last_z10 >= -3.5:  # Macro alcista: bloquear SELL salvo caída muy extrema
                                         allow_entry = False
                                         if int(time.time()) % 60 == 0:
                                             self.add_log(f"🌊 MACRO-BULL: SELL bloqueado (Z10={last_z10:.2f}, H4=+{h4_gap:.0f}p)")
                                     else:
                                         _lc, _pc = df_p['Close'].iloc[-1], df_p['Close'].iloc[-2]
                                         if _lc > _pc:
                                             allow_entry = False
                                             if int(time.time()) % 60 == 0:
                                                 self.add_log(f"📈 MACRO-BULL: SELL bloqueado, rebote iniciado (Z10={last_z10:.2f})")
                                         else:
                                             if int(time.time()) % 60 == 0:
                                                 self.add_log(f"⚡ MOMENTUM SELL (bull track, Z10={last_z10:.2f}, H4=+{h4_gap:.0f}p)")
                                 elif last_z10 < -4.0:
                                     _lc, _pc = df_p['Close'].iloc[-1], df_p['Close'].iloc[-2]
                                     if _lc > _pc:
                                         allow_entry = False
                                         if int(time.time()) % 60 == 0:
                                             self.add_log(f"📈 REVERSIÓN: SELL bloqueado, rebote confirmado (Z10={last_z10:.2f})")
                                     else:
                                         if int(time.time()) % 60 == 0:
                                             self.add_log(f"⚡ MOMENTUM SELL: Z10={last_z10:.2f} sin rebote.")
                         # ─── ANTI-COHETE (umbrales dinámicos según macro H4) ──────────
                         if res[0] == "OPEN" and allow_entry:
                             if "close_ma50_diff" in df_p.columns:
                                 last_ma50_diff = df_p["close_ma50_diff"].iloc[-1]
                                 if h4_bull:
                                     sell_thr, buy_thr = 0.0015, -0.0050  # Bull: DIP-BUY hasta -50p, SELL bloqueado desde +15p
                                 elif h4_bear:
                                     sell_thr, buy_thr = 0.0050, -0.0015  # Bear: RALLY-SELL hasta +50p, BUY bloqueado desde -15p
                                 else:
                                     sell_thr, buy_thr = 0.0025, -0.0025  # Neutral: umbrales originales
                                 if last_ma50_diff > sell_thr and res[1] == 0:
                                     allow_entry = False
                                     if int(time.time()) % 30 == 0:
                                         self.add_log(f"🚫 ANTI-COHETE: SELL bloqueado (+{last_ma50_diff*10000:.0f}p sobre MA50)")
                                 elif last_ma50_diff < buy_thr and res[1] == 1:
                                     allow_entry = False
                                     if int(time.time()) % 30 == 0:
                                         self.add_log(f"🚫 ANTI-COHETE: BUY bloqueado ({last_ma50_diff*10000:.0f}p bajo MA50)")
                                 elif res[1] == 0 and last_ma50_diff < -0.0040:
                                     allow_entry = False
                                     if int(time.time()) % 30 == 0:
                                         self.add_log(f"🚫 FOMO: SELL bloqueado, el precio ya cayó demasiado ({last_ma50_diff*10000:.0f}p bajo MA50)")
                                 elif res[1] == 1 and last_ma50_diff > 0.0040:
                                     allow_entry = False
                                     if int(time.time()) % 30 == 0:
                                         self.add_log(f"🚫 FOMO: BUY bloqueado, el precio ya subió demasiado (+{last_ma50_diff*10000:.0f}p sobre MA50)")
                        if res[0] == "OPEN" and allow_entry:
                            last_pa_eng = df_p["pa_engulfing"].iloc[-1] if "pa_engulfing" in df_p.columns else 0
                            last_pa_pin = df_p["pa_pinbar"].iloc[-1] if "pa_pinbar" in df_p.columns else 0
                            last_ict_fvg = df_p["ict_fvg"].iloc[-1] if "ict_fvg" in df_p.columns else 0
                            last_ict_sw = df_p["ict_sweep"].iloc[-1] if "ict_sweep" in df_p.columns else 0
                            if res[1] == 0:       
                                if last_pa_pin == 1 or last_pa_eng == 1:
                                    allow_entry = False
                                    if int(time.time()) % 30 == 0:
                                        self.add_log("👁️ VISION 360: Bloqueando VENTA. Rechazo Alcista (Pinbar/Engulfing).")
                            elif res[1] == 1:      
                                if last_pa_pin == -1 or last_pa_eng == -1:
                                    allow_entry = False
                                    if int(time.time()) % 30 == 0:
                                        self.add_log("👁️ VISION 360: Bloqueando COMPRA. Rechazo Bajista (Pinbar/Engulfing).")
                            if allow_entry:
                                if res[1] == 0:       
                                    if last_ict_fvg == 1 or last_ict_sw == 1:
                                        allow_entry = False
                                        if int(time.time()) % 30 == 0:
                                            self.add_log("🏛️ ICT FILTER: Bloqueando VENTA. Precio rebotando en Liquidity Sweep o FVG Alcista.")
                                elif res[1] == 1:      
                                    if last_ict_fvg == -1 or last_ict_sw == -1:
                                        allow_entry = False
                                        if int(time.time()) % 30 == 0:
                                            self.add_log("🏛️ ICT FILTER: Bloqueando COMPRA. Precio rebotando en Liquidity Sweep o FVG Bajista.")
                        if res[0] == "OPEN" and allow_entry and getattr(self, "genetic_ai", None):
                            gen_signal = self.genetic_ai.get_signal()
                            if gen_signal != 0.0:
                                if res[1] == 1 and gen_signal < -20:
                                    allow_entry = False
                                    if int(time.time()) % 30 == 0:
                                        self.add_log(f"🧬 GENETIC FILTER: Bloqueando COMPRA. El Genoma detectó VENTA masiva ({gen_signal:.0f}).")
                                elif res[1] == 0 and gen_signal > 20:
                                    allow_entry = False
                                    if int(time.time()) % 30 == 0:
                                        self.add_log(f"🧬 GENETIC FILTER: Bloqueando VENTA. El Genoma detectó COMPRA masiva ({gen_signal:.0f}).")
                        if res[0] == "OPEN" and allow_entry:
                            try:
                                past_closes = df_p['Close'].values[-14:]
                                atr_m = (max(past_closes) - min(past_closes)) if len(past_closes) > 0 else 0.0010
                                # H4 macro tiene prioridad sobre Z10 mean-reversion
                                if h4_bull and res[1] == 1:
                                    drift = (abs(h4_gap) * 0.000001) + (atr_m * 0.08)   # BUY en macro alcista: drift positivo
                                elif h4_bear and res[1] == 0:
                                    drift = -((abs(h4_gap) * 0.000001) + (atr_m * 0.08)) # SELL en macro bajista: drift negativo
                                elif (last_z10 < -2.0 and res[1] == 0) or (last_z10 > 2.0 and res[1] == 1):
                                    drift = last_z10 * (atr_m * 0.05)   # Momentum Z10
                                else:
                                    drift = -last_z10 * (atr_m * 0.1)   # Mean-reversion estándar
                                hits_sl = 0
                                sl_dist = min(20.0, float(res[2])) / 10000.0
                                for _ in range(300):
                                    steps = drift + np.random.normal(0, atr_m * 0.4, 15)
                                    path = np.cumsum(steps)
                                    if res[1] == 1:      
                                        if np.min(path) <= -sl_dist: hits_sl += 1
                                    else:       
                                        if np.max(path) >= sl_dist: hits_sl += 1
                                prob_muerte = hits_sl / 300.0
                                if prob_muerte > 0.55:
                                    allow_entry = False
                                    if int(time.time()) % 30 == 0:
                                        self.add_log(f"🌌 MULTIVERSO: {int(prob_muerte*100)}% prob SL — entrada cancelada.")
                                else:
                                    if int(time.time()) % 60 == 0:
                                        self.add_log(f"🌌 MULTIVERSO: {int((1-prob_muerte)*100)}% supervivencia ✓ | H4={h4_gap:.0f}p")
                            except Exception as e:
                                pass
                        time_since_last = time.time() - last_trade_time
                        if allow_entry and n_pos < MAX_POSITIONS and time_since_last < 300:
                            allow_entry = False                                  
                            if int(time.time()) % 30 == 0:
                                 self.add_log(f"⏳ COOLDOWN: Esperando {int(300 - time_since_last)}s para evitar ráfagas en misma vela.")
                        # 🛡️ FIX: Bloquear entrada si la meta diaria ya fue alcanzada
                        if self.meta_diaria_alcanzada:
                            allow_entry = False
                        if res[0] == "OPEN" and n_pos < MAX_POSITIONS and allow_entry:
                            market_entropy = self.neuro_engine.calculate_entropy(df_p["Close"].values)
                            synaptic_w = self.neuro_engine.synaptic_weight
                            base_lot = float(self.entry_lots.get())
                            current_wr = win_rate if win_rate > 35 else 40                                 
                            kelly_lot = self.neuro_engine.calculate_kelly_lot(base_lot, current_wr, acc.balance)
                            neuro_lot = kelly_lot * synaptic_w * (1 - (market_entropy * 0.5))
                            neuro_lot = max(0.01, min(0.5, round(neuro_lot, 2)))                                 
                            self.add_log(f"🧠 KELLY+NEURO: WR={current_wr:.1f}% | Lot={neuro_lot}")
                            lote = neuro_lot
                            t = mt5.symbol_info_tick(SYMBOL)
                            p_inf = mt5.symbol_info(SYMBOL)
                            px = t.ask if res[1] == 1 else t.bid
                            chosen_sl_pips = float(res[2])
                            if chosen_sl_pips > 20.0:
                                chosen_sl_pips = 20.0
                                self.add_log(f"🛡️ DISCIPLINE SHIELD: Reduciendo Stop Loss a máximo de {chosen_sl_pips} pips.")
                            chosen_tp_pips = chosen_sl_pips * 2.0 
                            sl_points = int(chosen_sl_pips * 10)                    
                            tp_points = int(chosen_tp_pips * 10)
                            self.add_log(f"📐 AI TARGET: SL={chosen_sl_pips} pips | TP={chosen_tp_pips} pips (1:2)")
                            if res[1] == 1:      
                                sl_v = px - (sl_points * p_inf.point)
                                tp_v = px + (tp_points * p_inf.point)
                                order_type = mt5.ORDER_TYPE_BUY
                            else:       
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
                                last_trade_time = time.time()                 
                                self.neuro_engine.last_trade_time = last_trade_time                   
                                self.neuro_engine._save_memory()                     
                                self.add_log(f"🚀 POSICIÓN {n_pos+1} ABIERTA (Modo Francotirador)")
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
                        self.after(0, lambda: self._refresh_terminal_console(last_fer, last_vam, last_z, last_z10, last_vsa))
                        self.after(0, lambda: self.update_multiverse_ui(df_p))
            time.sleep(0.1 if BOT_MODE >= 2 else 2.0)
    def update_multiverse_ui(self, df_p):
        if not hasattr(self, 'tabs') or self.tabs.get() != "🌌 QUANTUM MULTIVERSE":
            return
        try:
            self.ax_mv.clear()
            self.ax_mv.set_facecolor(self.colors["bg_card"])
            PAST_CANDLES = 30
            FUTURE_CANDLES = 15
            N_SIMULATIONS = 300
            if len(df_p) < PAST_CANDLES: return
            closes = df_p['Close'].tail(PAST_CANDLES).values
            atr_m = df_p['atr'].iloc[-1] if 'atr' in df_p.columns else 0.0010
            z = float(df_p["z_score_10"].iloc[-1]) if "z_score_10" in df_p.columns else 0.0
            x_past = np.arange(-PAST_CANDLES + 1, 1)
            self.ax_mv.plot(x_past, closes, color="#ef4444", linewidth=2.0, zorder=5)
            last_price = closes[-1]
            x_future = np.arange(0, FUTURE_CANDLES + 1)
            drift = -z * (atr_m * 0.1)
            for _ in range(N_SIMULATIONS):
                steps = np.insert(drift + np.random.normal(0, atr_m * 0.4, FUTURE_CANDLES), 0, 0)
                path = last_price + np.cumsum(steps)
                distance_from_mean = (path[-1] - (last_price - (z * atr_m)))
                weight = np.exp(-0.5 * (distance_from_mean / (atr_m * 2))**2)
                alpha = max(0.01, min(0.3, weight * 0.5))
                self.ax_mv.plot(x_future, path, color="#00e5ff", alpha=alpha, linewidth=0.8)
            try:
                self.canvas_mv.draw()
            except Exception:
                pass                                                 
        except Exception as e: 
            print("Multiverse Draw Error:", e)
    def _refresh_terminal_console(self, last_fer, last_vam, last_z, last_z10, last_vsa):
        try:
            now = time.time()
            if not hasattr(self, '_last_quant_log_time'):
                self._last_quant_log_time = 0
            timestamp = time.strftime('%H:%M:%S')
            self.last_terminal_quant = f"[{timestamp}] \U0001f9e0 [AGA-MORA] QUANT: FER={last_fer:.2f} | VAM={last_vam:.2f} | Z={last_z:.2f} | Z10={last_z10:.2f} | VSA={last_vsa:.2f}"
            import sys
            sys.stdout.write("\r" + " " * 115 + "\r" + self.last_terminal_sync + "\n")
            sys.stdout.write("\r" + " " * 115 + "\r" + self.last_terminal_quant + "\033[F")
            sys.stdout.flush()
            if now - self._last_quant_log_time >= 30:
                self._last_quant_log_time = now
                self.add_log(f"\U0001f9e0 [AGA-MORA] QUANT: FER={last_fer:.2f} | VAM={last_vam:.2f} | Z={last_z:.2f} | Z10={last_z10:.2f} | VSA={last_vsa:.2f}")
        except Exception:
            pass
    def add_log(self, msg):
        if not msg or not msg.strip():
            return
        if "SYNC OK" in msg or "QUANT:" in msg:
            self.after(0, lambda: self._update_dynamic_ui_log(msg))
            return
        timestamp = time.strftime('%H:%M:%S')
        full_msg = f"[{timestamp}] {msg}"
        self.after(0, lambda: self._perform_log_update(full_msg + "\n"))
        import sys
        sys.stdout.write("\r" + " " * 115 + "\r")
        sys.stdout.write(full_msg + "\n")
        if hasattr(self, 'last_terminal_sync'):
            sys.stdout.write("\r" + " " * 115 + "\r" + self.last_terminal_sync + "\n")
        if hasattr(self, 'last_terminal_quant'):
            sys.stdout.write("\r" + " " * 115 + "\r" + self.last_terminal_quant + "\033[F")
        sys.stdout.flush()
    def _update_dynamic_ui_log(self, text):
        """Borra la línea vieja y escribe la nueva al final para que quede fija actualizándose."""
        try:
            self.log_txt.configure(state="normal")
            prefix = "📡 [AGA-MORA] SYNC OK" if "SYNC OK" in text else "🧠 [AGA-MORA] QUANT:"
            idx = self.log_txt.search(prefix, "1.0", "end")
            if idx:
                self.log_txt.delete(idx, f"{idx} lineend + 1c")
            timestamp = time.strftime('%H:%M:%S')
            self.log_txt.insert("end", f"[{timestamp}] {text}\n")
            self.log_txt.configure(state="disabled")
            self.log_txt.see("end")
        except Exception:
            pass
    def _perform_log_update(self, text):
        try:
            self.log_txt.configure(state="normal")
            self.log_txt.insert("end", text)
            self.log_txt.configure(state="disabled")
            self.log_txt.see("end")
        except:
            pass
    def _check_log_queue(self):
        """Revisa la cola de logs distribuida entre hilos."""
        try:
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
                self._perform_log_update(msg)
        except:
            pass
        self.after(500, self._check_log_queue)
if __name__ == "__main__":
    import sys
    if "--headless" in sys.argv:
        BOT_MODE = 3
        print("🚀 INICIANDO EN MODO TURBO (HEADLESS)...")
    app = TitanyBotApp()
    if BOT_MODE == 3:
        app.withdraw()                  
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
            print("\n[Base44] Cerrando conexión y actualizando estado a OFFLINE...")
            try:
                if hasattr(app, 'sync_engine'):
                    app.sync_engine.update_mobile_app(0, 0, 0, 0, "OFFLINE", 0, 0.0, 0.0, 0.0)
            except Exception as e:
                print(f"Error al enviar estado OFFLINE: {e}")
            app.running = False
            print("✅ Sistema detenido correctamente.")
