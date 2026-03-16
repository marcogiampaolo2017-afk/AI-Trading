import os
import sys
import dill
import MetaTrader5 as mt5

# Añadir la carpeta de genética al system path para que 'dill' pueda
# encontrar las clases originales (gp_strategy_progress, deap, etc.)
current_dir = os.path.dirname(os.path.abspath(__file__))
genetic_dir = os.path.join(os.path.dirname(current_dir), 'models', 'genetics')
sys.path.append(genetic_dir)

class GeneticPredictor:
    """
    Carga el árbol genético (.dill) de DEAP y lo evalúa contra 
    las 16 entradas de mercado (M5) en tiempo real para AGA-MORA.
    """
    def __init__(self):
        self.model = None
        self.compiled_func = None
        self.load_model()
        
    def load_model(self):
        try:
            import gp_strategy_progress
        except Exception as e:
            print(f"❌ [GENETIC ENGINE] Fallo al cargar gp_strategy_progress: {e}")
            return

        model_path = os.path.join(genetic_dir, 'best_individual.dill')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = dill.load(f)
                
                # Compilamos la fórmula abstracta en código ejecutable de Python
                self.compiled_func = gp_strategy_progress.toolbox.compile(expr=self.model)
                print(f"🧬 [GENETIC ENGINE] Cerebro evolutivo cargado: {len(self.model)} nodos matemáticos activos.")
            except Exception as e:
                print(f"❌ [GENETIC ENGINE] Error al cargar la fórmula genética: {e}")
        else:
            print("⚠️ [GENETIC ENGINE] No se encontró 'best_individual.dill' en /models/genetics/")
            
    def get_signal(self):
        """
        Extrae datos de 4 pares en M5 y los pasa por la fórmula genética.
        Devuelve el 'Target Exposure' (+ = Fuerte Compra, - = Fuerte Venta).
        """
        if not self.compiled_func:
            return 0.0
            
        inputs = []
        pairs = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
        
        for p in pairs:
            # Pedimos la última vela completa M5 al servidor MT5
            try:
                rates = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M5, 0, 1)
                if rates is not None and len(rates) > 0:
                    inputs.extend([
                        float(rates[0]['open']), 
                        float(rates[0]['high']), 
                        float(rates[0]['low']), 
                        float(rates[0]['close'])
                    ])
                else:
                    # Relleno de seguridad para evitar división por cero
                    inputs.extend([1.0, 1.0, 1.0, 1.0]) 
            except Exception:
                inputs.extend([1.0, 1.0, 1.0, 1.0])
                
        try:
            # Ejecutamos la super-fórmula sobre estos 16 inputs
            signal = float(self.compiled_func(*inputs))
            return signal
        except Exception as e:
            # print(f"Error de inferencia genética: {e}")
            return 0.0
