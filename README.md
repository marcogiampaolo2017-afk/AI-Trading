# 🤖 Titany AI Trading - MONEY BANG BANG (Versión Turbo)

Este repositorio contiene el código fuente para entrenar y ejecutar a **Titany AI**, un bot de trading basado en Reinforcement Learning (Recurrent PPO) especializado en EURUSD.

## ⚡ Novedades de esta Versión (V11)
El código ha sido fuertemente modificado para resolver el "Síndrome de Pánico" que sufría la IA.
1. **Paciencia Forzada:** Recompensas geométricas por "surfear tendencias" y castigos masivos (-15 puntos) por cerrar posiciones prematuramente con micro-ganancias.
2. **Entrenamiento Multinúcleo:** Soporte nativo para utilizar el 100% de la CPU (`SubprocVecEnv`). Detecta automáticamente todos los núcleos disponibles.

---

## 🛠️ Instrucciones de Entrenamiento (Para PC de Alto Rendimiento)

Si has descargado este repositorio para ejecutar el entrenamiento en una computadora potente, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone https://github.com/marcogiampaolo2017-afk/Titany-AI-Trading-MONEY-BANG_BANG.git
cd Titany-AI-Trading-MONEY-BANG_BANG
```

### 2. Instalar los requisitos
(Es recomendable crear primero un entorno virtual con `python -m venv .venv` y activarlo).
```bash
pip install -r requirements.txt
```

### 3. Iniciar el Entrenamiento Turbo
Arranca el script de entrenamiento. El sistema debería detectar automáticamente los núcleos de tu CPU e iniciar múltiples entornos paralelos para acelerar el aprendizaje masivamente.

```bash
python train_agent.py
```

### 4. Extraer el Modelo
Una vez que termine el entrenamiento (las 3 fases completas por las que pasará), el script generará automáticamente un archivo llamado `best_model.zip` (o similar) dentro de la carpeta `/best_models/`. 

Devuélvele **ese archivo .zip** al dueño del repositorio para que pueda conectarlo al bot en vivo usando `TITANY_AI_Terminal_Pro.py`.

---
*Powered by Stable-Baselines3, PyTorch & MetaTrader5.*
