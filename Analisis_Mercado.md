# Titany AI: Explicación del Análisis de Mercado para Traders

Este documento explica de forma clara y directa cómo "piensa" y escanea el mercado el bot **Titany AI**. Se ha omitido el lenguaje complejo de programación para enfocarnos exclusivamente en la lógica de trading, los indicadores que analiza y cómo toma sus decisiones de entrada, mantenimiento y salida.

---

## 1. El Cerebro General: ¿Cómo ve el Mercado Titany AI?

A diferencia de los Asesores Expertos (EAs) clásicos basados en cruces estáticos de medias o condiciones "If-Then", Titany utiliza **Aprendizaje por Refuerzo (Reinforcement Learning - PPO Recurrente)**. Esto significa que la IA no sigue una receta ciega; *observa el contexto, simula el riesgo, ejecuta una acción y aprende de la recompensa o el dolor de la pérdida*.

**¿Qué "ve" en su pantalla mental para operar?**
* **El Pasado Reciente:** Estudia bloques de las últimas **30 velas** continuas (Price Action e historial dinámico).
* **El Termómetro de la Cuenta:** Sabe en todo tiempo si está comprada, vendida o fuera del mercado. Mide el tiempo que lleva dentro del trade, el margen libre de la cuenta, y muy importante: *el PNL flotante (Ganancias/Pérdidas no realizadas en pips)*. Esto le permite gestionar sus emociones (matemáticas) frente al drawdown.

---

## 2. Los 5 Módulos de Análisis que componen a Titany

La IA procesa datos crudos del precio y los transforma utilizando indicadores divididos en "Módulos de Combate". 

### A. Módulo Base (Price Action y Tendencia Clásica)
Detecta los fundamentos estructurales del precio de la misma forma que un analista técnico:
* **RSI (14 y 50):** Escanea múltiples temporalidades del agotamiento del precio.
* **ADX (14):** Filtra para saber si el mercado está en rango o en tendencia pura.
* **Bandas de Bollinger y Distancia a las Medias (MA 20 y MA 50):** Evalúa la sobre-extensión del precio respecto a sus medias móviles para calcular si el "caucho" estadístico se estiró demasiado.
* **Divergencias Precio/Volumen (PV Divergence):** Detecta cuando el precio sube pero el volumen real cae, anticipando un engaño inminente.

### B. Módulo "Sniper" (Análisis de Volumen Institucional - VSA)
La IA entiende que sin las "instituciones" el precio no se sostiene, por ello escanea el rastro de volumen:
* **VSA Ratio (Volume Spread Analysis):** Analiza la relación entre el tamaño de la vela y el tick volume. Un ratio fuerte (ej. >1.1 o >1.3) le indica a la IA que el movimiento está respaldado por dinero institucional ("momentun de tiburones") y no es un movimiento falso ("fakeout").
* **Volume Delta & Climax Candles:** Busca velas climáticas que delatan agotamiento masivo o "paradas" institucionales para no quedar atrapada en la cima de una mecha.

### C. Módulo Cuantitativo (Estructura Avanzada)
Aquí la IA filtra el ruido del mercado minorista:
* **FER (Fractal Efficiency Ratio):** Mide qué tan limpia es una dirección. Si el mercado hace mucho zigzag, el FER es bajo. Si sube de golpe sin retrocesos, el FER es muy alto (>0.30 - 0.45). Con un FER alto, la IA sabe que es momento de **apretar el acelerador** y subirse a la tendencia.
* **Z-Score "Lazarus":** Este es el equivalente estadístico a extremos del mercado. Un valor elevado (>1.2) le advierte de condiciones severas de desviación (porcentualmente anómalas), usadas para escenarios de toma de beneficios extremos o rebotes "tipo francotirador".
* **Entropía y Cohesión (Phi Score):** Mide si los indicadores "están de acuerdo". Si la Entropía es muy alta o el índice Phi es bajo, la IA asume que el mercado está en un caos impredecible ("Rango chopy") y automáticamente entra en Modo Espera (Hold).

### D. Módulo Física (Modelos Físicos de Mercado)
Utiliza la matemática de fluidos para analizar la presión de las órdenes:
* **Phys_Pressure y Phys_Viscosity:** Calcula cuánta presión hacia abajo o hacia arriba sufre el precio y cuánta "viscosidad" o nivel de soporte/fácil de atravesar existe.
* **Modelos Estocásticos (Fisher y Hawkes):** Modelos que proyectan si las compras actuales van a generar un efecto cascada (FOMO del mercado) o si se apagaran de inmediato.

---

## 3. Comportamiento y Gestión de Riesgo (Risk Management)

La forma en que Titany gestiona el *Trade Management* es lo que la define como un ente "viviente" en el gráfico:

1. **Paciencia Forzada ("Cura contra la Ansiedad del Trader")**: 
   Anteriormente, la IA ("Sufriendo Síndrome de Pánico") cerraba posiciones en breakeven o con micro-ganancias para aliviar la tensión del riesgo. El algoritmo actual fue rediseñado para recibir un **mega castigo (-15 a -20 puntos de premio interno)** si decide cerrar un trade ganador mientras los medidores de fuerza institucional (VSA y FER) aún muestran potencial para ir por más pips. Se la obliga a "surfear la tendencia".

2. **Simulador Causal "What-If" - Protección contra Latigazos**:
   Antes de disparar, la IA analiza el **ATR (Average True Range)**. Si simula mentalmente que la volatilidad es tan grande que un pico o mechazo (spike) la barrería aleatoriamente, decide **vetar** la entrada.

3. **Stop Loss y Take Profits Dinámicos**:
   Mide la respiración en tiempo real. Sus niveles de riesgo se multiplican en relación al ATR dinámico; no usa zonas de salida fijas, si el mercado está eufórico, ensanchará o apretará su SL y TP.

4. **Entiende sus Propios Costos (Dolor del Spread)**:
   A diferencia del backtesting irreal, Titany se entrena viviendo en una constante simulación de *spread, comisiones y slippage dinámicos retardos*. Constantemente asume el "peor escenario" de entrada por ping de conexión y requiere que su probabilidad de ganancia supere los costos combinados para ejecutar la operación.

---
**Resumen del Trader:**
Titany es un gestor de momentum que evita entornos de ruido (consolidaciones falsas) escaneando el peso del volumen (VSA), aguardando explosiones limpias (FER), protegiéndose de la manipulación usando desviación estadística (Z-Score) y siendo castigada violentamente si cierra temprano grandes tendencias institucionales.
