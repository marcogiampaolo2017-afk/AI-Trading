# Problemas detectados en los archivos Python

Fecha de revisión: 2026-10-21

Este documento recoge los puntos críticos que aparecen al revisar los archivos .py del proyecto con el diagnostico del editor y la compilación del código. No incluye correcciones; solo documenta los problemas reales observados.

## Resumen ejecutivo

Los problemas principales no son solo de sintaxis, sino de arquitectura y entorno:

- dependencia externa de MetaTrader5 no disponible o no reconocida por el editor
- mezcla de librerías de trading y de entrenamiento que no comparten la misma acción/espacio de observación
- uso de `None` en valores que luego se usan como floats
- manejo inseguro de `pandas_ta` / `pandas_ta_classic`
- un posible error en `os.cpu_count()` cuando devuelve `None`
- incompatibilidades de tipos en `gym` / `gymnasium` y en matplotlib

---

## 1) Dependencias del entorno y librerías externas

### 1.1 MetaTrader5 no está resuelto en el entorno analizado

Archivos afectados:
- [core/live_trading_system.py](core/live_trading_system.py)
- [core/TITANY_AI_Terminal_Pro.py](core/TITANY_AI_Terminal_Pro.py)

Se reportan errores del tipo:
- `"initialize" no es un atributo conocido del módulo "MetaTrader5"`
- `"login" no es un atributo conocido del módulo "MetaTrader5"`
- `"account_info" no es un atributo conocido del módulo "MetaTrader5"`
- `"positions_get" no es un atributo conocido del módulo "MetaTrader5"`

Esto suele significar una de estas cosas:

1. la librería no está instalada en el entorno actual
2. el entorno del editor no coincide con el entorno de ejecución
3. el proyecto se está analizando sin MetaTrader5 disponible
4. existen diferencias entre entorno local y entorno de Windows/MT5

### 1.2 pandas_ta / pandas_ta_classic puede quedar como `None`

Archivo afectado:
- [env/indicators.py](env/indicators.py)

La lógica hace esto:

- intenta importar `pandas_ta`
- si falla, intenta `pandas_ta_classic`
- si también falla, pone `ta = None`

Luego se usa `ta.rsi(...)`, `ta.macd(...)`, `ta.atr(...)`, etc. El analizador marca errores como:
- `"rsi" no es un atributo conocido de "None"`
- `"macd" no es un atributo conocido de "None"`

Esto implica que si la librería no existe en el entorno, el proyecto puede romperse en tiempo de ejecución aunque la sintaxis sea válida.

---

## 2) Problemas de arquitectura y inconsistencias del modelo

### 2.1 El espacio de acciones de entrenamiento y live no coincide

Archivos relevantes:
- [config.py](config.py)
- [core/live_trading_system.py](core/live_trading_system.py)
- [env/trading_env.py](env/trading_env.py)

El proyecto tiene una configuración importante en [config.py](config.py):
- `SL_TP_PAIRS` define un conjunto específico de pares `(sl, tp)`
- `ALLOW_MANUAL_CLOSE = False`

Pero el código de live y entorno genera acciones de forma distinta en varios sitios, y el archivo comenta explícitamente que la mismatched action space puede producir predicciones incorrectas. Es un problema clave del sistema porque la IA se entrena con un espacio de acciones y luego intenta operar con otro.

Esto puede provocar:

- modelos entrenados con otra dimensionalidad que no coinciden con la inferencia
- decisiones inválidas o sin sentido
- errores de predicción por desalineación entre train/live

### 2.2 `RecurrentPPO.load(...)` y la normalización se usan con env/arquitecturas distintas

Archivo afectado:
- [core/live_trading_system.py](core/live_trading_system.py)

Se detectan mensajes de dependencia de `VecNormalize` y `observation_space` con `None` y valores que no encajan. El analizador reporta:
- `"normalize_obs" no es un atributo conocido de "None"`
- `"predict" no es un atributo conocido de "None"`

Esto indica que el flujo de carga del modelo/normalizador puede dejar `self.v_env` o `self.model` en `None` antes de usarlos.

---

## 3) Problemas de `None` y tipos en la simulación de trading

### 3.1 `entry_price`, `sl_price`, `tp_price` y `exit_price` se usan como float aunque pueden ser `None`

Archivo afectado:
- [env/trading_env.py](env/trading_env.py)

El editor reporta errores como:
- `El operador "-" no se admite para los tipos "float" y "float | None"`
- `Argumento de tipo "float | None" no puede ser asignado a parámetro "exit_price" de tipo "float"`
- `"None" no se puede asignar a "float"`

Esto ocurre en lógica de cierre de posiciones, SL/TP y cálculo de P&L. La causa es que la clase inicializa varios valores a `None` y luego los usa como si siempre fueran numéricos.

### 3.2 `self.action_map` se inicia con tuplas que contienen `None` y luego se rellena con tuplas de float/ints

Archivo afectado:
- [env/trading_env.py](env/trading_env.py)

El editor marca mensajes del tipo:
- `Argumento de tipo "tuple[Literal['OPEN'], int, float, float]" no puede ser asignado a parámetro "object" de tipo "tuple[str, None, None, None]"`

Es decir, la lista se declara con un tipo base `None`, pero luego se agregan tuplas con valores reales; esto puede generar problemas de inferencia y errores de tipo en análisis estático.

---

## 4) Problemas de `gym` / `gymnasium`

Archivos afectados:
- [env/trading_env.py](env/trading_env.py)
- [core/live_trading_system.py](core/live_trading_system.py)

Se reporta:
- `El argumento de la clase debe ser una clase base`
- `"normalize_obs" no es un atributo conocido de "DummyVecEnv"`

Esto suele venir de una mezcla de importaciones y versiones:

- `gymnasium` importado en una rama
- `gym` importado como fallback en otra
- `DummyVecEnv` / `VecNormalize` usados con una API distinta según la versión

Es una señal clara de que el entorno de ejecución y la versión de las bibliotecas no están alineadas.

---

## 5) Problemas de compatibilidad y tipos en otras partes del repositorio

### 5.1 `os.cpu_count()` puede devolver `None`

Archivo afectado:
- [research/train_agent.py](research/train_agent.py)

Se detecta:
- `El operador "-" no es compatible con "None".`

La línea relevante es similar a:

```python
N_CORES = min(6, max(1, os.cpu_count() - 1))
```

Si `os.cpu_count()` devuelve `None`, la operación falla.

### 5.2 `plt.tight_layout(rect=[...])` recibe una lista, pero la función espera una tupla

Archivo afectado:
- [research/test_agent.py](research/test_agent.py)

Se reporta:
- `Argumento de tipo "list[int | float]" no puede ser asignado a parámetro "rect" de tipo "tuple[float, float, float, float] | None"`

Esto es un problema de tipado/compatibilidad, no necesariamente un fallo de ejecución, pero sí indica una zona frágil del código.

---

## 6) Problemas de consola / streams y entorno Windows

Archivo afectado:
- [core/live_trading_system.py](core/live_trading_system.py)

Se observa:
- `TextIO` no tiene el atributo `reconfigure`

Esto es típico en entornos donde la consola no soporta ese método, o en Python versiones/implementaciones donde `reconfigure` no está disponible. No es un bug fatal, pero sí un punto de compatibilidad que puede romper la salida de logs en algunos equipos.

---

## 7) Problemas de flujo lógico y runtime esperables

### 7.1 Lógica de cierre y desenlaces de la IA puede ser inconsistente

Archivo afectado:
- [env/trading_env.py](env/trading_env.py)

Hay varios puntos donde el flujo depende de condiciones que pueden no cumplirse:
- cierre de posición sin precio válido
- comprobación de SL/TP con precios `None`
- `return None` donde se espera `float`

Esto puede producir que la IA genere señales válidas pero sin salida real, o que la simulación falle al cerrar posiciones.

### 7.2 El sistema de live trading y el entrenamiento no comparten la misma semántica de la acción

Archivos relevantes:
- [config.py](config.py)
- [core/live_trading_system.py](core/live_trading_system.py)
- [env/trading_env.py](env/trading_env.py)
- [core/TITANY_AI_Terminal_Pro.py](core/TITANY_AI_Terminal_Pro.py)

El proyecto usa comentarios como:
- `Action space MUST match training`
- `old Cartesian... caused wrong trades`

Eso es una clara señal de que la lógica se ha reescrito varias veces y que el estado de producción puede no coincidir con el modelo entrenado.

---

## 8) Archivos del árbol del proyecto revisados

Además de los archivos más problemáticos, estos son los que aparecen en el árbol del proyecto y su estado observado:

- [core/genetic_engine.py](core/genetic_engine.py): sin errores reportados por el analizador; revisarlo sigue siendo importante porque interactúa con la lógica del entorno y puede depender del mismo estado de datos.
- [core/live_trading_system.py](core/live_trading_system.py): crítico. Tiene errores de MetaTrader5, `None` en observación/normalización, incompatibilidad de `gym`/`gymnasium`, y flujo potencialmente roto.
- [core/titany_continuous_trainer.py](core/titany_continuous_trainer.py): sin errores reportados, pero forma parte del flujo de entrenamiento y debe revisarse junto con la señalización del modelo.
- [core/titany_multiverse.py](core/titany_multiverse.py): sin errores reportados, aunque influye en la lógica de pantalla y en el pipeline de señales.
- [env/indicators.py](env/indicators.py): crítico. Usa `pandas_ta`/`pandas_ta_classic` sin validación fuerte y reporta acceso a `None`.
- [env/trading_env.py](env/trading_env.py): crítico. Tiene `None` en precios, inconsistencias de tipos, y mezcla de `gym`/`gymnasium`.
- [research/test_agent.py](research/test_agent.py): tiene una advertencia de compatibilidad con `matplotlib` (`rect` como lista en lugar de tupla).
- [research/train_agent.py](research/train_agent.py): tiene un problema en `os.cpu_count()` y un flujo de entrenamiento que debe manejar `None`.
- [research/verify_phase6.py](research/verify_phase6.py): sin errores reportados.
- [tests/test_integration.py](tests/test_integration.py): sin errores reportados.
- [tests/test_trading_env.py](tests/test_trading_env.py): sin errores reportados.

En otras palabras, el proyecto no solo tiene problemas en los archivos “más visibles”; también hay varios archivos que parecen estar bien a nivel de IDE pero forman parte de la cadena crítica de entrenamiento y runtime.

## 9) Conclusión

El proyecto tiene varios problemas de nivel crítico:

1. dependencia externa no disponible (`MetaTrader5` y `pandas_ta`)
2. inconsistencias entre train/live y action-space
3. `None` usado como valor numérico en puntos clave
4. incompatibilidades de librerías (`gym` vs `gymnasium`)
5. errores de tipado y flujo de ejecución en entorno de validación

Estos problemas hacen que el proyecto esté en un estado frágil: puede compilar parcialmente, pero no es fiable ni para entrenamiento ni para ejecución real sin una revisión profunda.

---

## 9) Archivos más relevantes para revisar primero

1. [env/trading_env.py](env/trading_env.py)
2. [env/indicators.py](env/indicators.py)
3. [core/live_trading_system.py](core/live_trading_system.py)
4. [config.py](config.py)
5. [research/train_agent.py](research/train_agent.py)
6. [research/test_agent.py](research/test_agent.py)

---

## 10) Estado de resolución (2026-10-21, sesión de corrección)

> **Resultado:** los 8 grupos del diagnóstico quedaron resueltos. Verificado con
> Pyright, `pytest`, `check_env.py`, compilación completa y ejecución real contra
> MetaTrader 5. El diagnóstico original se conserva intacto arriba.

### 10.1 Resumen medible

| Verificación | Antes | Después |
|---|---|---|
| `pip install -r Requirements.txt` | abortaba (`pywin32==310`, numpy en Python 3.15) | instala y verifica sin errores (Python 3.13) |
| `python check_env.py` | no existía | 17/17 módulos + `add_indicators()` → 12 features |
| **Pyright, proyecto completo** | **74 errores** | **0 errores, 0 warnings** |
| Pyright, los 6 archivos del documento | 11 errores | **0** |
| `pytest tests/ -q` | 28 passed / 38 skipped | **65 passed / 1 skipped** |
| Action space live vs entrenamiento | 4 acciones (CLOSE fija) vs 3 | **3 = 3**, derivado de `config` |
| Shadow Trainer | `Discrete(20)` legacy | `Discrete(3)` derivado de `config` |
| `os.cpu_count()` / `TextIO.reconfigure` / `rect` | errores de tipos | 0 |

### 10.2 Punto por punto

| # | Problema | Causa real | Solución aplicada |
|---|---|---|---|
| 1.1 | `MetaTrader5` sin atributos | paquete compilado sin tipos + intérprete equivocado (3.15) | venv 3.13 + `typings/MetaTrader5.pyi` + `python.analysis.stubPath` |
| 1.2 | `pandas_ta` como `None` | import opcional con fallback | `ta: Any` + fallback `pandas_ta_classic` (Requirements) |
| 2.1 | action space distinto | live añadía CLOSE; shadow usaba `Discrete(20)` | action map derivado de `config` en live, shadow y terminal |
| 2.2 | `RecurrentPPO`/`VecNormalize` `None` | atributos sin anotar, sin guardas | anotaciones + guardas + `isinstance(VecNormalize)` |
| 3.1 | precios `None` usados como float | atributos inicializados a `None` | `float \| None` + guardas en cierre, SL/TP y step |
| 3.2 | `action_map` tipado como `None` | lista inferida como `(str, None, None, None)` | tipo explícito de 4-tupla con miembros opcionales |
| 4 | `gym` vs `gymnasium` | fallback a `gym` (no instalado) | solo `gymnasium`, API 5-tuple única, base tipada |
| 5.1 | `os.cpu_count()` puede ser `None` | sin fallback | `(os.cpu_count() or 1) - 1` |
| 5.2 | `tight_layout(rect=[...])` | el tipo espera tupla | `rect=(0, 0.04, 1, 1)` |
| 6 | `TextIO.reconfigure` | atributo asumido | `getattr(stream, "reconfigure", None)` + `callable()` |
| 7.1 | cierre sin precio válido | flujo sin guardas | cierre no-op sin posición; retorno `float \| None` |
| 7.2 | semántica train/live | reescrituras históricas | mapa de acciones único en los 3 puntos de uso |

### 10.3 Archivos tocados en esta corrección

- `env/trading_env.py`: tipos de estado/precios, action map, guardas, gymnasium único.
- `env/indicators.py`: import dinámico tipado de `pandas_ta` / `pandas_ta_classic`.
- `core/live_trading_system.py`: action map alineado con entrenamiento, guardas de
  modelo/normalizador, consola, `positions_get` puede ser `None`.
- `core/TITANY_AI_Terminal_Pro.py`: action map tipado, respuesta del kill switch,
  `observation_space.shape` opcional, consola, atributos de widgets.
- `core/titany_continuous_trainer.py`: `ShadowMarketEnv` con acción derivada de config.
- `core/genetic_engine.py`, `core/titany_multiverse.py`: tipos de deap y pandas.
- `research/train_agent.py`, `research/test_agent.py`, `research/validate_v10_performance.py`,
  `research/verify_phase6.py`: `cpu_count`, `rect`, `where(other=)`, API 5-tuple.
- `models/genetics/gp_strategy_progress.py`: tipos de deap/pandas.
- `tests/`: helpers tipados y API gymnasium (ya no se contemplan 4-tuplas).
- Nuevos: `typings/MetaTrader5.pyi`, `typings/deap/*`, `pyrightconfig.json`.

### 10.4 Verificación reproducible

```powershell
.\.venv\Scripts\Activate.ps1
python check_env.py
npx --yes pyright --project pyrightconfig.json   # 0 errors
python -m pytest tests/ -q                       # 65 passed, 1 skipped
```

### 10.5 Notas

- El test `test_close_resets_position` se saltea a propósito cuando
  `ALLOW_MANUAL_CLOSE = False` (no existe la acción CLOSE). Antes fallaba por asumirla.
- La consola Windows se fuerza a UTF-8 en los entry points: sin eso, los emojis de
  los logs lanzan `UnicodeEncodeError` con cp1252.
- `research/validate_v10_performance.py` requiere un `models/*.zip` que el repo no
  versiona (ver `.gitignore`); sin modelo no es ejecutable, pero su código ya compila
  y tipa limpio.

