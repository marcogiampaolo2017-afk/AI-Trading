#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_env.py — verifica que el entorno de TITANY AI esté completo.

Uso:
    python check_env.py
    python check_env.py --verbose

Comprueba versión de Python, que las dependencias críticas importen y que el
proyecto pueda construir sus features. Sale con código 1 si algo falta, para
poder usarlo en un script o en CI.

Contexto (2026-10) — los dos errores clásicos:
  * "No matching distribution found for pywin32==310" / numpy Requires-Python
    -> el venv se creó con Python 3.14/3.15, que no tienen wheels.
    Solución: py -3.13 -m venv .venv
  * "import could not be resolved" en VS Code -> el intérprete seleccionado
    no es .venv\\Scripts\\python.exe (ver .vscode/settings.json).
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys

MIN_PY = (3, 11)
MAX_PY = (3, 13)

MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "matplotlib.backends.backend_tkagg": "matplotlib (GUI Tk)",
    "matplotlib.figure": "matplotlib (Figure)",
    "PIL": "Pillow",
    "requests": "requests",
    "yaml": "PyYAML",
    "psutil": "psutil",
    "websockets": "websockets",
    "torch": "torch",
    "stable_baselines3": "stable-baselines3",
    "sb3_contrib": "sb3-contrib",
    "gymnasium": "gymnasium",
    "tensorboard": "tensorboard",
    "MetaTrader5": "MetaTrader5",
    "customtkinter": "customtkinter",
}


def check_python(verbose: bool) -> list[str]:
    errors: list[str] = []
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    supported = MIN_PY <= (v.major, v.minor) <= MAX_PY
    mark = "OK  " if supported else "FAIL"
    print(f"[{mark}] Python {version} ({platform.machine()})")
    if not supported:
        errors.append(
            f"Python {version} no soportado: usá 3.11-3.13 "
            "(numpy/torch/MetaTrader5 todavía no publican wheels para 3.14+)"
        )
    if verbose:
        print(f"       ejecutable: {sys.executable}")
        print(f"       venv activo: {sys.prefix != sys.base_prefix}")
    return errors


def check_modules(verbose: bool) -> list[str]:
    errors: list[str] = []
    for module, package in MODULES.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "(sin __version__)")
            print(f"[OK  ] {package:<32} {module:<38} {version}")
        except Exception as exc:  # noqa: BLE001 - queremos ver el motivo real
            errors.append(f"{package} ({module}): {type(exc).__name__}: {exc}")
            print(f"[FAIL] {package:<32} {module:<38} {type(exc).__name__}: {exc}")
    return errors


def _import_pandas_ta(verbose: bool) -> list[str]:
    """Acepta pandas-ta (original) o pandas-ta-classic (fork numpy 2)."""
    errors: list[str] = []
    for module in ("pandas_ta", "pandas_ta_classic"):
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "(sin __version__)")
            print(f"[OK  ] {'pandas-ta (cualquiera)':<32} {module:<38} {version}")
            return errors
        except ImportError:
            continue
    errors.append("pandas_ta / pandas_ta_classic: no está instalado (pip install pandas-ta-classic)")
    print(f"[FAIL] {'pandas-ta (cualquiera)':<32} {'pandas_ta':<38} ModuleNotFoundError")
    return errors


def check_project(verbose: bool) -> list[str]:
    errors: list[str] = []
    try:
        import pandas as pd

        from env import indicators

        if not indicators._HAS_TA:
            errors.append(
                "indicators._HAS_TA = False: add_indicators() fallará. "
                "Instalá pandas-ta-classic (Requirements.txt)."
            )
            print("[FAIL] indicators._HAS_TA")
        else:
            print("[OK  ] indicators._HAS_TA = True")

        # 400 velas: suficiente para SMA(200) y los indicadores más largos.
        n = 400
        df = pd.DataFrame(
            {
                "Open": [1.10 + i * 0.0001 for i in range(n)],
                "High": [1.101 + i * 0.0001 for i in range(n)],
                "Low": [1.099 + i * 0.0001 for i in range(n)],
                "Close": [1.1005 + i * 0.0001 for i in range(n)],
                "Volume": [100 + i for i in range(n)],
            }
        )
        out, cols = indicators.add_indicators(df)
        if not cols:
            errors.append("add_indicators() no devolvió columnas")
        else:
            print(f"[OK  ] add_indicators() -> {len(cols)} columnas de features")
        if verbose:
            print(f"       features: {', '.join(cols)}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pipeline de features: {type(exc).__name__}: {exc}")
        print(f"[FAIL] pipeline de features: {type(exc).__name__}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica el entorno de TITANY AI")
    parser.add_argument("--verbose", "-v", action="store_true", help="muestra rutas y versiones largas")
    args = parser.parse_args()

    reconfigure = getattr(sys.stdout, "reconfigure", None)  # consola Windows: UTF-8
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass

    print("=" * 78)
    print("TITANY AI - chequeo de entorno")
    print("=" * 78)
    errors: list[str] = []
    errors += check_python(args.verbose)
    print("-" * 78)
    errors += check_modules(args.verbose)
    errors += _import_pandas_ta(args.verbose)
    print("-" * 78)
    errors += check_project(args.verbose)
    print("=" * 78)

    if errors:
        print(f"RESULTADO: {len(errors)} problema(s)")
        for e in errors:
            print(f"  - {e}")
        print("")
        print("Guia rapida:")
        print(r"  py -3.13 -m venv .venv")
        print(r"  .\.venv\Scripts\Activate.ps1")
        print(r"  python -m pip install --upgrade pip")
        print(r"  pip install -r Requirements.txt")
        return 1
    print("RESULTADO: entorno OK - todos los imports resuelven.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
