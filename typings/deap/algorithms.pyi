# Stub mínimo de deap.algorithms: la API principal se registra en runtime
# (creator.create, Toolbox.register, etc.), por eso se resuelve como Any.
from typing import Any

def __getattr__(name: str) -> Any: ...

