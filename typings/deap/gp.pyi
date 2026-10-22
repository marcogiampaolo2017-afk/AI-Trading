# Stub de deap.gp: casi toda la API se resuelve en runtime.
from typing import Any

genHalfAndHalf: Any
genFull: Any
genRamped: Any
compile: Any
cxOnePoint: Any
cxOnePointLeafBiased: Any
mutUniform: Any
mutNodeReplacement: Any
staticLimit: Any
PrimitiveTree: Any
PrimitiveSetTyped: Any

class PrimitiveSet:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def addPrimitive(self, *args: Any, **kwargs: Any) -> None: ...
    def addTerminal(self, *args: Any, **kwargs: Any) -> None: ...
    def addEphemeralConstant(self, *args: Any, **kwargs: Any) -> None: ...
    def renameArguments(self, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
