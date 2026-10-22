# Stubs mínimos de MetaTrader5 (paquete compilado, sin tipos propios).
# Objetivo: que Pylance/Pyright conozcan las funciones y constantes que usa el
# proyecto y no reporten "initialize/login/account_info no son atributos conocidos".
# Las estructuras de retorno se tipan como Any a propósito: son named-tuples
# generados en C y sus campos varían entre versiones del terminal.
from typing import Any

# ── Constantes de temporalidad ────────────────────────────────────────────────
TIMEFRAME_M1: int
TIMEFRAME_M5: int
TIMEFRAME_M15: int
TIMEFRAME_M30: int
TIMEFRAME_H1: int
TIMEFRAME_H4: int
TIMEFRAME_D1: int

# ── Constantes de órdenes / ejecución ────────────────────────────────────────
ORDER_TYPE_BUY: int
ORDER_TYPE_SELL: int
ORDER_FILLING_FOK: int
ORDER_FILLING_IOC: int
ORDER_FILLING_RETURN: int
ORDER_TIME_GTC: int
TRADE_ACTION_DEAL: int
TRADE_ACTION_SLTP: int
TRADE_ACTION_PENDING: int
TRADE_RETCODE_DONE: int
TRADE_RETCODE_PLACED: int
DEAL_ENTRY_IN: int
DEAL_ENTRY_OUT: int
RES_ERROR_PARAMS: int
RES_S_OK: int

# ── Funciones usadas por el proyecto ─────────────────────────────────────────
def initialize(
    path: str | None = ...,
    *,
    login: int | None = ...,
    password: str | None = ...,
    server: str | None = ...,
    timeout: int | None = ...,
    portable: bool = ...,
) -> bool: ...
def login(
    login: int,
    *,
    password: str | None = ...,
    server: str | None = ...,
    timeout: int | None = ...,
) -> bool: ...
def shutdown() -> None: ...
def last_error() -> tuple[int, str]: ...
def account_info() -> Any: ...
def terminal_info() -> Any: ...
def symbol_info(symbol: str) -> Any: ...
def symbol_info_tick(symbol: str) -> Any: ...
def symbol_select(symbol: str, enable: bool = ...) -> bool: ...
def copy_rates_from_pos(symbol: str, timeframe: int, start_pos: int, count: int) -> Any: ...
def copy_rates_range(symbol: str, timeframe: int, date_from: Any, date_to: Any) -> Any: ...
def copy_ticks_from(symbol: str, date_from: Any, count: int, flags: int) -> Any: ...
def positions_get(*, symbol: str | None = ..., ticket: int | None = ...) -> Any: ...
def orders_get(*, symbol: str | None = ..., ticket: int | None = ...) -> Any: ...
def history_deals_get(date_from: Any, date_to: Any, *, group: str | None = ...) -> Any: ...
def history_orders_get(date_from: Any, date_to: Any, *, group: str | None = ...) -> Any: ...
def order_send(request: dict[str, Any]) -> Any: ...
def order_check(request: dict[str, Any]) -> Any: ...
def configure(*args: Any, **kwargs: Any) -> bool: ...
def pack(*args: Any, **kwargs: Any) -> Any: ...

# Cualquier constante/función no listada arriba (versiones nuevas) no debe romper el tipado.
def __getattr__(name: str) -> Any: ...
