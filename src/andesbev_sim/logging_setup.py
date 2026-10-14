"""Logging centralizado: consola + archivo .log legible + archivo .json estructurado.

Cada corrida recibe un ``run_id`` (fecha-hora + nombre) y escribe sus registros en
``logs/<tipo>/<run_id>.log`` y ``logs/<tipo>/<run_id>.json``, donde ``tipo`` separa
las corridas de preparación de datos, las del escenario base y las de disrupción.

Para que cada registro lleve contexto (escenario, réplica, canal, sector) se usa
``logging.LoggerAdapter``: ``log = contexto(log, replica=3)`` agrega ese campo a
todo lo que se registre después con ``log``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

NOMBRE_RAIZ = "andesbev"

# Handlers adicionales que sobreviven a cada configurar_logging (p. ej. la GUI web,
# que transmite los registros en vivo por WebSocket)
HANDLERS_EXTRA: list[logging.Handler] = []

# Campos estándar de LogRecord que NO son contexto extra
_CAMPOS_ESTANDAR = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class FormatoJSON(logging.Formatter):
    """Una línea JSON por registro (formato JSON Lines), fácil de leer con pandas."""

    def format(self, record: logging.LogRecord) -> str:
        datos = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "nivel": record.levelname,
            "modulo": record.name,
            "mensaje": record.getMessage(),
        }
        extra = {k: v for k, v in vars(record).items() if k not in _CAMPOS_ESTANDAR}
        datos.update(extra)
        if record.exc_info:
            datos["excepcion"] = self.formatException(record.exc_info)
        return json.dumps(datos, ensure_ascii=False, default=str)


class _AdaptadorContexto(logging.LoggerAdapter):
    """Mezcla el contexto fijo del adaptador con el ``extra`` de cada llamada."""

    def process(self, msg, kwargs):
        kwargs["extra"] = {**self.extra, **kwargs.get("extra", {})}
        return msg, kwargs


def nuevo_run_id(nombre: str) -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{nombre}"


def configurar_logging(
    carpeta: Path,
    tipo: str,
    run_id: str,
    nivel_consola: str = "INFO",
    nivel_archivo: str = "INFO",
) -> tuple[logging.LoggerAdapter, dict[str, Path]]:
    """Prepara los tres destinos de log y devuelve el logger con el run_id fijado."""
    destino = Path(carpeta) / tipo
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {"log": destino / f"{run_id}.log", "json": destino / f"{run_id}.json"}

    raiz = logging.getLogger(NOMBRE_RAIZ)
    raiz.setLevel(logging.DEBUG)
    for h in list(raiz.handlers):  # evita duplicar handlers si se llama dos veces
        raiz.removeHandler(h)
        if h not in HANDLERS_EXTRA:
            h.close()
    raiz.propagate = False

    consola = logging.StreamHandler(sys.stdout)
    consola.setLevel(nivel_consola)
    consola.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )

    archivo = logging.FileHandler(rutas["log"], encoding="utf-8")
    archivo.setLevel(nivel_archivo)
    archivo.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )

    js = logging.FileHandler(rutas["json"], encoding="utf-8")
    js.setLevel(nivel_archivo)
    js.setFormatter(FormatoJSON())

    for h in (consola, archivo, js, *HANDLERS_EXTRA):
        raiz.addHandler(h)

    return _AdaptadorContexto(raiz, {"run_id": run_id, "tipo_corrida": tipo}), rutas


def contexto(log: logging.LoggerAdapter, **campos) -> logging.LoggerAdapter:
    """Devuelve un logger que agrega ``campos`` a cada registro."""
    return _AdaptadorContexto(log.logger, {**log.extra, **campos})


def cerrar_logging() -> None:
    raiz = logging.getLogger(NOMBRE_RAIZ)
    for h in list(raiz.handlers):
        raiz.removeHandler(h)
        if h not in HANDLERS_EXTRA:
            h.close()
