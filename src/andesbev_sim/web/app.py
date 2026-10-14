"""Servidor web ligero (FastAPI): puente entre el panel XHTML y el simulador.

Rutas:
    GET  /                      panel de control (XHTML)
    GET  /api/escenarios        YAML disponibles en config/
    POST /api/ejecutar          lanza base vs disrupción en segundo plano
    GET  /api/estado/{job}      progreso, estado y enlace al dashboard (Fetch API)
    GET  /api/logs/{job}        registros desde la posición ``desde`` (Fetch API)
    WS   /ws/progreso/{job}     progreso y logs en vivo (WebSocket)
    POST /api/dashboard/{job}   regenera el dashboard de esa corrida
    GET  /dashboard/{run_id}    sirve el dashboard XHTML generado

La simulación corre en un hilo aparte para no bloquear el servidor. Se permite
una corrida a la vez, porque el logging del simulador es global.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from andesbev_sim import RAIZ_REPO, etapas
from andesbev_sim.config import cargar_config
from andesbev_sim.logging_setup import HANDLERS_EXTRA, NOMBRE_RAIZ

AQUI = Path(__file__).resolve().parent
CARPETA_CONFIG = RAIZ_REPO / "config"
XHTML = "application/xhtml+xml; charset=utf-8"
RE_RUN = re.compile(r"^[0-9]{8}_[0-9]{6}_comparacion$")

app = FastAPI(title="AndesBev · simulación downstream", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------- #
# Estado de las corridas
# --------------------------------------------------------------------------- #
@dataclass
class Trabajo:
    id: str
    estado: str = "en_cola"  # en_cola | corriendo | listo | error
    progreso: float = 0.0
    mensaje: str = ""
    logs: list[dict] = field(default_factory=list)
    run_id: str | None = None
    carpeta: Path | None = None
    resumen: dict | None = None
    error: str | None = None


TRABAJOS: dict[str, Trabajo] = {}
CANDADO = threading.Lock()


class _HandlerTrabajo(logging.Handler):
    """Copia cada registro del simulador a la lista del trabajo (para la GUI)."""

    def __init__(self, trabajo: Trabajo):
        super().__init__(level=logging.INFO)
        self.t = trabajo

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.INFO and getattr(record, "replica", None) is not None:
            return  # el detalle por réplica queda en el archivo, no en pantalla
        self.t.logs.append(
            {
                "hora": self.formatter.formatTime(record, "%H:%M:%S") if self.formatter else "",
                "nivel": record.levelname,
                "escenario": getattr(record, "escenario", getattr(record, "tipo_corrida", "")),
                "mensaje": record.getMessage(),
            }
        )


class Parametros(BaseModel):
    """Lo que envía el formulario. Los rangos protegen al servidor de valores absurdos."""

    escenario_base: str = "base.yaml"
    escenario_disrupcion: str = "disrupcion.yaml"
    replicas: int = Field(30, ge=1, le=200)
    semilla: int = Field(42, ge=0, le=2**31 - 1)
    camiones_dedicada: int | None = Field(None, ge=0, le=100)
    camiones_capilar_nucleo: int | None = Field(None, ge=0, le=100)
    camiones_capilar_periferia: int | None = Field(None, ge=0, le=100)
    factor_capacidad: float = Field(1.0, ge=0.5, le=1.5)
    capacidad_restante_bloqueo: float = Field(0.3, ge=0.0, le=1.0)
    multiplicador_peak: float = Field(1.6, ge=1.0, le=3.0)


def _yaml_seguro(nombre: str) -> Path:
    """Solo se aceptan archivos .yaml que existan dentro de config/ (sin rutas)."""
    ruta = (CARPETA_CONFIG / Path(nombre).name).resolve()
    if ruta.parent != CARPETA_CONFIG.resolve() or ruta.suffix != ".yaml" or not ruta.exists():
        raise HTTPException(400, f"Escenario no válido: {nombre}")
    return ruta


def _correr(t: Trabajo, p: Parametros) -> None:
    handler = _HandlerTrabajo(t)
    handler.setFormatter(logging.Formatter())
    HANDLERS_EXTRA.append(handler)
    try:
        t.estado = "corriendo"

        def progreso(x: float, msg: str) -> None:
            t.progreso, t.mensaje = round(x, 3), msg

        ajustes = {
            "camiones": {
                "DEDICADA": p.camiones_dedicada,
                "CAPILAR_NUCLEO": p.camiones_capilar_nucleo,
                "CAPILAR_PERIFERIA": p.camiones_capilar_periferia,
            },
            "factor_capacidad": p.factor_capacidad,
            "capacidad_restante_bloqueo": p.capacidad_restante_bloqueo,
            "multiplicador_peak": p.multiplicador_peak,
        }
        r = etapas.comparar(
            _yaml_seguro(p.escenario_base),
            _yaml_seguro(p.escenario_disrupcion),
            {"replicas": p.replicas, "semilla": p.semilla},
            progreso,
            ajustes,
        )
        from andesbev_sim.reportes.dashboard import generar

        generar(r["carpeta"])
        t.run_id, t.carpeta = r["run_id"], r["carpeta"]
        c = r["comparacion"]["comparacion"]
        t.resumen = {
            "servicio_base": c["servicio_a_tiempo_base"],
            "servicio_disrupcion": c["servicio_a_tiempo_disrupcion"],
            "perdida_pp": c["perdida_servicio_pp"],
            "venta_perdida_extra": c["venta_perdida_extra_pallets"],
        }
        t.progreso, t.mensaje, t.estado = 1.0, "listo", "listo"
    except Exception as e:  # el detalle completo queda en el log del escenario
        t.estado, t.error = "error", str(e)
        t.logs.append({"hora": "", "nivel": "ERROR", "escenario": "", "mensaje": str(e)})
    finally:
        HANDLERS_EXTRA.remove(handler)
        logging.getLogger(NOMBRE_RAIZ).removeHandler(handler)
        CANDADO.release()


def _trabajo(job: str) -> Trabajo:
    if job not in TRABAJOS:
        raise HTTPException(404, "Trabajo no encontrado")
    return TRABAJOS[job]


def _estado(t: Trabajo) -> dict:
    return {
        "id": t.id,
        "estado": t.estado,
        "progreso": t.progreso,
        "mensaje": t.mensaje,
        "run_id": t.run_id,
        "dashboard": f"/dashboard/{t.run_id}" if t.run_id else None,
        "resumen": t.resumen,
        "error": t.error,
        "n_logs": len(t.logs),
    }


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
@app.get("/")
def panel() -> Response:
    html = (AQUI / "templates" / "panel.xhtml").read_text(encoding="utf-8")
    return Response(html, media_type=XHTML)


@app.get("/api/escenarios")
def escenarios() -> dict:
    import json

    nombres = sorted(p.name for p in CARPETA_CONFIG.glob("*.yaml"))
    base = cargar_config(CARPETA_CONFIG / "base.yaml")
    ruta_flota = base.ruta(base.datos.carpeta_intermedios) / "flota.json"
    sugeridos = {}
    if ruta_flota.exists():
        sugeridos = json.loads(ruta_flota.read_text(encoding="utf-8"))["camiones_sugeridos"]
    return {
        "escenarios": nombres,
        "replicas": base.corrida.replicas,
        "semilla": base.corrida.semilla,
        "camiones_sugeridos": sugeridos,
        "listo_para_simular": bool(sugeridos),
    }


@app.post("/api/ejecutar", status_code=202)
def ejecutar(p: Parametros) -> dict:
    _yaml_seguro(p.escenario_base)
    _yaml_seguro(p.escenario_disrupcion)
    if not CANDADO.acquire(blocking=False):
        raise HTTPException(409, "Ya hay una simulación en curso")
    t = Trabajo(id=uuid.uuid4().hex[:12])
    TRABAJOS[t.id] = t
    threading.Thread(target=_correr, args=(t, p), daemon=True).start()
    return _estado(t)


@app.get("/api/estado/{job}")
def estado(job: str) -> dict:
    return _estado(_trabajo(job))


@app.get("/api/logs/{job}")
def logs(job: str, desde: int = 0) -> dict:
    t = _trabajo(job)
    return {"logs": t.logs[max(desde, 0) :], "total": len(t.logs)}


@app.post("/api/dashboard/{job}")
def regenerar_dashboard(job: str) -> dict:
    t = _trabajo(job)
    if not t.carpeta:
        raise HTTPException(409, "La corrida aún no termina")
    from andesbev_sim.reportes.dashboard import generar

    generar(t.carpeta)
    return {"dashboard": f"/dashboard/{t.run_id}"}


@app.get("/dashboard/{run_id}")
def ver_dashboard(run_id: str) -> FileResponse:
    if not RE_RUN.match(run_id):
        raise HTTPException(400, "Identificador no válido")
    cfg = cargar_config(CARPETA_CONFIG / "base.yaml")
    carpeta = cfg.ruta(cfg.datos.carpeta_salidas) / "corridas" / run_id
    archivo = carpeta / f"dashboard_{run_id}.xhtml"
    if not archivo.exists():
        raise HTTPException(404, "Dashboard no encontrado")
    return FileResponse(archivo, media_type=XHTML)


@app.websocket("/ws/progreso/{job}")
async def ws_progreso(ws: WebSocket, job: str) -> None:
    await ws.accept()
    if job not in TRABAJOS:
        await ws.close(code=4404)
        return
    t, enviados = TRABAJOS[job], 0
    try:
        while True:
            nuevos = t.logs[enviados:]
            enviados += len(nuevos)
            await ws.send_json({"estado": _estado(t), "logs": nuevos})
            if t.estado in ("listo", "error"):
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        return
    await ws.close()


@app.middleware("http")
async def cabeceras_seguridad(request, call_next):
    r = await call_next(request)
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "no-referrer"
    r.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self' ws: wss:; frame-ancestors 'none'"
    )
    return r
