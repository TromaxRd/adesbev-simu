"""Inyección de disrupciones: convierte la lista del YAML en efectos día a día.

Cada disrupción produce, para cada réplica, una ventana de días [inicio, fin) y un
efecto sobre la red durante esa ventana:

    bloqueo_vial   la capacidad de las unidades afectadas queda en ``capacidad_restante``
    quiebre_stock  ``fraccion_afectada`` de la demanda de las clases ABC indicadas no se
                   puede despachar ese día y pasa a pedido pendiente (backlog)
    peak_anomalo   la demanda de las unidades afectadas se multiplica por
                   ``multiplicador_demanda``

``zonas`` acepta zonas (NUCLEO, PERIFERIA), sectores (N1, P3...) o pools
(DEDICADA, CAPILAR_NUCLEO, CAPILAR_PERIFERIA); vacío = toda la red.

Modo determinista: misma ventana en todas las réplicas (para medir recuperación).
Modo estocástico: cada día sin evento activo, el evento empieza con probabilidad
``probabilidad_diaria``; la duración se sortea si se da como rango [min, max].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from andesbev_sim.config import Disrupcion
from andesbev_sim.simulacion.red import Red


@dataclass
class Evento:
    replica: int
    tipo: str
    nombre: str
    inicio: int  # índice de día de despacho
    fin: int  # exclusivo
    determinista: bool


@dataclass
class Efectos:
    factor_capacidad: np.ndarray  # [réplica, día, unidad]
    factor_demanda: np.ndarray  # [réplica, día, unidad]
    fraccion_diferida: np.ndarray  # [réplica, día, segmento]
    eventos: list[Evento] = field(default_factory=list)


def mascara_unidades(red: Red, zonas: list[str] | None) -> np.ndarray:
    if not zonas:
        return np.ones(red.n, dtype=bool)
    u = red.unidades
    return (u["ZONA"].isin(zonas) | u["SECTOR"].isin(zonas) | u["POOL"].isin(zonas)).to_numpy()


def mascara_segmentos(segmentos: list[str], clases: list[str] | None) -> np.ndarray:
    clases = clases or ["A"]
    return np.array([s.split("-")[1] in clases for s in segmentos])


def _duracion(d: int | list[int], rng: np.random.Generator) -> int:
    return int(rng.integers(d[0], d[1] + 1)) if isinstance(d, list) else int(d)


def _ventanas(
    dis: Disrupcion, fechas: pd.DatetimeIndex, replicas: int, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    """Lista de (réplica, inicio, fin)."""
    n = len(fechas)
    out = []
    if dis.modo == "determinista":
        if dis.fecha_inicio is not None:
            inicio = int(np.searchsorted(fechas, pd.Timestamp(dis.fecha_inicio)))
        else:
            inicio = int(dis.dia_inicio)
        if inicio >= n:
            raise ValueError(
                f"La disrupción '{dis.nombre or dis.tipo}' empieza fuera del horizonte"
            )
        for r in range(replicas):
            out.append((r, inicio, min(n, inicio + _duracion(dis.duracion_dias, rng))))
    else:
        p = dis.probabilidad_diaria
        for r in range(replicas):
            t = 0
            while t < n:
                if rng.random() < p:
                    fin = min(n, t + _duracion(dis.duracion_dias, rng))
                    out.append((r, t, fin))
                    t = fin
                else:
                    t += 1
    return out


def generar(
    disrupciones: list[Disrupcion],
    red: Red,
    segmentos: list[str],
    fechas: pd.DatetimeIndex,
    replicas: int,
    rng: np.random.Generator,
    log: logging.LoggerAdapter,
) -> Efectos:
    d = len(fechas)
    ef = Efectos(
        factor_capacidad=np.ones((replicas, d, red.n)),
        factor_demanda=np.ones((replicas, d, red.n)),
        fraccion_diferida=np.zeros((replicas, d, len(segmentos))),
    )
    for dis in disrupciones:
        nombre = dis.nombre or dis.tipo
        mu = mascara_unidades(red, dis.zonas)
        for r, ini, fin in _ventanas(dis, fechas, replicas, rng):
            # Se trabaja sobre vistas [día, unidad] de la réplica para indexar sin ambigüedad
            if dis.tipo == "bloqueo_vial":
                ventana = ef.factor_capacidad[r, ini:fin]
                ventana[:, mu] *= dis.capacidad_restante
            elif dis.tipo == "peak_anomalo":
                ventana = ef.factor_demanda[r, ini:fin]
                ventana[:, mu] *= dis.multiplicador_demanda
            else:  # quiebre_stock
                ms = mascara_segmentos(segmentos, dis.clases_abc)
                ventana = ef.fraccion_diferida[r, ini:fin]
                ventana[:, ms] = np.maximum(ventana[:, ms], dis.fraccion_afectada)
            ef.eventos.append(Evento(r, dis.tipo, nombre, ini, fin, dis.modo == "determinista"))
        n_ev = sum(1 for e in ef.eventos if e.nombre == nombre)
        log.warning(
            "Disrupción '%s' (%s, %s): %d eventos en %d réplicas | zonas=%s",
            nombre,
            dis.tipo,
            dis.modo,
            n_ev,
            replicas,
            dis.zonas or "toda la red",
            extra={"disrupcion": nombre, "tipo": dis.tipo, "eventos": n_ev},
        )
    return ef


def tabla_eventos(ef: Efectos, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    filas = [
        {
            "replica": e.replica,
            "nombre": e.nombre,
            "tipo": e.tipo,
            "dia_inicio": e.inicio,
            "dia_fin": e.fin,
            "fecha_inicio": fechas[e.inicio].date(),
            "fecha_fin": fechas[e.fin - 1].date(),
            "duracion_dias": e.fin - e.inicio,
            "determinista": e.determinista,
        }
        for e in ef.eventos
    ]
    return pd.DataFrame(filas)
