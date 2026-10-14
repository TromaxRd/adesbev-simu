"""Motor de simulación de paso diario con pedidos pendientes (backlog).

Para cada día de despacho y cada réplica (todas las réplicas se calculan juntas,
como una dimensión más de los arreglos):

1. Llega la demanda del día (pronóstico estocástico del paso 3), repartida a las
   unidades canal-sector. Un peak anómalo la multiplica; un quiebre de stock deja
   parte de ella sin poder despacharse ese día.
2. Cada pool de flota tiene una capacidad diaria en pallets (camiones × pallets
   por camión) y en paradas (camiones × paradas por camión). Si lo pedido supera
   cualquiera de las dos, el pool atiende solo una fracción ``f`` de lo pedido.
   Un bloqueo vial reduce además la capacidad de las unidades afectadas.
3. Dentro de cada unidad se despacha primero lo más antiguo (FIFO).
4. Lo que no se despacha queda pendiente; lo que ya esperó ``dias_backlog_max``
   días se registra como venta perdida.
5. Lo despachado genera devoluciones (logística inversa) con la tasa empírica de
   su pool. Viajan en el espacio que liberan las entregas, así que no consumen
   capacidad de despacho, pero se registran como KPI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from andesbev_sim.simulacion.disrupciones import Efectos
from andesbev_sim.simulacion.red import POOLS, Red, a_unidades


@dataclass
class CapacidadPool:
    camiones: int
    pallets: float  # por día, pool completo
    paradas: float  # por día, pool completo


@dataclass
class Resultado:
    """Arreglos [réplica, día, unidad] en pallets (salvo indicación)."""

    demanda: np.ndarray
    despachado: np.ndarray
    a_tiempo: np.ndarray  # despachado el mismo día en que se pidió
    pendiente: np.ndarray  # backlog al cierre del día
    perdido: np.ndarray
    devuelto: np.ndarray
    visitas_pedidas: np.ndarray
    visitas_hechas: np.ndarray
    uso_pallets: np.ndarray  # [réplica, día, pool] fracción de capacidad usada
    uso_paradas: np.ndarray  # [réplica, día, pool]


def simular(
    red: Red,
    demanda_seg: np.ndarray,
    meses: np.ndarray,
    capacidad: dict[str, CapacidadPool],
    efectos: Efectos,
    dias_backlog_max: int,
    tasa_devolucion: dict[str, float],
    log: logging.LoggerAdapter,
    progreso: Callable[[float, str], None] | None = None,
) -> Resultado:
    rep, dias, _ = demanda_seg.shape
    n = red.n
    k = max(dias_backlog_max, 0)

    # Demanda despachable y diferida por quiebre, ya en unidades
    pal_total, _ = a_unidades(red, demanda_seg, meses)
    pal_dif, _ = a_unidades(red, demanda_seg * efectos.fraccion_diferida, meses)
    pal_total = pal_total * efectos.factor_demanda
    pal_dif = pal_dif * efectos.factor_demanda
    pv = red.pallets_visita[meses - 1]  # [día, unidad]

    pools = [red.mascara_pool(p) for p in POOLS]
    tasa_u = np.zeros(n)
    for p, m in zip(POOLS, pools, strict=True):
        tasa_u[m] = tasa_devolucion.get(p, 0.0)

    shape = (rep, dias, n)
    r = Resultado(
        demanda=pal_total,
        despachado=np.zeros(shape),
        a_tiempo=np.zeros(shape),
        pendiente=np.zeros(shape),
        perdido=np.zeros(shape),
        devuelto=np.zeros(shape),
        visitas_pedidas=np.zeros(shape),
        visitas_hechas=np.zeros(shape),
        uso_pallets=np.zeros((rep, dias, len(POOLS))),
        uso_paradas=np.zeros((rep, dias, len(POOLS))),
    )
    # backlog[edad] con edad 0 = esperó 1 día, ..., edad k-1 = esperó k días
    backlog = np.zeros((max(k, 1), rep, n))

    for t in range(dias):
        nuevo = pal_total[:, t] - pal_dif[:, t]  # despachable hoy
        viejo = backlog.sum(axis=0) if k else np.zeros((rep, n))
        pedido = viejo + nuevo
        visitas = pedido / pv[t]
        r.visitas_pedidas[:, t] = visitas

        # --- fracción atendible por pool (restricción de pallets y de paradas)
        f = np.ones((rep, n))
        for pool, m in zip(POOLS, pools, strict=True):
            if not m.any():
                continue
            c = capacidad[pool]
            p_req, v_req = pedido[:, m].sum(axis=1), visitas[:, m].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                f_pal = np.where(p_req > 0, c.pallets / p_req, 1.0)
                f_par = np.where(v_req > 0, c.paradas / v_req, 1.0)
            f_pool = np.clip(np.minimum(f_pal, f_par), 0, 1)
            f[:, m] = f_pool[:, None]
        f = f * efectos.factor_capacidad[:, t]
        capacidad_u = f * pedido  # pallets que se pueden despachar por unidad

        # --- FIFO: primero lo más antiguo
        restante = capacidad_u.copy()
        for edad in range(k - 1, -1, -1):
            sirve = np.minimum(backlog[edad], restante)
            backlog[edad] -= sirve
            restante -= sirve
        a_tiempo = np.minimum(nuevo, restante)
        despachado = capacidad_u - (restante - a_tiempo)

        # --- envejecer pendientes
        no_servido_hoy = (nuevo - a_tiempo) + pal_dif[:, t]
        if k:
            perdido = backlog[k - 1].copy()
            backlog[1:] = backlog[:-1].copy()
            backlog[0] = no_servido_hoy
        else:
            perdido = no_servido_hoy

        r.despachado[:, t] = despachado
        r.a_tiempo[:, t] = a_tiempo
        r.perdido[:, t] = perdido
        r.pendiente[:, t] = backlog.sum(axis=0) if k else 0.0
        r.devuelto[:, t] = despachado * tasa_u
        r.visitas_hechas[:, t] = despachado / pv[t]
        for j, (pool, m) in enumerate(zip(POOLS, pools, strict=True)):
            c = capacidad[pool]
            if c.pallets > 0:
                r.uso_pallets[:, t, j] = despachado[:, m].sum(axis=1) / c.pallets
            if c.paradas > 0:
                r.uso_paradas[:, t, j] = r.visitas_hechas[:, t][:, m].sum(axis=1) / c.paradas

        if progreso and (t % 15 == 0 or t == dias - 1):
            progreso((t + 1) / dias, f"día {t + 1}/{dias}")

    perdido_total = r.perdido.sum()
    if perdido_total > 0:
        log.warning(
            "Venta perdida: %.0f pallets promedio por réplica (%.2f%% de la demanda)",
            perdido_total / rep,
            100 * perdido_total / r.demanda.sum(),
            extra={"pallets_perdidos_replica": round(float(perdido_total / rep), 1)},
        )
    return r
