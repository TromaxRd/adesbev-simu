"""Ejecución de escenarios y comparación base vs disrupción.

Números aleatorios comunes: la demanda se genera con la semilla de la corrida y
es idéntica en el escenario base y en el de disrupción. Las disrupciones usan un
generador aparte. Así, réplica por réplica, la única diferencia entre ambos
escenarios es la falla inyectada, y la comparación mide solo su efecto.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from andesbev_sim.config import Config
from andesbev_sim.demanda import pronostico as pr
from andesbev_sim.kpis import metricas
from andesbev_sim.logging_setup import contexto
from andesbev_sim.simulacion import disrupciones, motor
from andesbev_sim.simulacion.red import POOLS, Red, construir_red

Progreso = Callable[[float, str], None]


@dataclass
class Corrida:
    nombre: str
    red: Red
    fechas: pd.DatetimeIndex
    capacidad: dict[str, motor.CapacidadPool]
    resultado: motor.Resultado
    eventos: pd.DataFrame
    kpis: dict


def capacidad_flota(cfg: Config, flota: dict) -> dict[str, motor.CapacidadPool]:
    out = {}
    for p in POOLS:
        n = getattr(cfg.flota.camiones, p)
        n = flota["camiones_sugeridos"][p] if n is None else n
        c = flota["capacidad_camion"][p]
        fac = cfg.flota.factor_capacidad
        out[p] = motor.CapacidadPool(n, n * c["pallets_dia"] * fac, n * c["paradas_dia"] * fac)
    return out


def generar_demanda(cfg: Config, params: pr.ParametrosDemanda) -> np.ndarray:
    rng = np.random.default_rng(cfg.corrida.semilla)
    return pr.simular(params.modelo, params.fechas, params.nivel, cfg.corrida.replicas, rng)


def ejecutar(
    cfg: Config,
    params: pr.ParametrosDemanda,
    flota: dict,
    demanda: np.ndarray,
    log: logging.LoggerAdapter,
    progreso: Progreso | None = None,
) -> Corrida:
    nombre = cfg.corrida.nombre
    log = contexto(log, escenario=nombre)
    red = construir_red(params.reparto, params.modelo.segmentos)
    fechas = params.fechas
    cap = capacidad_flota(cfg, flota)
    for p, c in cap.items():
        log.info(
            "Flota %s: %d camiones | %.0f pallets/día | %.0f paradas/día",
            p,
            c.camiones,
            c.pallets,
            c.paradas,
            extra={"pool": p, "camiones": c.camiones},
        )
    if cap["CAPILAR_PERIFERIA"].camiones == 0:
        log.error("El pool CAPILAR_PERIFERIA no tiene camiones: la periferia no se atenderá")

    rng_dis = np.random.default_rng(cfg.corrida.semilla + 1)
    ef = disrupciones.generar(
        cfg.disrupciones, red, params.modelo.segmentos, fechas, cfg.corrida.replicas, rng_dis, log
    )
    eventos = disrupciones.tabla_eventos(ef, fechas)
    for e in eventos[eventos["replica"] == 0].itertuples() if not eventos.empty else []:
        contexto(log, replica=0).warning(
            "Evento '%s' inyectado: %s a %s (%d días)",
            e.nombre,
            e.fecha_inicio,
            e.fecha_fin,
            e.duracion_dias,
            extra={"evento": e.nombre, "dia_inicio": e.dia_inicio, "dia_fin": e.dia_fin},
        )

    res = motor.simular(
        red,
        demanda,
        fechas.month.to_numpy(),
        cap,
        ef,
        cfg.simulacion.dias_backlog_max,
        flota["tasa_devolucion"],
        log,
        progreso,
    )
    k = metricas.kpis(res, red, fechas, cap)

    # Trazabilidad por réplica: impacto en KPIs por pool
    for r in range(cfg.corrida.replicas):
        for p in POOLS:
            m = red.mascara_pool(p)
            dem = res.demanda[r][:, m].sum()
            serv = res.a_tiempo[r][:, m].sum() / dem if dem else 1.0
            contexto(log, replica=r, pool=p).info(
                "Réplica %d | %s | servicio a tiempo %.2f%% | perdido %.1f pallets",
                r,
                p,
                100 * serv,
                res.perdido[r][:, m].sum(),
                extra={"servicio_a_tiempo": round(float(serv), 5)},
            )
    g = k["global"]
    log.info(
        "Escenario %s: servicio a tiempo %.2f%% ± %.2f | fill rate %.2f%% | backlog medio %.1f",
        nombre,
        100 * g["servicio_a_tiempo"]["media"],
        100 * g["servicio_a_tiempo"]["ic95"],
        100 * g["fill_rate"]["media"],
        g["backlog_promedio_pallets"]["media"],
    )
    return Corrida(nombre, red, fechas, cap, res, eventos, k)


def comparar(base: Corrida, dis: Corrida, tol: float, log: logging.LoggerAdapter) -> dict:
    tabla = metricas.recuperacion(base.resultado, dis.resultado, dis.eventos, tol)
    resumen = metricas.resumen_recuperacion(tabla)
    gb, gd = base.kpis["global"], dis.kpis["global"]
    comp = {
        "servicio_a_tiempo_base": gb["servicio_a_tiempo"]["media"],
        "servicio_a_tiempo_disrupcion": gd["servicio_a_tiempo"]["media"],
        "perdida_servicio_pp": 100
        * (gb["servicio_a_tiempo"]["media"] - gd["servicio_a_tiempo"]["media"]),
        "fill_rate_base": gb["fill_rate"]["media"],
        "fill_rate_disrupcion": gd["fill_rate"]["media"],
        "venta_perdida_extra_pallets": gd["venta_perdida_pallets"]["media"]
        - gb["venta_perdida_pallets"]["media"],
        "backlog_maximo_base": gb["backlog_maximo_pallets"]["media"],
        "backlog_maximo_disrupcion": gd["backlog_maximo_pallets"]["media"],
        "por_zona": {
            z: {
                "servicio_base": base.kpis["por_zona"][z]["servicio_a_tiempo"]["media"],
                "servicio_disrupcion": dis.kpis["por_zona"][z]["servicio_a_tiempo"]["media"],
            }
            for z in base.kpis["por_zona"]
        },
    }
    for row in resumen.itertuples() if not resumen.empty else []:
        log.warning(
            "Resiliencia '%s': recuperación %.1f días (p90 %.0f) | pérdida de servicio %.1f pp",
            row.nombre,
            row.dias_recuperacion,
            row.dias_recuperacion_p90,
            row.perdida_servicio_pp,
            extra={
                "evento": row.nombre,
                "dias_recuperacion": round(float(row.dias_recuperacion), 2),
            },
        )
    return {"comparacion": comp, "recuperacion": resumen, "recuperacion_detalle": tabla}


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def _a_json(x):
    if isinstance(x, pd.DataFrame):
        return x.to_dict("records")
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _a_json(v) for k, v in x.items()}
    return x


def guardar_corrida(c: Corrida, carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    k = c.kpis
    k["diario"].to_csv(carpeta / f"{c.nombre}_diario.csv", index=False)
    k["mensual"].to_csv(carpeta / f"{c.nombre}_mensual.csv", index=False)
    k["sectores"].to_csv(carpeta / f"{c.nombre}_sectores.csv", index=False)
    if not c.eventos.empty:
        c.eventos.to_csv(carpeta / f"{c.nombre}_eventos.csv", index=False)
    resto = {kk: v for kk, v in k.items() if kk not in ("diario", "mensual", "sectores")}
    with open(carpeta / f"{c.nombre}_kpis.json", "w", encoding="utf-8") as f:
        json.dump(_a_json(resto), f, ensure_ascii=False, indent=2, default=str)


def guardar_comparacion(comp: dict, carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    with open(carpeta / "comparacion.json", "w", encoding="utf-8") as f:
        json.dump(_a_json(comp["comparacion"]), f, ensure_ascii=False, indent=2)
    comp["recuperacion"].to_csv(carpeta / "recuperacion.csv", index=False)
    comp["recuperacion_detalle"].to_csv(carpeta / "recuperacion_detalle.csv", index=False)
