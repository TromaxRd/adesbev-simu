"""Flota: calibración con la operación real y dimensionamiento del piloto.

Calibración (datos 2023-2025, columnas CAMION y VIAJE del Parquet):
  Cada camión-día se clasifica según lo que repartió:
    DEDICADA           >= 80 % de sus pallets a clientes ancla
    CAPILAR_PERIFERIA  <= 20 % ancla y >= 50 % de sus pallets en la periferia
    CAPILAR_NUCLEO     <= 20 % ancla y el resto en el núcleo
    MIXTA              todo lo demás (hoy, la mitad de los camiones-día)
  La capacidad de un camión se toma como el percentil 95 de lo que realmente
  hizo en un día: pallets para la ruta dedicada (la que se llena) y paradas
  para las rutas capilares (las que se quedan sin tiempo antes que sin espacio).

Dimensionamiento: para cada pool se toma un percentil alto de la demanda diaria
esperada de 2026 (pallets y visitas) y se calculan los camiones necesarios para
cubrirla con la capacidad calibrada, quedándose con la restricción que mande.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from andesbev_sim.simulacion.red import POOLS, Red, a_unidades


@dataclass
class CapacidadCamion:
    pallets_dia: float
    paradas_dia: float


def perfilar_camion_dia(
    lineas: pd.DataFrame, clientes: pd.DataFrame, umbral_ancla: float, umbral_periferia: float
) -> pd.DataFrame:
    lin = lineas[["FECHA", "CAMION", "VIAJE", "ID_CLIENTE", "PALLETS"]].merge(
        clientes[["ID_CLIENTE", "CANAL", "ZONA"]], on="ID_CLIENTE"
    )
    lin["CAMION"] = lin["CAMION"].astype(str)
    lin["P_ANCLA"] = np.where(lin["CANAL"] == "ANCLA", lin["PALLETS"], 0.0)
    lin["P_PERIF"] = np.where(lin["ZONA"] == "PERIFERIA", lin["PALLETS"], 0.0)
    td = lin.groupby(["FECHA", "CAMION"]).agg(
        pallets=("PALLETS", "sum"),
        paradas=("ID_CLIENTE", "nunique"),
        viajes=("VIAJE", "nunique"),
        p_ancla=("P_ANCLA", "sum"),
        p_perif=("P_PERIF", "sum"),
    )
    sh_a = td["p_ancla"] / td["pallets"]
    sh_p = td["p_perif"] / td["pallets"]
    td["TIPO"] = np.select(
        [
            sh_a >= umbral_ancla,
            (sh_a <= 1 - umbral_ancla) & (sh_p >= umbral_periferia),
            sh_a <= 1 - umbral_ancla,
        ],
        ["DEDICADA", "CAPILAR_PERIFERIA", "CAPILAR_NUCLEO"],
        "MIXTA",
    )
    return td.reset_index()


def calibrar(td: pd.DataFrame, percentil: float) -> tuple[dict[str, CapacidadCamion], pd.DataFrame]:
    """Capacidad por tipo de ruta y resumen descriptivo de la operación observada."""
    q = percentil
    resumen = td.groupby("TIPO").agg(
        camion_dias=("pallets", "size"),
        camiones_por_dia=("FECHA", lambda s: s.value_counts().mean()),
        pallets_mediana=("pallets", "median"),
        pallets_p95=("pallets", lambda x: x.quantile(q)),
        paradas_mediana=("paradas", "median"),
        paradas_p95=("paradas", lambda x: x.quantile(q)),
        viajes_promedio=("viajes", "mean"),
    )
    resumen["pct_camion_dias"] = resumen["camion_dias"] / resumen["camion_dias"].sum()
    # Espacio físico de un camión: lo que llega a cargar una ruta dedicada
    espacio = float(resumen.loc["DEDICADA", "pallets_p95"])
    cap = {
        "DEDICADA": CapacidadCamion(espacio, float(resumen.loc["DEDICADA", "paradas_p95"])),
        "CAPILAR_NUCLEO": CapacidadCamion(
            espacio, float(resumen.loc["CAPILAR_NUCLEO", "paradas_p95"])
        ),
        "CAPILAR_PERIFERIA": CapacidadCamion(
            espacio, float(resumen.loc["CAPILAR_PERIFERIA", "paradas_p95"])
        ),
    }
    return cap, resumen.reset_index()


def tasas_devolucion(dev: pd.DataFrame, dem: pd.DataFrame, clientes: pd.DataFrame) -> dict:
    """Pallets devueltos / pallets despachados, por pool."""
    c = clientes[["ID_CLIENTE", "CANAL", "ZONA"]]
    from andesbev_sim.simulacion.red import pool_de

    c = c.assign(POOL=[pool_de(a, z) for a, z in zip(c["CANAL"], c["ZONA"], strict=True)])
    d = dev.merge(c, on="ID_CLIENTE").groupby("POOL")["PALLETS"].sum()
    v = dem.merge(c, on="ID_CLIENTE").groupby("POOL")["PALLETS"].sum()
    return (d / v).reindex(POOLS).fillna(0).round(5).to_dict()


def dimensionar(
    red: Red,
    esperado_seg: np.ndarray,
    meses: np.ndarray,
    cap: dict[str, CapacidadCamion],
    percentil: float,
    log: logging.LoggerAdapter,
) -> pd.DataFrame:
    """Camiones por pool para cubrir el percentil indicado de la demanda diaria esperada."""
    pal, vis = a_unidades(red, esperado_seg, meses)  # [día, unidad]
    filas = []
    for pool in POOLS:
        m = red.mascara_pool(pool)
        p_dia, v_dia = pal[:, m].sum(axis=1), vis[:, m].sum(axis=1)
        p_q, v_q = np.quantile(p_dia, percentil), np.quantile(v_dia, percentil)
        c = cap[pool]
        por_pallets, por_paradas = p_q / c.pallets_dia, v_q / c.paradas_dia
        n = math.ceil(max(por_pallets, por_paradas))
        filas.append(
            {
                "POOL": pool,
                "pallets_dia_promedio": p_dia.mean(),
                "pallets_dia_percentil": p_q,
                "visitas_dia_promedio": v_dia.mean(),
                "visitas_dia_percentil": v_q,
                "camiones_por_pallets": por_pallets,
                "camiones_por_paradas": por_paradas,
                "restriccion": "pallets" if por_pallets >= por_paradas else "paradas",
                "camiones": n,
            }
        )
        log.info(
            "Pool %s: %d camiones (manda %s)",
            pool,
            n,
            filas[-1]["restriccion"],
            extra={"pool": pool, "camiones": n},
        )
    return pd.DataFrame(filas)


def capacidades_a_dict(cap: dict[str, CapacidadCamion]) -> dict:
    return {k: asdict(v) for k, v in cap.items()}
