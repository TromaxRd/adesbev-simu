"""Estructura de la red: unidades (canal × sector) y pools de flota.

Una *unidad* es la combinación canal-sector (p. ej. CAPILAR en N2). Cada unidad
pertenece a un *pool* de flota que la atiende:

    DEDICADA           todas las unidades del canal ANCLA (rutas de alta carga)
    CAPILAR_NUCLEO     canal CAPILAR en el núcleo (y clientes sin coordenadas)
    CAPILAR_PERIFERIA  canal CAPILAR en la periferia

La demanda pronosticada viene por segmento (canal × ABC × XYZ). ``a_unidades``
la reparte a unidades con la participación histórica de cada sector en su canal
para ese mes, y calcula las visitas con los pallets por visita del sector-mes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POOLS = ["DEDICADA", "CAPILAR_NUCLEO", "CAPILAR_PERIFERIA"]


def pool_de(canal: str, zona: str) -> str:
    if canal == "ANCLA":
        return "DEDICADA"
    return "CAPILAR_PERIFERIA" if zona == "PERIFERIA" else "CAPILAR_NUCLEO"


@dataclass
class Red:
    unidades: pd.DataFrame  # CANAL, ZONA, SECTOR, POOL (una fila por unidad)
    seg_a_canal: np.ndarray  # [segmento, canal]
    canales: list[str]
    reparto: np.ndarray  # [mes 1..12 (índice 0..11), canal, unidad]
    pallets_visita: np.ndarray  # [mes, unidad]

    @property
    def n(self) -> int:
        return len(self.unidades)

    def mascara_pool(self, pool: str) -> np.ndarray:
        return (self.unidades["POOL"] == pool).to_numpy()


def construir_red(reparto: pd.DataFrame, segmentos: list[str]) -> Red:
    u = (
        reparto[["CANAL", "ZONA", "SECTOR"]]
        .drop_duplicates()
        .sort_values(["CANAL", "ZONA", "SECTOR"])
    )
    u = u.reset_index(drop=True)
    u["POOL"] = [pool_de(c, z) for c, z in zip(u["CANAL"], u["ZONA"], strict=True)]
    canales = sorted(u["CANAL"].unique())
    pos = {(c, s): i for i, (c, s) in enumerate(zip(u["CANAL"], u["SECTOR"], strict=True))}

    seg_a_canal = np.zeros((len(segmentos), len(canales)))
    for i, s in enumerate(segmentos):
        seg_a_canal[i, canales.index(s.split("-")[0])] = 1.0

    rep = np.zeros((12, len(canales), len(u)))
    pv = np.full((12, len(u)), np.nan)
    for r in reparto.itertuples():
        j = pos[(r.CANAL, r.SECTOR)]
        rep[r.MES - 1, canales.index(r.CANAL), j] = r.participacion
        pv[r.MES - 1, j] = r.pallets_por_visita
    # meses sin registro en una unidad: se usa su promedio anual
    prom = np.nanmean(pv, axis=0)
    pv = np.where(np.isnan(pv), prom, pv)
    return Red(u, seg_a_canal, canales, rep, pv)


def a_unidades(
    red: Red, demanda_seg: np.ndarray, meses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """demanda_seg [..., día, segmento] -> (pallets, visitas) [..., día, unidad]."""
    por_canal = demanda_seg @ red.seg_a_canal  # [..., día, canal]
    w = red.reparto[meses - 1]  # [día, canal, unidad]
    pallets = np.einsum("...dc,dcu->...du", por_canal, w)
    visitas = pallets / red.pallets_visita[meses - 1]
    return pallets, visitas
