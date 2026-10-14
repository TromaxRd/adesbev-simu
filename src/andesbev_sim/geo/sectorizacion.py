"""Zonificación núcleo/periferia y sectorización de clientes.

1. Cada cliente recibe una coordenada única y su distancia al centro de
   distribución (CD).
2. El CD de referencia se verifica recalculando el punto de Weber: el punto que
   minimiza la suma de distancias ponderadas por volumen (algoritmo de Weiszfeld).
3. Zona: NUCLEO si está a menos de ``radio_nucleo_km`` del CD, PERIFERIA si no,
   SIN_GEO si no tiene coordenadas.
4. Sectores: K-means sobre coordenadas en km. En el núcleo se pondera por visitas
   (tiempo de reparto); en la periferia no se pondera, porque ahí los sectores
   representan localidades (Vicuña, Andacollo, Tongoy...).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

KM_POR_GRADO = 111.32


def a_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Proyección equirectangular a km respecto de (lat0, lon0).

    A la escala del piloto (< 100 km) el error frente a la distancia geodésica es
    menor a 0,5 %, y permite usar K-means con distancias euclidianas.
    """
    x = (np.asarray(lon) - lon0) * KM_POR_GRADO * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * KM_POR_GRADO
    return np.column_stack([x, y])


def distancia_km(lat, lon, lat0: float, lon0: float) -> np.ndarray:
    return np.linalg.norm(a_km(lat, lon, lat0, lon0), axis=1)


def punto_weber(
    lat: np.ndarray, lon: np.ndarray, peso: np.ndarray, tol_km: float = 1e-4, max_iter: int = 500
) -> tuple[float, float]:
    """Mediana geométrica ponderada (Weiszfeld). Parte del centroide ponderado."""
    lat0, lon0 = float(np.average(lat, weights=peso)), float(np.average(lon, weights=peso))
    xy = a_km(lat, lon, lat0, lon0)
    p = np.array([0.0, 0.0])
    for _ in range(max_iter):
        d = np.maximum(np.linalg.norm(xy - p, axis=1), 1e-9)
        w = peso / d
        nuevo = (xy * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(nuevo - p) < tol_km:
            p = nuevo
            break
        p = nuevo
    lat_w = lat0 + p[1] / KM_POR_GRADO
    lon_w = lon0 + p[0] / (KM_POR_GRADO * np.cos(np.radians(lat0)))
    return float(lat_w), float(lon_w)


def _kmeans(xy: np.ndarray, k: int, peso: np.ndarray | None, semilla: int) -> np.ndarray:
    k = min(k, len(xy))
    modelo = KMeans(n_clusters=k, n_init=20, random_state=semilla)
    return modelo.fit_predict(xy, sample_weight=peso)


def _renumerar(clientes: pd.DataFrame, mascara: pd.Series, prefijo: str, etiquetas) -> None:
    """Asigna códigos N1, N2... ordenados de mayor a menor carga (visitas)."""
    tmp = clientes.loc[mascara, ["visitas"]].assign(cl=etiquetas)
    orden = tmp.groupby("cl")["visitas"].sum().sort_values(ascending=False).index
    nombre = {cl: f"{prefijo}{i + 1}" for i, cl in enumerate(orden)}
    clientes.loc[mascara, "SECTOR"] = [nombre[c] for c in etiquetas]


def sectorizar(
    clientes: pd.DataFrame,
    cd: tuple[float, float],
    radio_nucleo_km: float,
    k_nucleo: int,
    k_periferia: int,
    ponderar_por: str,
    semilla: int,
    log: logging.LoggerAdapter,
) -> pd.DataFrame:
    """Agrega DIST_CD_KM, ZONA y SECTOR a la tabla de clientes."""
    c = clientes.copy()
    con_geo = c["LAT"].notna() & c["LON"].notna()
    c["DIST_CD_KM"] = np.nan
    c.loc[con_geo, "DIST_CD_KM"] = distancia_km(c.loc[con_geo, "LAT"], c.loc[con_geo, "LON"], *cd)
    c["ZONA"] = np.where(
        ~con_geo, "SIN_GEO", np.where(c["DIST_CD_KM"] <= radio_nucleo_km, "NUCLEO", "PERIFERIA")
    )
    c["SECTOR"] = "SIN_GEO"

    nuc = c["ZONA"] == "NUCLEO"
    xy = a_km(c.loc[nuc, "LAT"], c.loc[nuc, "LON"], *cd)
    peso = None if ponderar_por == "ninguno" else c.loc[nuc, ponderar_por].to_numpy(float)
    _renumerar(c, nuc, "N", _kmeans(xy, k_nucleo, peso, semilla))

    per = c["ZONA"] == "PERIFERIA"
    if per.any():
        xy = a_km(c.loc[per, "LAT"], c.loc[per, "LON"], *cd)
        _renumerar(c, per, "P", _kmeans(xy, k_periferia, None, semilla))

    for zona, n in c["ZONA"].value_counts().items():
        log.info("Zona %s: %d clientes", zona, n, extra={"zona": zona, "clientes": int(n)})
    if (~con_geo).any():
        log.warning(
            "%d clientes sin coordenadas quedan en el sector SIN_GEO", int((~con_geo).sum())
        )
    return c


def resumen_sectores(clientes: pd.DataFrame, dias: int) -> pd.DataFrame:
    """Una fila por sector: clientes, carga, distancia y mezcla de canal."""
    g = clientes.groupby(["ZONA", "SECTOR"])
    r = g.agg(
        clientes=("ID_CLIENTE", "size"),
        pallets=("PALLETS", "sum"),
        visitas=("visitas", "sum"),
        dist_cd_km_mediana=("DIST_CD_KM", "median"),
        lat=("LAT", "mean"),
        lon=("LON", "mean"),
        comuna_dominante=("COMUNA", lambda s: s.mode().iat[0] if s.notna().any() else ""),
    )
    pal_ancla = clientes[clientes["CANAL"] == "ANCLA"].groupby(["ZONA", "SECTOR"])["PALLETS"].sum()
    r["pct_pallets_ancla"] = (pal_ancla.reindex(r.index).fillna(0) / r["pallets"]).round(3)
    r["visitas_por_dia"] = (r["visitas"] / dias).round(1)
    r["pallets_por_dia"] = (r["pallets"] / dias).round(2)
    r["pallets_por_visita"] = (r["pallets"] / r["visitas"]).round(3)
    r["pct_pallets"] = (r["pallets"] / r["pallets"].sum()).round(4)
    return r.reset_index().sort_values(["ZONA", "visitas"], ascending=[True, False])
