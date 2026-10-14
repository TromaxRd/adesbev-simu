"""Segmentación de la cartera: productos (ABC/XYZ) y clientes (canal ancla/capilar).

ABC — cuánto pesa cada SKU en el volumen total (pallets). Se ordenan de mayor a
      menor y se corta por porcentaje acumulado: A hasta 80 %, B hasta 95 %.
XYZ — qué tan estable es su demanda. Coeficiente de variación (CV = desviación /
      media) de los pallets semanales, contado desde la primera hasta la última
      semana con venta de ese SKU (así un producto lanzado en 2025 no se castiga
      por las semanas en que todavía no existía).
Canal — cada segmento comercial (SUPERMERCADOS, ALMACEN...) es ANCLA si la
      mediana de pallets que recibe por visita supera el umbral, y CAPILAR si no.
      Así la regla sale de la operación observada y no de un juicio a priori.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def clasificar_abc(volumen: pd.Series, cortes: tuple[float, float]) -> pd.DataFrame:
    """volumen: pallets por SKU. Devuelve participación, acumulado y clase."""
    v = volumen.sort_values(ascending=False)
    part = v / v.sum()
    acum = part.cumsum()
    # El acumulado ANTERIOR decide la clase: así el SKU que cruza el 80 % sigue en A
    previo = acum - part
    clase = np.where(previo < cortes[0], "A", np.where(previo < cortes[1], "B", "C"))
    return pd.DataFrame({"pallets": v, "participacion": part, "acumulado": acum, "ABC": clase})


def clasificar_xyz(semanal: pd.DataFrame, cortes: tuple[float, float]) -> pd.DataFrame:
    """semanal: columnas SKU, SEMANA, PALLETS (solo semanas con venta)."""
    filas = []
    todas = pd.Index(sorted(semanal["SEMANA"].unique()))
    for sku, g in semanal.groupby("SKU"):
        rango = todas[(todas >= g["SEMANA"].min()) & (todas <= g["SEMANA"].max())]
        serie = g.set_index("SEMANA")["PALLETS"].reindex(rango, fill_value=0.0)
        media = serie.mean()
        cv = serie.std(ddof=0) / media if media > 0 else np.inf
        filas.append((sku, len(rango), float(cv)))
    r = pd.DataFrame(filas, columns=["SKU", "semanas_vigente", "CV"]).set_index("SKU")
    r["XYZ"] = np.where(r["CV"] < cortes[0], "X", np.where(r["CV"] < cortes[1], "Y", "Z"))
    return r


def canal_por_segmento(lineas: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Mediana de pallets por visita (cliente-día) de cada segmento comercial."""
    visitas = lineas.groupby(["SEGMENTO", "ID_CLIENTE", "FECHA"], observed=True)["PALLETS"].sum()
    r = visitas.groupby(level="SEGMENTO", observed=True).agg(
        pallets_visita_mediana="median", visitas="size"
    )
    r["clientes"] = lineas.groupby("SEGMENTO", observed=True)["ID_CLIENTE"].nunique()
    r["CANAL"] = np.where(r["pallets_visita_mediana"] > umbral, "ANCLA", "CAPILAR")
    return r.sort_values("pallets_visita_mediana", ascending=False)


def tabla_clientes(lineas: pd.DataFrame, canales: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cliente con su coordenada, segmento, canal y volumen."""
    c = lineas.groupby("ID_CLIENTE").agg(
        LAT=("LAT", "median"),
        LON=("LON", "median"),
        SEGMENTO=("SEGMENTO", "first"),
        COMUNA=("COMUNA", "first"),
        PALLETS=("PALLETS", "sum"),
        visitas=("FECHA", "nunique"),
    )
    c["SEGMENTO"] = c["SEGMENTO"].astype(str)
    c["COMUNA"] = c["COMUNA"].astype(str)
    c["CANAL"] = c["SEGMENTO"].map(canales["CANAL"]).fillna("CAPILAR")
    return c.reset_index()


def segmentar_skus(
    lineas: pd.DataFrame,
    cortes_abc: tuple[float, float],
    cortes_xyz: tuple[float, float],
    log: logging.LoggerAdapter,
) -> pd.DataFrame:
    vol = lineas.groupby("SKU")["PALLETS"].sum()
    abc = clasificar_abc(vol[vol > 0], cortes_abc)
    semana = lineas["FECHA"].dt.to_period("W-SUN").dt.start_time
    sem = lineas.assign(SEMANA=semana).groupby(["SKU", "SEMANA"], as_index=False)["PALLETS"].sum()
    xyz = clasificar_xyz(sem, cortes_xyz)
    cat = lineas.groupby("SKU")["CATEGORIA"].agg(lambda s: s.mode().iat[0])
    skus = abc.join(xyz).join(cat.astype(str))
    skus["CLASE"] = skus["ABC"] + skus["XYZ"]
    for clase, g in skus.groupby("ABC"):
        log.info(
            "Clase %s: %d SKU, %.1f%% del volumen",
            clase,
            len(g),
            100 * g["participacion"].sum(),
            extra={"clase_abc": clase, "skus": len(g)},
        )
    return skus.reset_index()
