"""Lectura del Parquet v4, validaciones y separación de flujos.

Este módulo es la ÚNICA puerta de entrada de datos al simulador. Lee solo las
columnas necesarias (el Parquet completo no cabe cómodo en memoria), revisa la
calidad de lo que entra y deja dos archivos intermedios livianos:

    data/interim/lineas_demanda.parquet      TIPO_FLUJO == "DEMANDA"
    data/interim/lineas_devolucion.parquet   TIPO_FLUJO == "DEVOLUCION"

Los pasos siguientes (segmentación, pronóstico, flota) leen esos intermedios, no
el Parquet original. Así la depuración se hace una sola vez y queda registrada.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

# Columnas del Parquet v4 que usa el simulador y nombre corto que reciben
COLUMNAS = {
    "FECHA": "FECHA",
    "CAMION": "CAMION",
    "VIAJE": "VIAJE",
    "ID-CLIENTE": "ID_CLIENTE",
    "PRODUCTO": "SKU",
    "CATEGORIA": "CATEGORIA",
    "UNIDAD": "UNIDAD",
    "CANTIDAD": "CANTIDAD",
    "PALLETS_EQUIV": "PALLETS",
    "VALOR LINEA": "VALOR",
    "TIPO_FLUJO": "TIPO_FLUJO",
    "SEGMENTO": "SEGMENTO",
    "COMUNA_NOMBRE": "COMUNA",
    "LATITUD": "LAT",
    "LONGITUD": "LON",
}
CATEGORICAS = ["CAMION", "CATEGORIA", "UNIDAD", "TIPO_FLUJO", "SEGMENTO", "COMUNA"]


class DatosInvalidosError(Exception):
    """El Parquet no tiene la estructura mínima para simular."""


@dataclass
class ResumenIngesta:
    filas_leidas: int = 0
    fecha_min: str = ""
    fecha_max: str = ""
    dias_con_despacho: int = 0
    domingos_con_despacho: list[str] = field(default_factory=list)
    filas_por_flujo: dict[str, int] = field(default_factory=dict)
    demanda: dict[str, float] = field(default_factory=dict)
    devolucion: dict[str, float] = field(default_factory=dict)
    descartes: dict[str, int] = field(default_factory=dict)
    unidades: dict[str, dict[str, float]] = field(default_factory=dict)
    flota_observada: dict[str, float] = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)


def leer_parquet(ruta: Path, log: logging.LoggerAdapter) -> pd.DataFrame:
    """Lee solo las columnas necesarias y verifica que existan."""
    if not ruta.exists():
        raise DatosInvalidosError(
            f"No se encontró el Parquet en {ruta}. Revisa 'datos.parquet' en el YAML "
            "o define la variable de entorno ANDESBEV_PARQUET."
        )
    import pyarrow.parquet as pq

    disponibles = set(pq.ParquetFile(ruta).schema_arrow.names)
    faltantes = [c for c in COLUMNAS if c not in disponibles]
    if faltantes:
        raise DatosInvalidosError(
            f"Faltan columnas en el Parquet: {faltantes}. ¿Es la salida de consolidar_ventas v4?"
        )

    log.info("Leyendo %s (%d columnas de %d)", ruta.name, len(COLUMNAS), len(disponibles))
    df = pd.read_parquet(ruta, columns=list(COLUMNAS)).rename(columns=COLUMNAS)
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    df["FECHA"] = pd.to_datetime(df["FECHA"]).dt.normalize()
    log.info(
        "Filas leídas: %s | memoria: %.0f MB",
        f"{len(df):,}",
        df.memory_usage(deep=True).sum() / 1e6,
        extra={"filas": len(df)},
    )
    return df


def depurar(df: pd.DataFrame, r: ResumenIngesta, log: logging.LoggerAdapter) -> pd.DataFrame:
    """Quita filas que no se pueden simular y registra cada descarte."""
    reglas = {
        "cantidad_cero_o_negativa": df["CANTIDAD"] <= 0,
        "sin_pallets_equivalentes": df["PALLETS"].isna(),
        "sin_fecha": df["FECHA"].isna(),
    }
    malas = pd.Series(False, index=df.index)
    for nombre, mascara in reglas.items():
        n = int(mascara.sum())
        r.descartes[nombre] = n
        if n:
            log.warning(
                "Descarte '%s': %s filas", nombre, f"{n:,}", extra={"regla": nombre, "filas": n}
            )
        malas |= mascara
    return df.loc[~malas].copy()


def _totales(sub: pd.DataFrame) -> dict[str, float]:
    return {
        "lineas": int(len(sub)),
        "cantidad": float(sub["CANTIDAD"].sum()),
        "pallets": round(float(sub["PALLETS"].sum()), 1),
        "valor_mm_clp": round(float(sub["VALOR"].sum()) / 1e6, 1),
        "clientes": int(sub["ID_CLIENTE"].nunique()),
        "skus": int(sub["SKU"].nunique()),
    }


def validar(df: pd.DataFrame, r: ResumenIngesta, log: logging.LoggerAdapter) -> None:
    """Chequeos de calidad. No detienen la corrida: dejan WARNING en el log."""
    r.fecha_min, r.fecha_max = str(df["FECHA"].min().date()), str(df["FECHA"].max().date())
    dias = df["FECHA"].drop_duplicates()
    r.dias_con_despacho = int(len(dias))
    r.domingos_con_despacho = sorted(str(d.date()) for d in dias[dias.dt.dayofweek == 6])
    log.info("Período %s a %s | %d días", r.fecha_min, r.fecha_max, r.dias_con_despacho)
    if r.domingos_con_despacho:
        msg = f"Hay {len(r.domingos_con_despacho)} domingos con despacho: {r.domingos_con_despacho}"
        log.warning(msg)
        r.advertencias.append(msg)

    # Meses: deben estar los 36 sin huecos
    meses = df["FECHA"].dt.to_period("M").drop_duplicates().sort_values()
    esperados = pd.period_range(meses.min(), meses.max(), freq="M")
    huecos = sorted(str(m) for m in set(esperados) - set(meses))
    if huecos:
        msg = f"Meses sin datos: {huecos}"
        log.error(msg)
        r.advertencias.append(msg)
    else:
        log.info("Cobertura mensual completa: %d meses", len(esperados))

    # Georreferencia: sin coordenadas no se puede asignar sector
    sin_geo = df["LAT"].isna() | df["LON"].isna()
    p = float(sin_geo.mean())
    msg = f"{p:.2%} de las líneas no tiene coordenadas (se asignarán a 'sin_geo')"
    (log.warning if p > 0 else log.info)(msg, extra={"pct_sin_geo": round(p, 5)})
    if p > 0:
        r.advertencias.append(msg)

    # Mezcla de unidades: justifica usar pallets como unidad física
    dem = df[df["TIPO_FLUJO"] == "DEMANDA"]
    u = dem.groupby("UNIDAD", observed=True).agg(
        lineas=("CANTIDAD", "size"), cantidad=("CANTIDAD", "sum"), pallets=("PALLETS", "sum")
    )
    r.unidades = {
        k: {kk: round(float(vv), 1) for kk, vv in v.items()} for k, v in u.to_dict("index").items()
    }
    if len(u) > 1:
        log.info("Unidades mezcladas %s: se simula en pallets equivalentes", list(u.index))


def observar_flota(dem: pd.DataFrame, r: ResumenIngesta, log: logging.LoggerAdapter) -> None:
    """Primera mirada a la flota real (camión-día). El paso 4 la usa para calibrar."""
    td = dem.groupby(["FECHA", "CAMION"], observed=True).agg(
        pallets=("PALLETS", "sum"), paradas=("ID_CLIENTE", "nunique")
    )
    por_dia = td.groupby(level="FECHA").size()
    r.flota_observada = {
        "camiones_distintos": int(dem["CAMION"].nunique()),
        "camiones_por_dia_mediana": float(por_dia.median()),
        "pallets_por_camion_dia_mediana": round(float(td["pallets"].median()), 1),
        "pallets_por_camion_dia_p90": round(float(td["pallets"].quantile(0.9)), 1),
        "paradas_por_camion_dia_mediana": float(td["paradas"].median()),
        "paradas_por_camion_dia_p90": float(td["paradas"].quantile(0.9)),
    }
    log.info("Flota observada: %s", r.flota_observada, extra={"flota": r.flota_observada})


def ejecutar_ingesta(
    ruta_parquet: Path, carpeta_salida: Path, log: logging.LoggerAdapter
) -> ResumenIngesta:
    """Paso completo: leer, depurar, validar, separar flujos y guardar intermedios."""
    r = ResumenIngesta()
    df = leer_parquet(ruta_parquet, log)
    r.filas_leidas = int(len(df))
    r.filas_por_flujo = {str(k): int(v) for k, v in df["TIPO_FLUJO"].value_counts().items()}
    log.info("Filas por tipo de flujo: %s", r.filas_por_flujo)

    # El simulador usa solo demanda y devoluciones; el resto (envases, subestándar,
    # fleteros) queda fuera de este repositorio y ya está analizado en el Frente 6.
    df = df[df["TIPO_FLUJO"].isin(["DEMANDA", "DEVOLUCION"])]
    df = depurar(df, r, log)
    validar(df, r, log)

    dem = df[df["TIPO_FLUJO"] == "DEMANDA"].drop(columns="TIPO_FLUJO")
    dev = df[df["TIPO_FLUJO"] == "DEVOLUCION"].drop(columns="TIPO_FLUJO")
    r.demanda, r.devolucion = _totales(dem), _totales(dev)
    tasa = r.devolucion["pallets"] / r.demanda["pallets"]
    log.info(
        "Demanda: %s pallets | Devolución: %s pallets | tasa %.2f%%",
        f"{r.demanda['pallets']:,.0f}",
        f"{r.devolucion['pallets']:,.0f}",
        100 * tasa,
        extra={"tasa_devolucion_pallets": round(tasa, 5)},
    )
    observar_flota(dem, r, log)

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    dem.to_parquet(carpeta_salida / "lineas_demanda.parquet", index=False)
    dev.to_parquet(carpeta_salida / "lineas_devolucion.parquet", index=False)
    with open(carpeta_salida / "resumen_ingesta.json", "w", encoding="utf-8") as f:
        json.dump(asdict(r), f, ensure_ascii=False, indent=2)
    log.info("Intermedios guardados en %s", carpeta_salida)
    return r
