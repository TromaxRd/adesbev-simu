"""Generador de una muestra SINTÉTICA con la estructura del Parquet v4.

El Parquet real (~100 MB, datos de la empresa) no se sube a GitHub. Para que las
pruebas de integración y el pipeline de CI corran de punta a punta se usa esta
muestra: 3 años de despachos de ~130 clientes inventados, con estacionalidad,
canal ancla/capilar, núcleo y periferia, camiones y devoluciones. Las cifras NO
representan a AndesBev; solo reproducen la forma de los datos.

Uso:  python -m andesbev_sim.muestra data/sample/ventas_muestra.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from andesbev_sim.ingesta.loader import COLUMNAS

CD = (-29.94, -71.26)
PUEBLOS = {  # localidades de la periferia (lat, lon)
    "VICUÑA": (-30.03, -70.71),
    "ANDACOLLO": (-30.23, -71.08),
    "TONGOY": (-30.25, -71.49),
    "LA HIGUERA": (-29.48, -71.22),
    "ALTOVALSOL": (-29.94, -71.05),
}
IDX_MES = np.array([1.15, 1.22, 0.97, 0.91, 0.84, 0.84, 0.86, 0.91, 1.19, 0.90, 0.97, 1.25])
IDX_DOW = np.array([0.77, 1.03, 1.17, 1.05, 1.07, 0.92])
CATEGORIAS = ["CERVEZAS", "GASEOSAS", "MINERALES", "NECTARES", "ENERGETICAS"]


def _clientes(rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    cid = 1000

    def agregar(n, seg, lat0, lon0, disp, pal_visita, frec, comuna):
        nonlocal cid
        for _ in range(n):
            cid += 1
            filas.append(
                (
                    cid,
                    seg,
                    comuna,
                    lat0 + rng.normal(0, disp),
                    lon0 + rng.normal(0, disp),
                    pal_visita * rng.uniform(0.6, 1.4),
                    frec,
                )
            )

    agregar(8, "SUPERMERCADOS", CD[0], CD[1], 0.03, 4.0, 0.5, "LA SERENA")
    agregar(4, "MAYORISTA", CD[0], CD[1] - 0.03, 0.03, 1.5, 0.4, "COQUIMBO")
    agregar(60, "ALMACEN", CD[0] + 0.02, CD[1] + 0.01, 0.035, 0.12, 0.35, "LA SERENA")
    agregar(30, "ALMACEN", CD[0] - 0.03, CD[1] - 0.05, 0.03, 0.12, 0.35, "COQUIMBO")
    agregar(8, "BOTILLERIA", CD[0], CD[1], 0.04, 0.4, 0.3, "LA SERENA")
    for nombre, (la, lo) in PUEBLOS.items():
        agregar(5, "ALMACEN", la, lo, 0.01, 0.12, 0.3, nombre)
    agregar(1, "SUPERMERCADOS", *PUEBLOS["VICUÑA"], 0.005, 2.0, 0.3, "VICUÑA")
    agregar(3, "ALMACEN", np.nan, np.nan, 0.0, 0.1, 0.2, "COQUIMBO")  # sin coordenadas
    return pd.DataFrame(
        filas, columns=["ID", "SEGMENTO", "COMUNA", "LAT", "LON", "PAL_VISITA", "FREC"]
    )


def generar(semilla: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    cli = _clientes(rng)
    dias = pd.date_range("2023-01-02", "2025-12-31", freq="D")
    dias = dias[dias.dayofweek < 6]
    dias = dias[~((dias.month == 1) & (dias.day == 1)) & ~((dias.month == 9) & (dias.day == 18))]
    mes_f = IDX_MES[dias.month - 1] * IDX_DOW[dias.dayofweek]
    skus = np.arange(100, 120)
    p_sku = 1 / np.arange(1, 21) ** 1.1
    p_sku /= p_sku.sum()

    bloques = []
    for c in cli.itertuples():
        visita = rng.random(len(dias)) < np.clip(c.FREC * mes_f, 0, 1)
        f = dias[visita]
        if len(f) == 0:
            continue
        n_lin = rng.integers(1, 4, len(f))
        fecha = np.repeat(f.to_numpy(), n_lin)
        total = c.PAL_VISITA * rng.lognormal(0, 0.35, len(f)) * IDX_MES[f.month - 1] ** 0.5
        pal = np.repeat(total / n_lin, n_lin)
        ancla = c.SEGMENTO in ("SUPERMERCADOS", "MAYORISTA")
        bloques.append(
            pd.DataFrame(
                {
                    "FECHA": fecha,
                    "ID-CLIENTE": c.ID,
                    "PRODUCTO": rng.choice(skus, size=len(fecha), p=p_sku),
                    "PALLETS_EQUIV": pal,
                    "SEGMENTO": c.SEGMENTO,
                    "COMUNA_NOMBRE": c.COMUNA,
                    "LATITUD": c.LAT,
                    "LONGITUD": c.LON,
                    "_ancla": ancla,
                    "_perif": c.COMUNA in PUEBLOS,
                }
            )
        )
    df = pd.concat(bloques, ignore_index=True)
    df["CANTIDAD"] = np.maximum(1, np.round(df["PALLETS_EQUIV"] * 60)).astype(int)
    df["PALLETS_EQUIV"] = df["CANTIDAD"] / 60
    df["UNIDAD"] = "CJ"
    df["CATEGORIA"] = np.array(CATEGORIAS)[df["PRODUCTO"] % len(CATEGORIAS)]
    df["VALOR LINEA"] = df["CANTIDAD"] * 8000
    tipo = np.where(df["_ancla"], 0, np.where(df["_perif"], 1, 2))
    df["CAMION"] = [f"E{t}{rng.integers(1, 4)}" for t in tipo]
    df["VIAJE"] = rng.integers(1, 3, len(df))
    df["TIPO_FLUJO"] = np.where(rng.random(len(df)) < 0.03, "DEVOLUCION", "DEMANDA")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    return df[list(COLUMNAS)].sort_values("FECHA").reset_index(drop=True)


if __name__ == "__main__":
    destino = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample/ventas_muestra.parquet")
    destino.parent.mkdir(parents=True, exist_ok=True)
    m = generar()
    m.to_parquet(destino, index=False)
    print(f"Muestra sintética: {len(m):,} filas -> {destino}")
