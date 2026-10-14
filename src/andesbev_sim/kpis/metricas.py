"""KPIs de una corrida y comparación entre escenarios.

Definiciones (todas en pallets):

- Nivel de servicio a tiempo  = despachado el mismo día del pedido / demanda.
- Fill rate final             = 1 - venta perdida / demanda (lo atrasado que se
                                alcanzó a entregar dentro del plazo sí cuenta).
- Backlog                     = pedidos pendientes al cierre del día.
- Utilización de flota        = pallets (o paradas) usados / capacidad del pool.
- Días de recuperación        = días de despacho desde que termina una disrupción
                                hasta que el backlog vuelve al nivel del escenario
                                base en la misma réplica (misma demanda).

Cada KPI se calcula por réplica y se informa como promedio con intervalo de
confianza del 95 % (± 1,96 · desviación / raíz del número de réplicas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from andesbev_sim.simulacion.motor import CapacidadPool, Resultado
from andesbev_sim.simulacion.red import POOLS, Red


def _ic(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    h = float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return {"media": m, "ic95": h}


def _bloque(res: Resultado, m: np.ndarray | None = None) -> dict:
    """KPIs para un subconjunto de unidades (m = máscara; None = toda la red)."""
    sel = (lambda a: a) if m is None else (lambda a: a[..., m])
    dem = sel(res.demanda).sum(axis=(1, 2))
    a_t = sel(res.a_tiempo).sum(axis=(1, 2))
    per = sel(res.perdido).sum(axis=(1, 2))
    pen = sel(res.pendiente).sum(axis=2)  # [réplica, día]
    dev = sel(res.devuelto).sum(axis=(1, 2))
    des = sel(res.despachado).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "demanda_pallets": _ic(dem),
            "servicio_a_tiempo": _ic(np.where(dem > 0, a_t / dem, 1.0)),
            "fill_rate": _ic(np.where(dem > 0, 1 - per / dem, 1.0)),
            "venta_perdida_pallets": _ic(per),
            "backlog_promedio_pallets": _ic(pen.mean(axis=1)),
            "backlog_maximo_pallets": _ic(pen.max(axis=1)),
            "dias_con_atraso": _ic((pen > 1.0).sum(axis=1)),
            "devolucion_pallets": _ic(dev),
            "tasa_devolucion": _ic(np.where(des > 0, dev / des, 0.0)),
        }


def kpis(
    res: Resultado,
    red: Red,
    fechas: pd.DatetimeIndex,
    capacidad: dict[str, CapacidadPool],
) -> dict:
    u = red.unidades
    meses = fechas.month.to_numpy()

    salida = {
        "global": _bloque(res),
        "por_pool": {p: _bloque(res, red.mascara_pool(p)) for p in POOLS},
        "por_zona": {
            z: _bloque(res, (u["ZONA"] == z).to_numpy()) for z in sorted(u["ZONA"].unique())
        },
        "flota": {
            p: {
                "camiones": capacidad[p].camiones,
                "capacidad_pallets_dia": capacidad[p].pallets,
                "capacidad_paradas_dia": capacidad[p].paradas,
                "uso_pallets": _ic(res.uso_pallets[:, :, j].mean(axis=1)),
                "uso_paradas": _ic(res.uso_paradas[:, :, j].mean(axis=1)),
                "dias_saturado": _ic(
                    (np.maximum(res.uso_pallets[:, :, j], res.uso_paradas[:, :, j]) >= 0.999).sum(
                        axis=1
                    )
                ),
            }
            for j, p in enumerate(POOLS)
        },
    }

    # --- Sectores (zona + sector, ambos canales)
    filas = []
    for (zona, sector), g in u.groupby(["ZONA", "SECTOR"]):
        b = _bloque(res, u.index.isin(g.index))
        filas.append(
            {
                "ZONA": zona,
                "SECTOR": sector,
                "demanda_pallets_dia": b["demanda_pallets"]["media"] / len(fechas),
                "servicio_a_tiempo": b["servicio_a_tiempo"]["media"],
                "fill_rate": b["fill_rate"]["media"],
                "backlog_promedio": b["backlog_promedio_pallets"]["media"],
                "venta_perdida": b["venta_perdida_pallets"]["media"],
            }
        )
    salida["sectores"] = pd.DataFrame(filas)

    # --- Mensual (estacionalidad): promedio entre réplicas
    filas = []
    for mes in range(1, 13):
        d = meses == mes
        if not d.any():
            continue
        fila = {
            "MES": mes,
            "dias": int(d.sum()),
            "demanda_pallets_dia": res.demanda[:, d].sum(axis=2).mean(),
            "despachado_pallets_dia": res.despachado[:, d].sum(axis=2).mean(),
            "servicio_a_tiempo": res.a_tiempo[:, d].sum() / max(res.demanda[:, d].sum(), 1e-9),
            "backlog_promedio": res.pendiente[:, d].sum(axis=2).mean(),
            "venta_perdida_pallets": res.perdido[:, d].sum(axis=(1, 2)).mean(),
        }
        for j, p in enumerate(POOLS):
            fila[f"uso_{p}"] = np.maximum(res.uso_pallets[:, d, j], res.uso_paradas[:, d, j]).mean()
        filas.append(fila)
    mensual = pd.DataFrame(filas)
    salida["mensual"] = mensual
    salida["factor_peak_valle"] = float(
        mensual["demanda_pallets_dia"].max() / mensual["demanda_pallets_dia"].min()
    )

    # --- Serie diaria (para gráficos)
    tot = lambda a: a.sum(axis=2)  # noqa: E731
    salida["diario"] = pd.DataFrame(
        {
            "FECHA": fechas,
            "demanda": tot(res.demanda).mean(axis=0),
            "despachado": tot(res.despachado).mean(axis=0),
            "pendiente": tot(res.pendiente).mean(axis=0),
            "pendiente_p95": np.percentile(tot(res.pendiente), 95, axis=0),
            "perdido": tot(res.perdido).mean(axis=0),
            "devuelto": tot(res.devuelto).mean(axis=0),
            **{
                f"uso_{p}": np.maximum(res.uso_pallets[:, :, j], res.uso_paradas[:, :, j]).mean(
                    axis=0
                )
                for j, p in enumerate(POOLS)
            },
        }
    )
    return salida


def recuperacion(
    base: Resultado, dis: Resultado, eventos: pd.DataFrame, tol: float
) -> pd.DataFrame:
    """Días de recuperación y pérdida de servicio por evento (réplicas pareadas)."""
    if eventos.empty:
        return pd.DataFrame()
    pb, pd_ = base.pendiente.sum(axis=2), dis.pendiente.sum(axis=2)  # [réplica, día]
    dias = pb.shape[1]
    filas = []
    for e in eventos.itertuples():
        r, ini, fin = e.replica, e.dia_inicio, e.dia_fin
        exceso = pd_[r, fin - 1 :] - pb[r, fin - 1 :]
        ok = np.flatnonzero(exceso <= tol)
        rec = int(ok[0]) if len(ok) else dias - fin + 1
        censurado = len(ok) == 0
        hasta = min(dias, fin + rec)
        sl = slice(ini, hasta)
        dem = base.demanda[r, sl].sum()
        s_b = base.a_tiempo[r, sl].sum() / dem if dem else 1.0
        s_d = dis.a_tiempo[r, sl].sum() / dis.demanda[r, sl].sum() if dem else 1.0
        filas.append(
            {
                "replica": r,
                "nombre": e.nombre,
                "tipo": e.tipo,
                "fecha_inicio": e.fecha_inicio,
                "duracion_dias": e.duracion_dias,
                "dias_recuperacion": rec,
                "censurado": censurado,
                "servicio_base_ventana": s_b,
                "servicio_disrupcion_ventana": s_d,
                "perdida_servicio_pp": 100 * (s_b - s_d),
                "backlog_maximo_extra": float((pd_[r, sl] - pb[r, sl]).max()),
                "venta_perdida_extra": float(dis.perdido[r, sl].sum() - base.perdido[r, sl].sum()),
            }
        )
    return pd.DataFrame(filas)


def resumen_recuperacion(tabla: pd.DataFrame) -> pd.DataFrame:
    if tabla.empty:
        return tabla
    g = tabla.groupby(["nombre", "tipo"])
    return g.agg(
        eventos=("replica", "size"),
        duracion_promedio=("duracion_dias", "mean"),
        dias_recuperacion=("dias_recuperacion", "mean"),
        dias_recuperacion_p90=("dias_recuperacion", lambda x: x.quantile(0.9)),
        perdida_servicio_pp=("perdida_servicio_pp", "mean"),
        backlog_maximo_extra=("backlog_maximo_extra", "mean"),
        venta_perdida_extra=("venta_perdida_extra", "mean"),
    ).reset_index()
