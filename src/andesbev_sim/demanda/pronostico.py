"""Pronóstico estacional empírico por segmento (canal × ABC × XYZ).

Modelo multiplicativo para cada segmento s y día de despacho t:

    D_s(t) = L_s · Imes_s(mes de t) · Idow_s(día de semana de t) · e(t)

- L_s      nivel desestacionalizado del último año observado (pallets/día de
           un mes y un día de semana "promedio").
- Imes_s   índice mensual: cuánto sobre o bajo el promedio anual está cada mes
           (promedio de los 3 años; 1,20 = mes 20 % sobre el promedio).
- Idow_s   índice de día de semana (lunes a sábado).
- e(t)     residuo: lo que la estacionalidad no explica. NO se ajusta a una
           distribución teórica: se guardan los residuos históricos y la
           simulación los remuestrea (bootstrap) en dos escalas: el efecto del
           mes completo (de un año histórico sorteado) y el efecto del día (de un
           día del mismo mes y día de semana). Se toma el vector completo de los
           18 segmentos, para conservar la correlación entre ellos (un día malo
           suele ser malo para todos).

El reparto a sectores usa la participación histórica de cada sector dentro de
su canal, mes a mes (Tongoy casi duplica su peso en enero-febrero), y las
visitas se obtienen dividiendo por los pallets por visita de ese sector-mes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

DIAS_DESPACHO = 6  # lunes (0) a sábado (5)
MIN_CANDIDATOS = 3  # días históricos mínimos para sortear un residuo


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #
def etiquetar(lineas: pd.DataFrame, clientes: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """Agrega CANAL, ZONA, SECTOR y SEGMENTO (p. ej. 'ANCLA-A-X') a cada línea."""
    lin = lineas[["FECHA", "ID_CLIENTE", "SKU", "PALLETS"]].merge(
        clientes[["ID_CLIENTE", "CANAL", "ZONA", "SECTOR"]], on="ID_CLIENTE"
    )
    lin = lin.merge(skus[["SKU", "ABC", "XYZ"]], on="SKU", how="left")
    lin[["ABC", "XYZ"]] = lin[["ABC", "XYZ"]].fillna({"ABC": "C", "XYZ": "Z"})
    lin["SEGMENTO"] = lin["CANAL"] + "-" + lin["ABC"] + "-" + lin["XYZ"]
    # Los pocos domingos con despacho se suman al sábado anterior
    domingo = lin["FECHA"].dt.dayofweek == 6
    lin.loc[domingo, "FECHA"] -= pd.Timedelta(days=1)
    return lin


def serie_diaria(lin: pd.DataFrame) -> pd.DataFrame:
    """Matriz días de despacho × segmentos (pallets). Días sin venta de un segmento = 0."""
    return lin.pivot_table(
        index="FECHA", columns="SEGMENTO", values="PALLETS", aggfunc="sum", fill_value=0.0
    ).sort_index()


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@dataclass
class ModeloEstacional:
    nivel_anual: pd.DataFrame  # año × segmento
    idx_mes: pd.DataFrame  # 1..12 × segmento
    idx_dow: pd.DataFrame  # 0..5 × segmento
    residuos: pd.DataFrame  # días históricos × segmento (+ columna MES)

    @property
    def segmentos(self) -> list[str]:
        return list(self.idx_mes.columns)

    def nivel(self, anio_base: int, crecimiento: float, anios_adelante: int) -> pd.Series:
        return self.nivel_anual.loc[anio_base] * (1 + crecimiento) ** anios_adelante


def ajustar(serie: pd.DataFrame, iteraciones: int = 20) -> ModeloEstacional:
    """Descomposición multiplicativa por ajuste alternado (backfitting).

    Se estiman por turnos el nivel de cada año, el índice mensual y el de día de
    semana, cada uno con los otros dos ya descontados, hasta que se estabilizan.
    Así el nivel queda "desestacionalizado": no depende de cuántos lunes o
    cuántos días de enero tuvo ese año, y se puede proyectar a otro calendario.
    """
    anio = serie.index.year
    mes = serie.index.month
    dow = serie.index.dayofweek
    y = serie.to_numpy()
    segs = serie.columns
    idx_mes = pd.DataFrame(1.0, index=range(1, 13), columns=segs)
    idx_dow = pd.DataFrame(1.0, index=range(DIAS_DESPACHO), columns=segs)

    for _ in range(iteraciones):
        f_mes = idx_mes.reindex(mes).to_numpy()
        f_dow = idx_dow.reindex(dow).to_numpy()
        nivel_anual = pd.DataFrame(y / (f_mes * f_dow), index=serie.index).groupby(anio).mean()
        nivel_anual.columns = segs
        f_niv = nivel_anual.reindex(anio).to_numpy()

        rel = pd.DataFrame(y / (f_niv * f_dow), index=serie.index, columns=segs)
        # promedio de cada año-mes y luego entre años (cada año pesa lo mismo)
        idx_mes = rel.groupby([anio, mes]).mean().groupby(level=1).mean()
        idx_mes = idx_mes.reindex(range(1, 13)).fillna(1.0)
        idx_mes = idx_mes / idx_mes.mean()

        f_mes = idx_mes.reindex(mes).to_numpy()
        rel = pd.DataFrame(y / (f_niv * f_mes), index=serie.index, columns=segs)
        idx_dow = rel.groupby(dow).mean().reindex(range(DIAS_DESPACHO)).fillna(1.0)
        idx_dow = idx_dow / idx_dow.mean()

    esperado_h = (
        nivel_anual.reindex(anio).to_numpy()
        * idx_mes.reindex(mes).to_numpy()
        * idx_dow.reindex(dow).to_numpy()
    )
    residuos = pd.DataFrame(
        np.where(esperado_h > 0, y / np.where(esperado_h > 0, esperado_h, 1), 1.0),
        index=serie.index,
        columns=segs,
    )
    residuos["MES"] = mes
    return ModeloEstacional(nivel_anual, idx_mes, idx_dow, residuos)


def esperado(modelo: ModeloEstacional, fechas: pd.DatetimeIndex, nivel: pd.Series) -> pd.DataFrame:
    """Pronóstico puntual (sin ruido) para cada fecha y segmento."""
    segs = modelo.segmentos
    v = (
        nivel[segs].to_numpy()
        * modelo.idx_mes.loc[fechas.month, segs].to_numpy()
        * modelo.idx_dow.loc[fechas.dayofweek, segs].to_numpy()
    )
    return pd.DataFrame(v, index=fechas, columns=segs)


def simular(
    modelo: ModeloEstacional,
    fechas: pd.DatetimeIndex,
    nivel: pd.Series,
    replicas: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Demanda estocástica [réplica, día, segmento] por bootstrap de residuos.

    El residuo histórico de cada día se separa en dos partes y cada una se
    remuestrea por su lado, para reproducir la variabilidad real a ambas escalas:
      1. Efecto del mes: cuánto se desvió ese mes completo de lo esperado
         (promedio de sus residuos). Por réplica y mes se sortea el de un año
         histórico, así un mes "fuerte" o "flojo" puede repetirse.
      2. Efecto del día: el residuo dividido por el efecto de su mes. Se sortea
         entre los días del mismo mes y mismo día de semana de todos los años,
         tomando el vector completo de los 18 segmentos.
    """
    base = esperado(modelo, fechas, nivel).to_numpy()
    res = modelo.residuos
    segs = modelo.segmentos
    anio_h = res.index.year.to_numpy()
    mes_h, dow_h = res["MES"].to_numpy(), res.index.dayofweek.to_numpy()
    efecto_mes = res[segs].groupby([anio_h, mes_h]).transform("mean").to_numpy()
    efecto_mes = np.where(efecto_mes > 0, efecto_mes, 1.0)
    efecto_dia = res[segs].to_numpy() / efecto_mes
    anios = np.unique(anio_h)
    tabla_mes = {
        (a, m): efecto_mes[np.flatnonzero((anio_h == a) & (mes_h == m))[0]]
        for a in anios
        for m in range(1, 13)
        if ((anio_h == a) & (mes_h == m)).any()
    }
    anio_rm = rng.choice(anios, size=(replicas, 13))

    cache: dict[tuple[int, int], np.ndarray] = {}
    salida = np.empty((replicas, len(fechas), len(segs)))
    for j, f in enumerate(fechas):
        clave = (f.month, f.dayofweek)
        if clave not in cache:
            c = np.flatnonzero((mes_h == f.month) & (dow_h == f.dayofweek))
            cache[clave] = c if len(c) >= MIN_CANDIDATOS else np.flatnonzero(mes_h == f.month)
        dias = rng.choice(cache[clave], size=replicas)
        meses = np.stack(
            [tabla_mes.get((a, f.month), np.ones(len(segs))) for a in anio_rm[:, f.month]]
        )
        salida[:, j, :] = base[j] * meses * efecto_dia[dias]
    return salida


# --------------------------------------------------------------------------- #
# Reparto a sectores
# --------------------------------------------------------------------------- #
def reparto_sectores(lin: pd.DataFrame) -> pd.DataFrame:
    """Participación de cada sector en su canal, por mes, y pallets por visita."""
    lin = lin.assign(MES=lin["FECHA"].dt.month)
    pal = lin.groupby(["CANAL", "MES", "ZONA", "SECTOR"])["PALLETS"].sum()
    visitas = (
        lin.drop_duplicates(["ID_CLIENTE", "FECHA"])
        .groupby(["CANAL", "MES", "ZONA", "SECTOR"])
        .size()
    )
    r = pd.DataFrame({"pallets": pal, "visitas": visitas}).reset_index()
    r["participacion"] = r["pallets"] / r.groupby(["CANAL", "MES"])["pallets"].transform("sum")
    r["pallets_por_visita"] = r["pallets"] / r["visitas"]
    return r


# --------------------------------------------------------------------------- #
# Calendario y backtest
# --------------------------------------------------------------------------- #
def calendario(inicio: date, dias: int, feriados: list[date]) -> pd.DatetimeIndex:
    """Días de despacho: lunes a sábado, sin feriados, dentro del horizonte."""
    rango = pd.date_range(pd.Timestamp(inicio), periods=dias, freq="D")
    habiles = rango[rango.dayofweek < DIAS_DESPACHO]
    return habiles.difference(pd.DatetimeIndex(pd.to_datetime(feriados)))


def wape(real: np.ndarray, pron: np.ndarray) -> float:
    """Error absoluto ponderado: suma |real - pronóstico| / suma real."""
    return float(np.abs(real - pron).sum() / np.abs(real).sum())


def backtest(
    serie: pd.DataFrame,
    anio_prueba: int,
    replicas: int,
    rng: np.random.Generator,
    log: logging.LoggerAdapter,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Ajusta con los años anteriores y pronostica ``anio_prueba`` con su calendario real."""
    entren = serie[serie.index.year < anio_prueba]
    prueba = serie[serie.index.year == anio_prueba]
    m = ajustar(entren)
    ultimo = int(entren.index.year.max())
    nivel = m.nivel_anual.loc[ultimo]
    pron = esperado(m, prueba.index, nivel)
    sim = simular(m, prueba.index, nivel, replicas, rng).sum(axis=2)  # total por día

    real_t, pron_t = prueba.sum(axis=1), pron.sum(axis=1)
    plano = pd.Series(nivel.sum(), index=prueba.index)  # referencia sin estacionalidad
    p5, p95 = np.percentile(sim, 5, axis=0), np.percentile(sim, 95, axis=0)
    diario = pd.DataFrame(
        {"real": real_t, "pronostico": pron_t, "p5": p5, "p95": p95, "sin_estacionalidad": plano}
    )
    meses = prueba.index.month
    mensual = diario[["real", "pronostico", "sin_estacionalidad"]].groupby(meses).sum()
    # La banda mensual se calcula sobre los totales mensuales de cada réplica
    sim_mes = np.stack([sim[:, meses == m].sum(axis=1) for m in mensual.index], axis=1)
    mensual["p5"] = np.percentile(sim_mes, 5, axis=0)
    mensual["p95"] = np.percentile(sim_mes, 95, axis=0)
    mensual.index.name = "MES"

    por_seg = pd.DataFrame(
        {
            "pct_volumen": prueba.sum() / prueba.to_numpy().sum(),
            "wape_mensual": [
                wape(
                    prueba[s].groupby(prueba.index.month).sum().to_numpy(),
                    pron[s].groupby(pron.index.month).sum().to_numpy(),
                )
                for s in prueba.columns
            ],
            "sesgo": pron.sum() / prueba.sum() - 1,
        }
    ).sort_values("pct_volumen", ascending=False)

    metricas = {
        "anio_prueba": anio_prueba,
        "wape_diario": wape(real_t.to_numpy(), pron_t.to_numpy()),
        "wape_diario_sin_estacionalidad": wape(real_t.to_numpy(), plano.to_numpy()),
        "wape_mensual": wape(mensual["real"].to_numpy(), mensual["pronostico"].to_numpy()),
        "wape_mensual_sin_estacionalidad": wape(
            mensual["real"].to_numpy(), mensual["sin_estacionalidad"].to_numpy()
        ),
        "sesgo_total": float(pron_t.sum() / real_t.sum() - 1),
        "cobertura_intervalo_90": float(((real_t >= p5) & (real_t <= p95)).mean()),
        "por_segmento": por_seg.round(4).reset_index(names="SEGMENTO").to_dict("records"),
    }
    log.info(
        "Backtest %d: WAPE diario %.1f%% (sin estacionalidad %.1f%%), mensual %.1f%%, "
        "sesgo %+.1f%%, cobertura IC90 %.0f%%",
        anio_prueba,
        100 * metricas["wape_diario"],
        100 * metricas["wape_diario_sin_estacionalidad"],
        100 * metricas["wape_mensual"],
        100 * metricas["sesgo_total"],
        100 * metricas["cobertura_intervalo_90"],
        extra={k: v for k, v in metricas.items() if k != "por_segmento"},
    )
    return metricas, diario, mensual


# --------------------------------------------------------------------------- #
# Persistencia: lo que el motor de simulación lee
# --------------------------------------------------------------------------- #
@dataclass
class ParametrosDemanda:
    """Todo lo que el motor necesita para generar demanda estocástica."""

    modelo: ModeloEstacional
    nivel: pd.Series  # pallets/día por segmento al inicio del horizonte
    fechas: pd.DatetimeIndex  # días de despacho del horizonte
    reparto: pd.DataFrame  # CANAL, MES, ZONA, SECTOR, participacion, pallets_por_visita


def guardar(p: ParametrosDemanda, carpeta) -> None:
    from pathlib import Path

    c = Path(carpeta)
    c.mkdir(parents=True, exist_ok=True)
    p.modelo.nivel_anual.rename(index=str).to_parquet(c / "nivel_anual.parquet")
    p.modelo.idx_mes.rename(index=str).to_parquet(c / "indice_mensual.parquet")
    p.modelo.idx_dow.rename(index=str).to_parquet(c / "indice_dia_semana.parquet")
    p.modelo.residuos.to_parquet(c / "residuos.parquet")
    p.nivel.rename("nivel").to_frame().to_parquet(c / "nivel_horizonte.parquet")
    pd.DataFrame({"FECHA": p.fechas}).to_parquet(c / "calendario.parquet", index=False)
    p.reparto.to_parquet(c / "reparto_sectores.parquet", index=False)


def cargar(carpeta) -> ParametrosDemanda:
    from pathlib import Path

    c = Path(carpeta)

    def _int_idx(df: pd.DataFrame) -> pd.DataFrame:
        df.index = df.index.astype(int)
        return df

    modelo = ModeloEstacional(
        nivel_anual=_int_idx(pd.read_parquet(c / "nivel_anual.parquet")),
        idx_mes=_int_idx(pd.read_parquet(c / "indice_mensual.parquet")),
        idx_dow=_int_idx(pd.read_parquet(c / "indice_dia_semana.parquet")),
        residuos=pd.read_parquet(c / "residuos.parquet"),
    )
    return ParametrosDemanda(
        modelo=modelo,
        nivel=pd.read_parquet(c / "nivel_horizonte.parquet")["nivel"],
        fechas=pd.DatetimeIndex(pd.read_parquet(c / "calendario.parquet")["FECHA"]),
        reparto=pd.read_parquet(c / "reparto_sectores.parquet"),
    )
