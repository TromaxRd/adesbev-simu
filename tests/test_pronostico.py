import logging
from datetime import date

import numpy as np
import pandas as pd
import pytest

from andesbev_sim.demanda import pronostico as pr

LOG = logging.LoggerAdapter(logging.getLogger("test"), {})


def _serie_sintetica(ruido: float = 0.0, semilla: int = 0) -> pd.DataFrame:
    """3 años, 2 segmentos, estacionalidad conocida: enero +30 %, lunes -20 %."""
    rng = np.random.default_rng(semilla)
    fechas = pr.calendario(date(2023, 1, 2), 3 * 365, [])
    mes = np.where(fechas.month == 1, 1.3, 1.0)
    dow = np.where(fechas.dayofweek == 0, 0.8, 1.0)
    base = np.c_[100 * mes * dow, 10 * mes * dow]
    return pd.DataFrame(
        base * rng.lognormal(0, ruido, base.shape) if ruido else base,
        index=fechas,
        columns=["ANCLA-A-X", "CAPILAR-C-Z"],
    )


def test_calendario_sin_domingos_ni_feriados():
    f = pr.calendario(date(2026, 9, 14), 14, [date(2026, 9, 18), date(2026, 9, 19)])
    assert (f.dayofweek < 6).all()
    assert pd.Timestamp("2026-09-18") not in f and len(f) == 10


def test_indices_promedian_uno_y_recuperan_la_estacionalidad():
    m = pr.ajustar(_serie_sintetica())
    assert m.idx_mes.mean().to_numpy() == pytest.approx([1, 1])
    assert m.idx_dow.mean().to_numpy() == pytest.approx([1, 1])
    # enero / resto = 1,3 ; lunes / resto = 0,8
    assert m.idx_mes.loc[1, "ANCLA-A-X"] / m.idx_mes.loc[6, "ANCLA-A-X"] == pytest.approx(1.3)
    assert m.idx_dow.loc[0, "ANCLA-A-X"] / m.idx_dow.loc[2, "ANCLA-A-X"] == pytest.approx(0.8)


def test_sin_ruido_el_backtest_es_perfecto():
    met, _, _ = pr.backtest(_serie_sintetica(), 2025, 50, np.random.default_rng(1), LOG)
    assert met["wape_mensual"] < 1e-6 and abs(met["sesgo_total"]) < 1e-6
    # el modelo sin estacionalidad sí se equivoca
    assert met["wape_diario_sin_estacionalidad"] > 0.05


def test_simulacion_conserva_el_promedio():
    serie = _serie_sintetica(ruido=0.2)
    m = pr.ajustar(serie)
    fechas = pr.calendario(date(2026, 1, 1), 365, [])
    nivel = m.nivel(2025, 0.0, 1)
    sim = pr.simular(m, fechas, nivel, 300, np.random.default_rng(2))
    assert sim.shape == (300, len(fechas), 2)
    esp = pr.esperado(m, fechas, nivel).to_numpy().sum()
    assert sim.sum(axis=(1, 2)).mean() == pytest.approx(esp, rel=0.02)
    assert (sim >= 0).all()


def test_reparto_suma_uno_por_canal_y_mes():
    lin = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2024-01-02"] * 3 + ["2024-02-05"] * 2),
            "ID_CLIENTE": [1, 2, 3, 1, 3],
            "CANAL": ["CAPILAR"] * 5,
            "ZONA": ["NUCLEO", "NUCLEO", "PERIFERIA", "NUCLEO", "PERIFERIA"],
            "SECTOR": ["N1", "N2", "P1", "N1", "P1"],
            "PALLETS": [1.0, 2.0, 1.0, 3.0, 1.0],
        }
    )
    r = pr.reparto_sectores(lin)
    assert r.groupby(["CANAL", "MES"])["participacion"].sum().to_numpy() == pytest.approx([1, 1])
    assert r.loc[(r.MES == 2) & (r.SECTOR == "N1"), "participacion"].item() == pytest.approx(0.75)
