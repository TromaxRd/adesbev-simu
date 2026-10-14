import logging
from datetime import date

import numpy as np
import pandas as pd
import pytest

from andesbev_sim.config import Disrupcion
from andesbev_sim.simulacion import disrupciones, motor
from andesbev_sim.simulacion.red import construir_red

LOG = logging.LoggerAdapter(logging.getLogger("test"), {})
SEGS = ["ANCLA-A-X", "CAPILAR-A-X", "CAPILAR-C-Z"]


def _red():
    filas = []
    for mes in range(1, 13):
        filas += [
            ("ANCLA", mes, "NUCLEO", "N1", 1.0, 2.0),
            ("CAPILAR", mes, "NUCLEO", "N1", 0.7, 0.1),
            ("CAPILAR", mes, "PERIFERIA", "P1", 0.3, 0.1),
        ]
    rep = pd.DataFrame(
        filas, columns=["CANAL", "MES", "ZONA", "SECTOR", "participacion", "pallets_por_visita"]
    )
    return construir_red(rep, SEGS)


def _cap(ded=100.0, nuc=100.0, per=100.0, paradas=1e6):
    return {
        "DEDICADA": motor.CapacidadPool(1, ded, paradas),
        "CAPILAR_NUCLEO": motor.CapacidadPool(1, nuc, paradas),
        "CAPILAR_PERIFERIA": motor.CapacidadPool(1, per, paradas),
    }


def _correr(red, demanda, cap, ef=None, k=2):
    rep, dias, _ = demanda.shape
    ef = ef or disrupciones.Efectos(
        np.ones((rep, dias, red.n)), np.ones((rep, dias, red.n)), np.zeros((rep, dias, len(SEGS)))
    )
    meses = np.ones(dias, dtype=int)
    return motor.simular(red, demanda, meses, cap, ef, k, {}, LOG)


def test_red_asigna_pools():
    u = _red().unidades.set_index(["CANAL", "SECTOR"])["POOL"]
    assert u[("ANCLA", "N1")] == "DEDICADA"
    assert u[("CAPILAR", "N1")] == "CAPILAR_NUCLEO"
    assert u[("CAPILAR", "P1")] == "CAPILAR_PERIFERIA"


def test_conservacion_de_flujo():
    """Demanda = despachado + venta perdida + pendiente final, réplica por réplica."""
    rng = np.random.default_rng(0)
    demanda = rng.uniform(5, 30, size=(4, 40, 3))
    r = _correr(_red(), demanda, _cap(20, 15, 5))
    entra = r.demanda.sum(axis=(1, 2))
    sale = r.despachado.sum(axis=(1, 2)) + r.perdido.sum(axis=(1, 2)) + r.pendiente[:, -1].sum(1)
    assert sale == pytest.approx(entra)


def test_capacidad_holgada_servicio_total():
    r = _correr(_red(), np.full((2, 10, 3), 10.0), _cap())
    assert r.a_tiempo.sum() == pytest.approx(r.demanda.sum())
    assert r.perdido.sum() == 0 and r.pendiente.sum() == 0


def test_restriccion_de_paradas_manda():
    # CAPILAR: 20 pallets/día a 0,1 pallets por visita = 200 visitas, pero solo 100 paradas
    cap = _cap(paradas=100.0)
    dem = np.zeros((1, 1, 3))
    dem[0, 0, 1] = 20.0
    r = _correr(_red(), dem, cap)
    nuc = r.despachado[0, 0, 1]  # unidad CAPILAR-N1: 14 pallets pedidos = 140 visitas
    assert nuc == pytest.approx(14.0 * 100 / 140)


def test_backlog_se_pierde_despues_de_k_dias():
    dem = np.zeros((1, 5, 3))
    dem[0, 0, 0] = 10.0  # un pedido ancla el día 0 y luego nada
    cap = _cap(ded=0.0)  # la flota dedicada no puede despachar
    r = _correr(_red(), dem, cap, k=2)
    perdido = r.perdido[0, :, 0]
    assert perdido[2] == pytest.approx(10.0)  # esperó 2 días -> perdido al tercer día
    assert perdido.sum() == pytest.approx(10.0)


def test_bloqueo_solo_afecta_la_zona():
    red = _red()
    fechas = pd.bdate_range("2026-03-02", periods=10)
    d = Disrupcion(
        tipo="bloqueo_vial",
        modo="determinista",
        dia_inicio=2,
        duracion_dias=3,
        zonas=["PERIFERIA"],
        capacidad_restante=0.0,
    )
    ef = disrupciones.generar([d], red, SEGS, fechas, 1, np.random.default_rng(0), LOG)
    per = disrupciones.mascara_unidades(red, ["PERIFERIA"])
    assert (ef.factor_capacidad[0, 2:5][:, per] == 0).all()
    assert (ef.factor_capacidad[0, 2:5][:, ~per] == 1).all()
    assert (ef.factor_capacidad[0, [0, 1, 5]] == 1).all()


def test_fecha_inicio_se_traduce_a_dia_de_despacho():
    red = _red()
    fechas = pd.DatetimeIndex(pd.bdate_range("2026-09-07", periods=10))
    d = Disrupcion(
        tipo="peak_anomalo",
        modo="determinista",
        fecha_inicio=date(2026, 9, 12),  # sábado
        duracion_dias=2,
        multiplicador_demanda=2.0,
    )
    ef = disrupciones.generar([d], red, SEGS, fechas, 1, np.random.default_rng(0), LOG)
    assert ef.eventos[0].inicio == 5  # lunes 14 (bdate_range no incluye sábados)
    assert (ef.factor_demanda[0, 5:7] == 2.0).all()


def test_estocastico_respeta_probabilidad():
    red = _red()
    fechas = pd.bdate_range("2026-01-01", periods=300)
    d = Disrupcion(
        tipo="quiebre_stock",
        modo="estocastico",
        probabilidad_diaria=0.02,
        duracion_dias=1,
        fraccion_afectada=0.5,
    )
    ef = disrupciones.generar([d], red, SEGS, fechas, 50, np.random.default_rng(3), LOG)
    eventos_por_replica = len(ef.eventos) / 50
    assert 3 < eventos_por_replica < 9  # esperado: 300 × 0,02 = 6
    assert ef.fraccion_diferida.max() == 0.5
