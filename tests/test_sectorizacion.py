import logging

import numpy as np
import pandas as pd
import pytest

from andesbev_sim.geo.sectorizacion import distancia_km, punto_weber, sectorizar

CD = (-29.94, -71.26)


@pytest.fixture
def log():
    return logging.LoggerAdapter(logging.getLogger("test"), {})


def _clientes():
    rng = np.random.default_rng(0)
    n = 200
    # 180 clientes a menos de ~5 km del CD, 15 a ~50 km, 5 sin coordenadas
    lat = np.r_[
        CD[0] + rng.normal(0, 0.02, 180), CD[0] + 0.45 + rng.normal(0, 0.01, 15), [np.nan] * 5
    ]
    lon = np.r_[CD[1] + rng.normal(0, 0.02, 180), CD[1] + rng.normal(0, 0.01, 15), [np.nan] * 5]
    return pd.DataFrame(
        {"ID_CLIENTE": range(n), "LAT": lat, "LON": lon, "visitas": rng.integers(1, 50, n)}
    )


def test_distancia_un_grado_de_latitud():
    assert distancia_km([CD[0] + 1], [CD[1]], *CD)[0] == pytest.approx(111.32, rel=1e-3)


def test_weber_de_puntos_simetricos_es_el_centro():
    lat = np.array([-30.0, -30.0, -29.9, -29.9])
    lon = np.array([-71.3, -71.2, -71.3, -71.2])
    w = punto_weber(lat, lon, np.ones(4))
    assert w == pytest.approx((-29.95, -71.25), abs=1e-4)


def test_weber_se_acerca_al_cliente_pesado():
    lat, lon = np.array([-30.0, -29.9]), np.array([-71.25, -71.25])
    w = punto_weber(lat, lon, np.array([1.0, 10.0]))
    assert w[0] > -29.95  # más cerca de -29.9, el cliente con peso 10


def test_cada_cliente_en_un_solo_sector_y_zona_correcta(log):
    c = sectorizar(
        _clientes(),
        CD,
        radio_nucleo_km=15,
        k_nucleo=4,
        k_periferia=1,
        ponderar_por="visitas",
        semilla=1,
        log=log,
    )
    assert c["SECTOR"].notna().all() and len(c) == 200
    assert (c["ZONA"] == "NUCLEO").sum() == 180
    assert (c["ZONA"] == "PERIFERIA").sum() == 15
    assert set(c.loc[c["ZONA"] == "SIN_GEO", "SECTOR"]) == {"SIN_GEO"}
    assert set(c.loc[c["ZONA"] == "NUCLEO", "SECTOR"]) == {"N1", "N2", "N3", "N4"}
    # N1 es el sector con más visitas
    carga = c[c["ZONA"] == "NUCLEO"].groupby("SECTOR")["visitas"].sum()
    assert carga.idxmax() == "N1"
