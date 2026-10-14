import json

import pandas as pd
import pytest

from andesbev_sim.ingesta.loader import COLUMNAS, DatosInvalidosError, ejecutar_ingesta
from andesbev_sim.logging_setup import cerrar_logging, configurar_logging


def _parquet_minimo(ruta):
    """Parquet sintético con la estructura del v4: 3 líneas de demanda y 1 devolución."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "CAMION": ["E1", "E1", "E2", "E2"],
            "VIAJE": [1, 1, 2, 2],
            "ID-CLIENTE": [10, 11, 10, 10],
            "PRODUCTO": [100, 100, 101, 100],
            "CATEGORIA": ["GASEOSAS", "GASEOSAS", "CERVEZAS", "GASEOSAS"],
            "UNIDAD": ["CJ", "CJ", "CU", "CJ"],
            "CANTIDAD": [10, 50, 0, 1],  # la tercera línea (cantidad 0) debe descartarse
            "PALLETS_EQUIV": [0.2, 1.0, 0.0, 0.02],
            "VALOR LINEA": [1000, 5000, 0, 100],
            "TIPO_FLUJO": ["DEMANDA", "DEMANDA", "DEMANDA", "DEVOLUCION"],
            "SEGMENTO": ["ALMACEN", "SUPERMERCADOS", "ALMACEN", "ALMACEN"],
            "COMUNA_NOMBRE": ["LA SERENA", "COQUIMBO", "LA SERENA", "LA SERENA"],
            "LATITUD": [-29.9, -29.95, -29.9, None],  # la devolución no tiene coordenadas
            "LONGITUD": [-71.2, -71.3, -71.2, None],
        }
    )
    assert list(df.columns) == list(COLUMNAS)
    df.to_parquet(ruta, index=False)


@pytest.fixture
def log(tmp_path):
    lg, _ = configurar_logging(tmp_path / "logs", "test", "run_test")
    yield lg
    cerrar_logging()


def test_ingesta_separa_flujos_y_descarta(tmp_path, log):
    pq = tmp_path / "v4.parquet"
    _parquet_minimo(pq)
    r = ejecutar_ingesta(pq, tmp_path / "interim", log)

    assert r.filas_leidas == 4
    assert r.descartes["cantidad_cero_o_negativa"] == 1
    assert r.demanda["lineas"] == 2 and r.devolucion["lineas"] == 1
    assert (tmp_path / "interim" / "lineas_demanda.parquet").exists()
    dev = pd.read_parquet(tmp_path / "interim" / "lineas_devolucion.parquet")
    assert dev["CANTIDAD"].sum() == 1


def test_log_json_es_valido(tmp_path, log):
    pq = tmp_path / "v4.parquet"
    _parquet_minimo(pq)
    ejecutar_ingesta(pq, tmp_path / "interim", log)
    lineas = (tmp_path / "logs" / "test" / "run_test.json").read_text(encoding="utf-8").splitlines()
    registros = [json.loads(x) for x in lineas]
    assert {"timestamp", "nivel", "mensaje", "run_id"} <= set(registros[0])
    assert any(r["nivel"] == "WARNING" for r in registros)  # descarte + sin geo


def test_falta_columna_detiene(tmp_path, log):
    pq = tmp_path / "malo.parquet"
    pd.DataFrame({"FECHA": [pd.Timestamp("2024-01-01")]}).to_parquet(pq)
    with pytest.raises(DatosInvalidosError):
        ejecutar_ingesta(pq, tmp_path / "interim", log)
