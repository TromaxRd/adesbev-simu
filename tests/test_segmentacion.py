import numpy as np
import pandas as pd

from andesbev_sim.demanda.segmentacion import canal_por_segmento, clasificar_abc, clasificar_xyz


def test_abc_suma_100_y_una_clase_por_sku():
    vol = pd.Series({"s1": 50.0, "s2": 30.0, "s3": 10.0, "s4": 6.0, "s5": 4.0})
    r = clasificar_abc(vol, (0.80, 0.95))
    assert np.isclose(r["participacion"].sum(), 1.0)
    assert r.index.is_unique
    # s1 (0-50 %) y s2 (50-80 %) son A; s3 empieza en 80 % -> B; s4 empieza en 90 % -> B
    assert r["ABC"].to_dict() == {"s1": "A", "s2": "A", "s3": "B", "s4": "B", "s5": "C"}


def test_xyz_estable_vs_erratico():
    semanas = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    estable = pd.DataFrame({"SKU": "e", "SEMANA": semanas, "PALLETS": 10.0})
    erratico = pd.DataFrame({"SKU": "z", "SEMANA": semanas[::3], "PALLETS": 40.0})
    r = clasificar_xyz(pd.concat([estable, erratico]), (0.5, 1.0))
    assert r.loc["e", "XYZ"] == "X" and r.loc["e", "CV"] == 0
    assert r.loc["z", "XYZ"] == "Z"  # vende 1 de cada 3 semanas: CV > 1


def test_xyz_no_castiga_productos_nuevos():
    # Producto que existe solo las últimas 4 semanas, con venta pareja
    semanas = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    viejo = pd.DataFrame({"SKU": "v", "SEMANA": semanas, "PALLETS": 5.0})
    nuevo = pd.DataFrame({"SKU": "n", "SEMANA": semanas[-4:], "PALLETS": 5.0})
    r = clasificar_xyz(pd.concat([viejo, nuevo]), (0.5, 1.0))
    assert r.loc["n", "XYZ"] == "X" and r.loc["n", "semanas_vigente"] == 4


def test_canal_por_tamano_de_visita():
    lineas = pd.DataFrame(
        {
            "SEGMENTO": ["SUPER", "SUPER", "ALMACEN", "ALMACEN"],
            "ID_CLIENTE": [1, 1, 2, 3],
            "FECHA": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-02"]),
            "PALLETS": [3.0, 5.0, 0.1, 0.2],
        }
    )
    r = canal_por_segmento(lineas, umbral=0.6)
    assert r.loc["SUPER", "CANAL"] == "ANCLA"
    assert r.loc["ALMACEN", "CANAL"] == "CAPILAR"
