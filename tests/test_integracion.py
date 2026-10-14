"""Prueba de punta a punta: muestra sintética -> todas las etapas -> dashboard XHTML."""

import json
from pathlib import Path

import pytest
from lxml import etree

import main
from andesbev_sim import RAIZ_REPO
from andesbev_sim.muestra import generar


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    pq = tmp / "muestra.parquet"
    generar().to_parquet(pq, index=False)
    comun = (
        "corrida: {replicas: 3}\n"
        f"datos: {{parquet: '{pq.as_posix()}', carpeta_intermedios: '{(tmp / 'interim').as_posix()}',"
        f" carpeta_salidas: '{(tmp / 'out').as_posix()}'}}\n"
        f"logging: {{carpeta: '{(tmp / 'logs').as_posix()}', nivel_consola: WARNING}}\n"
    )
    base, dis = tmp / "base.yaml", tmp / "dis.yaml"
    base.write_text(f"hereda: '{(RAIZ_REPO / 'config/base.yaml').as_posix()}'\n" + comun, "utf-8")
    dis.write_text(
        f"hereda: '{(RAIZ_REPO / 'config/disrupcion.yaml').as_posix()}'\n" + comun, "utf-8"
    )
    codigo = main.main(["--etapa", "todo", "--escenario", str(base), "--disrupcion", str(dis)])
    assert codigo == 0
    return tmp


def test_genera_intermedios(corrida):
    interim = corrida / "interim"
    for f in ["lineas_demanda.parquet", "clientes.parquet", "skus.parquet", "flota.json"]:
        assert (interim / f).exists(), f
    assert (interim / "pronostico" / "residuos.parquet").exists()


def test_logs_separados_por_tipo_de_corrida(corrida):
    logs = corrida / "logs"
    for tipo in ["preparacion", "base", "disrupcion"]:
        assert list((logs / tipo).glob("*.log")), tipo
        assert list((logs / tipo).glob("*.json")), tipo
    registros = [
        json.loads(x)
        for x in next((logs / "disrupcion").glob("*.json")).read_text("utf-8").splitlines()
    ]
    assert any(r["nivel"] == "WARNING" and "inyectado" in r["mensaje"] for r in registros)
    assert any("replica" in r for r in registros)  # trazabilidad por réplica


def test_dashboard_xhtml_bien_formado(corrida):
    dash = list((corrida / "out" / "corridas").glob("*/dashboard_*.xhtml"))
    assert len(dash) == 1
    raiz = etree.parse(str(dash[0])).getroot()  # falla si el XML está mal formado
    assert raiz.tag == "{http://www.w3.org/1999/xhtml}html"
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    secciones = raiz.xpath("//x:section/@id", namespaces=ns)
    assert secciones == ["resumen", "flota", "estacional", "territorio", "resiliencia", "auditoria"]
    assert len(raiz.xpath("//x:canvas", namespaces=ns)) == 7


def test_disrupcion_empeora_el_servicio(corrida):
    comp = json.loads(
        next((corrida / "out" / "corridas").glob("*/comparacion.json")).read_text("utf-8")
    )
    assert comp["servicio_a_tiempo_disrupcion"] < comp["servicio_a_tiempo_base"]
    assert Path(next((corrida / "out" / "corridas").glob("*/recuperacion.csv"))).exists()
