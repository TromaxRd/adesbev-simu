from fastapi.testclient import TestClient
from lxml import etree

from andesbev_sim.web.app import app

cliente = TestClient(app)


def test_panel_es_xhtml_bien_formado():
    r = cliente.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/xhtml+xml")
    raiz = etree.fromstring(r.content)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    botones = raiz.xpath("//x:button/text()", namespaces=ns)
    assert botones == ["Ejecutar simulación", "Ver logs", "Generar dashboard"]


def test_cabeceras_de_seguridad():
    r = cliente.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_lista_escenarios():
    d = cliente.get("/api/escenarios").json()
    assert {"base.yaml", "disrupcion.yaml"} <= set(d["escenarios"])


def test_rechaza_rutas_fuera_de_config():
    r = cliente.post("/api/ejecutar", json={"escenario_base": "../../etc/passwd.yaml"})
    assert r.status_code == 400


def test_valida_rangos_del_formulario():
    r = cliente.post("/api/ejecutar", json={"replicas": 10_000})
    assert r.status_code == 422


def test_dashboard_con_identificador_invalido():
    assert cliente.get("/dashboard/..%2F..%2Fsecreto").status_code in (400, 404)
    assert cliente.get("/dashboard/no_es_un_run").status_code == 400


def test_trabajo_inexistente():
    assert cliente.get("/api/estado/nada").status_code == 404
