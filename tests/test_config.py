import pytest
from pydantic import ValidationError

from andesbev_sim.config import cargar_config


def test_base_carga():
    cfg = cargar_config("config/base.yaml")
    assert cfg.corrida.nombre == "base"
    assert cfg.disrupciones == []


def test_disrupcion_hereda_de_base():
    cfg = cargar_config("config/disrupcion.yaml")
    assert cfg.corrida.nombre == "disrupcion"
    assert cfg.corrida.semilla == 42  # viene de base.yaml
    assert cfg.disrupciones[0].tipo == "bloqueo_vial"


def test_override_de_replicas():
    assert cargar_config("config/base.yaml", replicas=5).corrida.replicas == 5


def test_campo_mal_escrito_falla(tmp_path):
    malo = tmp_path / "malo.yaml"
    malo.write_text("corrida: {nombre: x, replicass: 3}\ndatos: {parquet: a.parquet}\n")
    with pytest.raises(ValidationError):
        cargar_config(malo)


def test_disrupcion_invalida_falla(tmp_path):
    malo = tmp_path / "malo.yaml"
    malo.write_text(
        "corrida: {nombre: x}\ndatos: {parquet: a.parquet}\n"
        "disrupciones: [{tipo: terremoto, modo: determinista}]\n"
    )
    with pytest.raises(ValidationError):
        cargar_config(malo)
