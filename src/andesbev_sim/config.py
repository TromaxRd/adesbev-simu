"""Carga y validación de la configuración de escenarios (archivos YAML).

Un escenario puede heredar de otro con la clave ``hereda``: primero se carga el
archivo padre y luego se le superponen los valores del hijo. Después de
combinar, todo se valida con pydantic; un campo mal escrito o con un tipo
incorrecto detiene la carga con un mensaje claro, antes de simular nada.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from andesbev_sim import RAIZ_REPO


class _Estricto(BaseModel):
    """Base común: rechaza claves desconocidas (atrapa errores de tipeo)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Corrida(_Estricto):
    nombre: str
    semilla: int = 42
    replicas: int = Field(30, ge=1, le=1000)
    horizonte_dias: int = Field(365, ge=1, le=1095)


class Datos(_Estricto):
    parquet: Path
    carpeta_intermedios: Path = Path("data/interim")
    carpeta_salidas: Path = Path("outputs")
    unidad_fisica: Literal["pallets", "cajas"] = "pallets"


class Logging(_Estricto):
    carpeta: Path = Path("logs")
    nivel_consola: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    nivel_archivo: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Disrupcion(_Estricto):
    tipo: Literal["bloqueo_vial", "quiebre_stock", "peak_anomalo"]
    modo: Literal["determinista", "estocastico"]
    nombre: str | None = None
    # Determinista: fecha_inicio (primer día de despacho desde esa fecha) o
    # dia_inicio (índice del día de despacho en el horizonte, desde 0)
    fecha_inicio: date | None = None
    dia_inicio: int | None = Field(None, ge=0)
    duracion_dias: int | list[int] = 1
    probabilidad_diaria: float | None = Field(None, ge=0, le=1)
    zonas: list[str] | None = None
    clases_abc: list[str] | None = None
    capacidad_restante: float | None = Field(None, ge=0, le=1)
    multiplicador_demanda: float | None = Field(None, gt=0)
    # Quiebre de stock: fracción de la demanda de esas clases que no se puede despachar
    fraccion_afectada: float = Field(1.0, gt=0, le=1)

    @model_validator(mode="after")
    def _coherente(self) -> Disrupcion:
        if self.modo == "determinista" and self.fecha_inicio is None and self.dia_inicio is None:
            raise ValueError("una disrupción determinista necesita fecha_inicio o dia_inicio")
        if self.modo == "estocastico" and self.probabilidad_diaria is None:
            raise ValueError("una disrupción estocástica necesita probabilidad_diaria")
        if self.tipo == "bloqueo_vial" and self.capacidad_restante is None:
            raise ValueError("bloqueo_vial necesita capacidad_restante")
        if self.tipo == "peak_anomalo" and self.multiplicador_demanda is None:
            raise ValueError("peak_anomalo necesita multiplicador_demanda")
        return self

    @field_validator("duracion_dias")
    @classmethod
    def _duracion_valida(cls, v: int | list[int]) -> int | list[int]:
        if isinstance(v, list) and (len(v) != 2 or v[0] > v[1] or v[0] < 1):
            raise ValueError("duracion_dias como rango debe ser [min, max] con 1 <= min <= max")
        if isinstance(v, int) and v < 1:
            raise ValueError("duracion_dias debe ser >= 1")
        return v


class Segmentacion(_Estricto):
    # % acumulado de volumen que cierra la clase A y la clase B (el resto es C)
    cortes_abc: tuple[float, float] = (0.80, 0.95)
    # Coeficiente de variación semanal que separa X | Y | Z
    cortes_xyz_cv: tuple[float, float] = (0.5, 1.0)
    # Segmento comercial = ANCLA si su mediana de pallets por visita supera este umbral
    umbral_ancla_pallets_visita: float = Field(0.6, gt=0)

    @field_validator("cortes_abc", "cortes_xyz_cv")
    @classmethod
    def _crecientes(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not 0 < v[0] < v[1]:
            raise ValueError("los cortes deben ser positivos y crecientes")
        return v


class CentroDistribucion(_Estricto):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    # True: se recalcula como punto de Weber (Weiszfeld) y se compara con lat/lon
    recalcular_weber: bool = True


class Territorio(_Estricto):
    centro_distribucion: CentroDistribucion
    radio_nucleo_km: float = Field(15, gt=0)
    sectores_nucleo: int = Field(8, ge=1, le=30)
    sectores_periferia: int = Field(5, ge=1, le=15)
    # Peso de cada cliente al formar sectores del núcleo
    ponderar_nucleo_por: Literal["visitas", "pallets", "ninguno"] = "visitas"


class Pronostico(_Estricto):
    inicio_horizonte: date  # primer día simulado
    anio_base: int  # año cuyo nivel se proyecta (el último completo)
    crecimiento_anual: float = Field(0.0, ge=-0.5, le=0.5)
    anio_backtest: int
    replicas_backtest: int = Field(300, ge=50, le=5000)
    feriados: list[date] = []


class CamionesPorPool(_Estricto):
    # None = usar el dimensionamiento sugerido por el paso 4
    DEDICADA: int | None = Field(None, ge=0)
    CAPILAR_NUCLEO: int | None = Field(None, ge=0)
    CAPILAR_PERIFERIA: int | None = Field(None, ge=0)


class Flota(_Estricto):
    percentil_capacidad: float = Field(0.95, gt=0.5, lt=1)
    umbral_ruta_dedicada: float = Field(0.8, gt=0.5, le=1)
    umbral_ruta_periferia: float = Field(0.5, gt=0, le=1)
    percentil_dimensionamiento: float = Field(0.90, gt=0.5, le=1)
    camiones: CamionesPorPool = CamionesPorPool()
    # Multiplica la capacidad de todos los camiones (sensibilidad; 1 = calibrada)
    factor_capacidad: float = Field(1.0, gt=0, le=3)


class Simulacion(_Estricto):
    # Un pedido atrasado más de estos días se registra como venta perdida
    dias_backlog_max: int = Field(2, ge=0, le=10)
    # Recuperado = backlog vuelve a <= base + tolerancia (en pallets)
    tolerancia_recuperacion_pallets: float = Field(5.0, ge=0)


class Config(_Estricto):
    corrida: Corrida
    datos: Datos
    logging: Logging = Logging()
    segmentacion: Segmentacion = Segmentacion()
    territorio: Territorio
    pronostico: Pronostico
    flota: Flota = Flota()
    simulacion: Simulacion = Simulacion()
    disrupciones: list[Disrupcion] = []

    def ruta(self, p: Path) -> Path:
        """Convierte una ruta relativa del YAML en absoluta respecto del repo."""
        return p if p.is_absolute() else (RAIZ_REPO / p).resolve()

    @property
    def ruta_parquet(self) -> Path:
        """Ruta del Parquet; la variable de entorno ANDESBEV_PARQUET tiene prioridad."""
        env = os.environ.get("ANDESBEV_PARQUET")
        return Path(env).resolve() if env else self.ruta(self.datos.parquet)


def _combinar(base: dict[str, Any], hijo: dict[str, Any]) -> dict[str, Any]:
    """Combinación profunda: los diccionarios se mezclan, el resto se reemplaza."""
    resultado = dict(base)
    for clave, valor in hijo.items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _combinar(resultado[clave], valor)
        else:
            resultado[clave] = valor
    return resultado


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    with open(ruta, encoding="utf-8") as f:
        datos = yaml.safe_load(f) or {}
    padre = datos.pop("hereda", None)
    if padre:
        datos = _combinar(_leer_yaml(ruta.parent / padre), datos)
    return datos


def cargar_config(ruta: str | Path, **overrides: Any) -> Config:
    """Lee un escenario YAML (con herencia), aplica overrides y lo valida.

    Ejemplo de override desde la CLI: ``cargar_config(ruta, replicas=50)``.
    """
    ruta = Path(ruta)
    if not ruta.is_absolute() and not ruta.exists():
        ruta = RAIZ_REPO / ruta
    datos = _leer_yaml(ruta)
    corrida_ov = {k: v for k, v in overrides.items() if v is not None}
    if corrida_ov:
        datos["corrida"] = {**datos.get("corrida", {}), **corrida_ov}
    return Config.model_validate(datos)
