"""Orquestación de etapas. ``main.py`` (CLI) y la GUI llaman a estas funciones.

Cada etapa lee lo que dejó la anterior en ``data/interim`` y escribe:
- intermedios para la etapa siguiente (Parquet, en data/interim)
- tablas para revisar en Excel (CSV con ; y coma decimal, en outputs/tablas)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from andesbev_sim.config import Config


class EtapaPreviaFaltanteError(Exception):
    """Se pidió una etapa sin haber corrido la anterior."""


def _interim(cfg: Config) -> Path:
    return cfg.ruta(cfg.datos.carpeta_intermedios)


def _tablas(cfg: Config) -> Path:
    p = cfg.ruta(cfg.datos.carpeta_salidas) / "tablas"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _leer(cfg: Config, nombre: str, etapa_previa: str) -> pd.DataFrame:
    ruta = _interim(cfg) / nombre
    if not ruta.exists():
        raise EtapaPreviaFaltanteError(
            f"No existe {ruta.name}. Corre antes: python main.py --etapa {etapa_previa}"
        )
    return pd.read_parquet(ruta)


def guardar_tabla(df: pd.DataFrame, ruta: Path) -> None:
    """CSV listo para Excel en configuración chilena (; separador, coma decimal)."""
    df.to_csv(ruta, sep=";", decimal=",", index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# Paso 1
# --------------------------------------------------------------------------- #
def ingesta(cfg: Config, log: logging.LoggerAdapter):
    from andesbev_sim.ingesta.loader import ejecutar_ingesta

    return ejecutar_ingesta(cfg.ruta_parquet, _interim(cfg), log)


# --------------------------------------------------------------------------- #
# Paso 2
# --------------------------------------------------------------------------- #
def segmentacion(cfg: Config, log: logging.LoggerAdapter) -> dict:
    from andesbev_sim.demanda.segmentacion import (
        canal_por_segmento,
        segmentar_skus,
        tabla_clientes,
    )
    from andesbev_sim.geo.sectorizacion import punto_weber, resumen_sectores, sectorizar

    s, t = cfg.segmentacion, cfg.territorio
    lineas = _leer(cfg, "lineas_demanda.parquet", "ingesta")
    dias = int(lineas["FECHA"].nunique())

    # --- Productos: ABC / XYZ
    skus = segmentar_skus(lineas, s.cortes_abc, s.cortes_xyz_cv, log)

    # --- Clientes: canal ancla / capilar
    canales = canal_por_segmento(lineas, s.umbral_ancla_pallets_visita)
    log.info(
        "Segmentos ANCLA (> %.2f pallets/visita): %s",
        s.umbral_ancla_pallets_visita,
        list(canales.index[canales["CANAL"] == "ANCLA"]),
    )
    clientes = tabla_clientes(lineas, canales)

    # --- Territorio: verificar el CD y sectorizar
    cd = (t.centro_distribucion.lat, t.centro_distribucion.lon)
    info_cd = {"lat": cd[0], "lon": cd[1]}
    if t.centro_distribucion.recalcular_weber:
        geo = clientes.dropna(subset=["LAT", "LON"])
        w = punto_weber(geo["LAT"].to_numpy(), geo["LON"].to_numpy(), geo["PALLETS"].to_numpy())
        from andesbev_sim.geo.sectorizacion import distancia_km

        dif_m = float(distancia_km([w[0]], [w[1]], *cd)[0] * 1000)
        info_cd.update(
            weber_lat=round(w[0], 5), weber_lon=round(w[1], 5), diferencia_m=round(dif_m)
        )
        nivel = log.info if dif_m < 1000 else log.warning
        nivel("Punto de Weber recalculado: %.5f, %.5f (a %.0f m del CD configurado)", *w, dif_m)

    clientes = sectorizar(
        clientes,
        cd,
        t.radio_nucleo_km,
        t.sectores_nucleo,
        t.sectores_periferia,
        t.ponderar_nucleo_por,
        cfg.corrida.semilla,
        log,
    )
    sectores = resumen_sectores(clientes, dias)

    # --- Cruces que usará el pronóstico (paso 3)
    lin = lineas.merge(clientes[["ID_CLIENTE", "CANAL", "ZONA"]], on="ID_CLIENTE").merge(
        skus[["SKU", "ABC", "XYZ"]], on="SKU", how="left"
    )
    matriz = pd.crosstab(skus["ABC"], skus["XYZ"], values=skus["participacion"], aggfunc="sum")
    canal_abc = pd.crosstab(lin["CANAL"], lin["ABC"], values=lin["PALLETS"], aggfunc="sum")
    canal_abc = (canal_abc / canal_abc.to_numpy().sum()).round(4)

    # --- Guardar
    out = _interim(cfg)
    clientes.to_parquet(out / "clientes.parquet", index=False)
    skus.to_parquet(out / "skus.parquet", index=False)
    tb = _tablas(cfg)
    guardar_tabla(skus, tb / "paso2_skus_abc_xyz.csv")
    guardar_tabla(canales.reset_index(), tb / "paso2_canal_por_segmento.csv")
    guardar_tabla(clientes, tb / "paso2_clientes.csv")
    guardar_tabla(sectores, tb / "paso2_sectores.csv")
    guardar_tabla(matriz.fillna(0).round(4).reset_index(), tb / "paso2_matriz_abc_xyz.csv")

    resumen = {
        "dias": dias,
        "centro_distribucion": info_cd,
        "skus_por_clase": skus["CLASE"].value_counts().sort_index().to_dict(),
        "volumen_por_clase_abc": skus.groupby("ABC")["participacion"].sum().round(4).to_dict(),
        "canales": canales.reset_index().to_dict("records"),
        "zonas": clientes.groupby("ZONA")
        .agg(clientes=("ID_CLIENTE", "size"), pallets=("PALLETS", "sum"))
        .round(1)
        .to_dict("index"),
        "canal_x_abc_pct_pallets": canal_abc.to_dict("index"),
    }
    with open(out / "resumen_segmentacion.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)
    log.info("Tablas para Excel en %s", tb)
    return {
        "resumen": resumen,
        "skus": skus,
        "canales": canales,
        "sectores": sectores,
        "matriz": matriz,
        "clientes": clientes,
    }


# --------------------------------------------------------------------------- #
# Paso 3
# --------------------------------------------------------------------------- #
def pronostico(cfg: Config, log: logging.LoggerAdapter) -> dict:
    import numpy as np

    from andesbev_sim.demanda import pronostico as pr

    p = cfg.pronostico
    lineas = _leer(cfg, "lineas_demanda.parquet", "ingesta")
    clientes = _leer(cfg, "clientes.parquet", "segmentacion")
    skus = _leer(cfg, "skus.parquet", "segmentacion")
    lin = pr.etiquetar(lineas, clientes, skus)
    serie = pr.serie_diaria(lin)
    log.info("Serie diaria: %d días × %d segmentos", *serie.shape)

    # --- 1. Validación: ¿el método habría predicho bien el último año?
    rng = np.random.default_rng(cfg.corrida.semilla)
    metricas, bt_diario, bt_mensual = pr.backtest(
        serie, p.anio_backtest, p.replicas_backtest, rng, log
    )

    # --- 2. Modelo final con los 3 años y parámetros del horizonte
    modelo = pr.ajustar(serie)
    total = pr.ajustar(serie.sum(axis=1).to_frame("TOTAL"))  # solo para reportar
    anios = p.inicio_horizonte.year - p.anio_base
    nivel = modelo.nivel(p.anio_base, p.crecimiento_anual, anios)
    fechas = pr.calendario(p.inicio_horizonte, cfg.corrida.horizonte_dias, p.feriados)
    reparto = pr.reparto_sectores(lin)
    params = pr.ParametrosDemanda(modelo, nivel, fechas, reparto)
    pr.guardar(params, _interim(cfg) / "pronostico")
    log.info(
        "Horizonte: %d días de despacho desde %s | nivel %.1f pallets/día",
        len(fechas),
        fechas.min().date(),
        nivel.sum(),
    )

    # --- 3. Vista previa del horizonte: esperado y banda P5-P95 por mes
    sim = pr.simular(modelo, fechas, nivel, 200, np.random.default_rng(cfg.corrida.semilla))
    tot = sim.sum(axis=2)  # réplica × día
    meses = fechas.month
    mens = pd.DataFrame(
        {
            "dias_despacho": pd.Series(meses).value_counts().sort_index().to_numpy(),
            "pallets_esperados": pr.esperado(modelo, fechas, nivel)
            .sum(axis=1)
            .groupby(meses)
            .sum()
            .to_numpy(),
            "p5": [np.percentile(tot[:, meses == m].sum(axis=1), 5) for m in range(1, 13)],
            "p95": [np.percentile(tot[:, meses == m].sum(axis=1), 95) for m in range(1, 13)],
        },
        index=pd.Index(range(1, 13), name="MES"),
    )
    mens["pallets_dia_promedio"] = mens["pallets_esperados"] / mens["dias_despacho"]
    pico = tot.max(axis=1)

    tb = _tablas(cfg)
    idx = modelo.idx_mes.copy()
    idx.insert(0, "TOTAL", total.idx_mes["TOTAL"])
    guardar_tabla(idx.round(3).reset_index(names="MES"), tb / "paso3_indice_mensual.csv")
    guardar_tabla(
        modelo.idx_dow.assign(TOTAL=total.idx_dow["TOTAL"]).round(3).reset_index(names="DIA"),
        tb / "paso3_indice_dia_semana.csv",
    )
    guardar_tabla(bt_mensual.round(1).reset_index(), tb / "paso3_backtest_mensual.csv")
    guardar_tabla(bt_diario.round(1).reset_index(), tb / "paso3_backtest_diario.csv")
    guardar_tabla(pd.DataFrame(metricas["por_segmento"]), tb / "paso3_backtest_segmentos.csv")
    guardar_tabla(mens.round(1).reset_index(), tb / "paso3_horizonte_mensual.csv")

    resumen = {
        "backtest": {k: v for k, v in metricas.items() if k != "por_segmento"},
        "nivel_pallets_dia": round(float(nivel.sum()), 1),
        "dias_horizonte": len(fechas),
        "indice_mensual_total": total.idx_mes["TOTAL"].round(3).to_dict(),
        "indice_dia_semana_total": total.idx_dow["TOTAL"].round(3).to_dict(),
        "dia_pico_p50": round(float(np.median(pico)), 1),
        "dia_pico_p95": round(float(np.percentile(pico, 95)), 1),
    }
    with open(_interim(cfg) / "resumen_pronostico.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)
    return {
        "resumen": resumen,
        "metricas": metricas,
        "bt_mensual": bt_mensual,
        "horizonte": mens,
        "idx": idx,
    }


# --------------------------------------------------------------------------- #
# Paso 4
# --------------------------------------------------------------------------- #
def flota(cfg: Config, log: logging.LoggerAdapter) -> dict:
    from andesbev_sim.demanda import pronostico as pr
    from andesbev_sim.simulacion import flota as fl
    from andesbev_sim.simulacion.red import construir_red

    f = cfg.flota
    lineas = _leer(cfg, "lineas_demanda.parquet", "ingesta")
    dev = _leer(cfg, "lineas_devolucion.parquet", "ingesta")
    clientes = _leer(cfg, "clientes.parquet", "segmentacion")
    params = pr.cargar(_interim(cfg) / "pronostico")

    # --- 1. Cómo opera hoy la flota (camión-día)
    td = fl.perfilar_camion_dia(lineas, clientes, f.umbral_ruta_dedicada, f.umbral_ruta_periferia)
    cap, observado = fl.calibrar(td, f.percentil_capacidad)
    for k, c in cap.items():
        log.info(
            "Capacidad %s: %.1f pallets/día, %.0f paradas/día", k, c.pallets_dia, c.paradas_dia
        )
    pct_mixta = float((td["TIPO"] == "MIXTA").mean())
    log.info("Camiones-día mixtos (sin canal definido): %.0f%%", 100 * pct_mixta)

    # --- 2. Logística inversa: tasa de devolución por pool
    tasas = fl.tasas_devolucion(dev, lineas, clientes)
    log.info("Tasa de devolución por pool: %s", tasas)

    # --- 3. Dimensionamiento del piloto 2026
    red = construir_red(params.reparto, params.modelo.segmentos)
    esp = pr.esperado(params.modelo, params.fechas, params.nivel).to_numpy()
    dim = fl.dimensionar(
        red, esp, params.fechas.month.to_numpy(), cap, f.percentil_dimensionamiento, log
    )
    sugeridos = dict(zip(dim["POOL"], dim["camiones"].astype(int), strict=True))

    resultado = {
        "capacidad_camion": fl.capacidades_a_dict(cap),
        "camiones_sugeridos": sugeridos,
        "tasa_devolucion": tasas,
        "pct_camion_dias_mixtos": round(pct_mixta, 4),
        "camiones_por_dia_hoy": round(float(td.groupby("FECHA").size().mean()), 1),
    }
    with open(_interim(cfg) / "flota.json", "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, ensure_ascii=False, indent=2)
    tb = _tablas(cfg)
    guardar_tabla(observado.round(3), tb / "paso4_flota_observada.csv")
    guardar_tabla(dim.round(2), tb / "paso4_dimensionamiento.csv")
    return {"resultado": resultado, "observado": observado, "dimensionamiento": dim}


# --------------------------------------------------------------------------- #
# Paso 5
# --------------------------------------------------------------------------- #
def _carpeta_corrida(cfg: Config, run_id: str) -> Path:
    return cfg.ruta(cfg.datos.carpeta_salidas) / "corridas" / run_id


def preparar_simulacion(cfg: Config):
    """Carga los insumos de los pasos 3 y 4 y genera la demanda estocástica común."""
    from andesbev_sim.demanda import pronostico as pr
    from andesbev_sim.simulacion import escenarios as esc

    if not (_interim(cfg) / "pronostico").exists():
        raise EtapaPreviaFaltanteError(
            "Falta el pronóstico. Corre: python main.py --etapa pronostico"
        )
    ruta_flota = _interim(cfg) / "flota.json"
    if not ruta_flota.exists():
        raise EtapaPreviaFaltanteError("Falta la flota. Corre: python main.py --etapa flota")
    params = pr.cargar(_interim(cfg) / "pronostico")
    with open(ruta_flota, encoding="utf-8") as f:
        flota = json.load(f)
    # El calendario se arma con el horizonte de ESTE escenario
    p = cfg.pronostico
    params.fechas = pr.calendario(p.inicio_horizonte, cfg.corrida.horizonte_dias, p.feriados)
    demanda = esc.generar_demanda(cfg, params)
    return params, flota, demanda


def simular(cfg: Config, log: logging.LoggerAdapter, run_id: str, progreso=None):
    """Un escenario (el del YAML indicado)."""
    from andesbev_sim.simulacion import escenarios as esc

    params, flota, demanda = preparar_simulacion(cfg)
    log.info(
        "Demanda generada: %d réplicas × %d días × %d segmentos (semilla %d)",
        *demanda.shape,
        cfg.corrida.semilla,
    )
    corrida = esc.ejecutar(cfg, params, flota, demanda, log, progreso)
    esc.guardar_corrida(corrida, _carpeta_corrida(cfg, run_id))
    return corrida


def aplicar_ajustes(cfg: Config, ajustes: dict | None) -> None:
    """Ajustes desde la GUI: camiones por pool, factor de capacidad e intensidad
    de las disrupciones (sin tocar los YAML)."""
    if not ajustes:
        return
    for pool, n in (ajustes.get("camiones") or {}).items():
        if n is not None:
            setattr(cfg.flota.camiones, pool, int(n))
    if ajustes.get("factor_capacidad") is not None:
        cfg.flota.factor_capacidad = float(ajustes["factor_capacidad"])
    for d in cfg.disrupciones:
        if d.tipo == "bloqueo_vial" and ajustes.get("capacidad_restante_bloqueo") is not None:
            d.capacidad_restante = float(ajustes["capacidad_restante_bloqueo"])
        if d.tipo == "peak_anomalo" and ajustes.get("multiplicador_peak") is not None:
            d.multiplicador_demanda = float(ajustes["multiplicador_peak"])


def comparar(
    ruta_base: str | Path,
    ruta_disrupcion: str | Path,
    overrides: dict | None = None,
    progreso=None,
    ajustes: dict | None = None,
) -> dict:
    """Corre base y disrupción con la misma demanda y compara. Cada escenario
    escribe su propio log en logs/base/ y logs/disrupcion/."""
    from andesbev_sim.config import cargar_config
    from andesbev_sim.logging_setup import cerrar_logging, configurar_logging, nuevo_run_id
    from andesbev_sim.simulacion import escenarios as esc

    overrides = overrides or {}
    cfg_b = cargar_config(ruta_base, **overrides)
    cfg_d = cargar_config(ruta_disrupcion, **overrides)
    aplicar_ajustes(cfg_b, ajustes)
    aplicar_ajustes(cfg_d, ajustes)
    if cfg_b.corrida.semilla != cfg_d.corrida.semilla:
        raise ValueError("Base y disrupción deben tener la misma semilla (demanda común)")
    run_id = nuevo_run_id("comparacion")
    carpeta = _carpeta_corrida(cfg_b, run_id)
    corridas, rutas_log = {}, {}

    params, flota, demanda = preparar_simulacion(cfg_b)
    for i, cfg in enumerate((cfg_b, cfg_d)):
        tipo = cfg.corrida.nombre
        log, rutas = configurar_logging(
            cfg.ruta(cfg.logging.carpeta),
            tipo,
            f"{run_id}_{tipo}",
            cfg.logging.nivel_consola,
            cfg.logging.nivel_archivo,
        )
        rutas_log[tipo] = rutas
        try:
            log.info("Inicio escenario '%s' (%d réplicas)", tipo, cfg.corrida.replicas)

            def prog(x, msg, i=i, tipo=tipo):
                if progreso:
                    progreso(0.5 * i + 0.5 * x, f"{tipo}: {msg}")

            corridas[tipo] = esc.ejecutar(cfg, params, flota, demanda, log, prog)
            esc.guardar_corrida(corridas[tipo], carpeta)
            if i == 1:
                comp = esc.comparar(
                    corridas[cfg_b.corrida.nombre],
                    corridas[tipo],
                    cfg.simulacion.tolerancia_recuperacion_pallets,
                    log,
                )
                esc.guardar_comparacion(comp, carpeta)
        except Exception:
            log.exception("El escenario '%s' terminó con error", tipo)
            raise
        finally:
            cerrar_logging()

    with open(carpeta / "corrida.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "escenarios": list(corridas),
                "replicas": cfg_b.corrida.replicas,
                "semilla": cfg_b.corrida.semilla,
                "dias_despacho": len(params.fechas),
                "logs": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in rutas_log.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if progreso:
        progreso(1.0, "listo")
    return {
        "run_id": run_id,
        "carpeta": carpeta,
        "corridas": corridas,
        "comparacion": comp,
        "rutas_log": rutas_log,
        "params": params,
    }


# --------------------------------------------------------------------------- #
# Paso 6
# --------------------------------------------------------------------------- #
def ultima_comparacion(cfg: Config) -> Path:
    base = cfg.ruta(cfg.datos.carpeta_salidas) / "corridas"
    candidatas = sorted(p for p in base.glob("*_comparacion") if (p / "corrida.json").exists())
    if not candidatas:
        raise EtapaPreviaFaltanteError(
            "No hay comparaciones. Corre: python main.py --etapa comparar"
        )
    return candidatas[-1]


def dashboard(cfg: Config, log: logging.LoggerAdapter, carpeta: Path | None = None) -> Path:
    from andesbev_sim.reportes.dashboard import generar

    carpeta = Path(carpeta) if carpeta else ultima_comparacion(cfg)
    ruta = generar(carpeta)
    log.info("Dashboard generado: %s (%.0f KB)", ruta, ruta.stat().st_size / 1024)
    return ruta
