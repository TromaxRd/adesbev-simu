"""Punto de entrada del simulador AndesBev.

Uso (desde la carpeta del repositorio):

    python main.py --etapa ingesta        # paso 1
    python main.py --etapa segmentacion   # paso 2
    python main.py --etapa pronostico     # paso 3
    python main.py --etapa flota          # paso 4
    python main.py --etapa simular --escenario config/disrupcion.yaml
    python main.py --etapa comparar       # paso 5: base vs disrupción (+ dashboard)
    python main.py --etapa dashboard      # paso 6: regenera el dashboard de la última corrida
    python main.py --modo gui             # paso 7: panel web en http://localhost:8000

Las etapas se van habilitando paso a paso; el modo GUI llega en el paso 7.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Permite correr "python main.py" sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from andesbev_sim import etapas  # noqa: E402
from andesbev_sim.config import cargar_config  # noqa: E402
from andesbev_sim.logging_setup import (  # noqa: E402
    cerrar_logging,
    configurar_logging,
    nuevo_run_id,
)

ETAPAS = [
    "ingesta",
    "segmentacion",
    "pronostico",
    "flota",
    "simular",
    "comparar",
    "dashboard",
    "todo",
]


def _reenviar(argv: list[str] | None) -> list[str]:
    """Argumentos originales sin --etapa (para encadenar etapas en 'todo')."""
    a = list(sys.argv[1:] if argv is None else argv)
    if "--etapa" in a:
        i = a.index("--etapa")
        del a[i : i + 2]
    return a


def _argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulación downstream AndesBev")
    p.add_argument("--modo", choices=["cli", "gui"], default="cli")
    p.add_argument("--etapa", choices=ETAPAS, default="ingesta")
    p.add_argument("--escenario", default="config/base.yaml", help="YAML del escenario")
    p.add_argument(
        "--disrupcion", default="config/disrupcion.yaml", help="YAML de disrupción (comparar)"
    )
    p.add_argument("--corrida", help="Carpeta de outputs/corridas para el dashboard")
    p.add_argument("--replicas", type=int, help="Sobrescribe corrida.replicas")
    p.add_argument("--semilla", type=int, help="Sobrescribe corrida.semilla")
    p.add_argument("--host", default="127.0.0.1", help="GUI: interfaz de red (0.0.0.0 en Docker)")
    p.add_argument("--port", type=int, default=8000, help="GUI: puerto")
    return p.parse_args(argv)


def etapa_ingesta(cfg, log) -> None:
    r = etapas.ingesta(cfg, log)
    print("\n================ RESUMEN DE INGESTA ================")
    print(f"Período           : {r.fecha_min} a {r.fecha_max} ({r.dias_con_despacho} días)")
    print(f"Filas leídas      : {r.filas_leidas:,}")
    for nombre, t in (("Demanda", r.demanda), ("Devolución", r.devolucion)):
        print(
            f"{nombre:<18}: {t['lineas']:>10,} líneas | {t['pallets']:>10,.0f} pallets | "
            f"{t['clientes']:,} clientes | {t['skus']:,} SKU"
        )
    print(f"Tasa devolución   : {r.devolucion['pallets'] / r.demanda['pallets']:.2%} (en pallets)")
    print(f"Descartes         : {r.descartes}")
    print(f"Flota observada   : {r.flota_observada}")
    if r.advertencias:
        print("Advertencias      :")
        for a in r.advertencias:
            print(f"  - {a}")


def etapa_segmentacion(cfg, log) -> None:
    import pandas as pd

    r = etapas.segmentacion(cfg, log)
    pd.set_option("display.width", 140, "display.max_columns", 20)
    res = r["resumen"]
    print("\n================ RESUMEN DE SEGMENTACIÓN ================")
    print("\nMatriz ABC/XYZ (% del volumen en pallets):")
    print((100 * r["matriz"].fillna(0)).round(1).to_string())
    print("\nSKU por clase:", res["skus_por_clase"])
    print("\nCanal por segmento comercial:")
    print(r["canales"].round(3).to_string())
    print("\nCentro de distribución:", res["centro_distribucion"])
    print("\nZonas:", res["zonas"])
    cols = [
        "ZONA",
        "SECTOR",
        "comuna_dominante",
        "clientes",
        "visitas_por_dia",
        "pallets_por_dia",
        "pallets_por_visita",
        "pct_pallets_ancla",
        "dist_cd_km_mediana",
    ]
    print("\nSectores:")
    print(r["sectores"][cols].round(2).to_string(index=False))


def etapa_pronostico(cfg, log) -> None:
    r = etapas.pronostico(cfg, log)
    b = r["resumen"]["backtest"]
    print("\n================ RESUMEN DE PRONÓSTICO ================")
    print(f"\nBacktest {b['anio_prueba']} (ajuste con años anteriores):")
    print(
        f"  WAPE diario   : {b['wape_diario']:.1%}  (sin estacionalidad: "
        f"{b['wape_diario_sin_estacionalidad']:.1%})"
    )
    print(
        f"  WAPE mensual  : {b['wape_mensual']:.1%}  (sin estacionalidad: "
        f"{b['wape_mensual_sin_estacionalidad']:.1%})"
    )
    print(f"  Sesgo total   : {b['sesgo_total']:+.1%}")
    print(f"  Cobertura IC90: {b['cobertura_intervalo_90']:.0%} de los días dentro de P5-P95")
    print("\nBacktest mensual (pallets):")
    print(r["bt_mensual"].round(0).to_string())
    print("\nÍndice mensual (TOTAL y segmentos principales):")
    print(
        r["idx"][["TOTAL", "ANCLA-A-X", "ANCLA-A-Y", "CAPILAR-A-X", "CAPILAR-A-Y"]]
        .round(2)
        .to_string()
    )
    print("\nÍndice día de semana (0=lunes):", r["resumen"]["indice_dia_semana_total"])
    print(
        f"\nHorizonte: {r['resumen']['dias_horizonte']} días de despacho | nivel "
        f"{r['resumen']['nivel_pallets_dia']} pallets/día"
    )
    print(r["horizonte"].round(0).to_string())
    print(
        f"Día más cargado del año (mediana de réplicas): {r['resumen']['dia_pico_p50']} pallets"
        f" | P95: {r['resumen']['dia_pico_p95']}"
    )


def etapa_flota(cfg, log) -> None:
    import pandas as pd

    pd.set_option("display.width", 160, "display.max_columns", 20)
    r = etapas.flota(cfg, log)
    print("\n================ RESUMEN DE FLOTA ================")
    print("\nOperación observada 2023-2025 (camión-día):")
    print(r["observado"].round(2).to_string(index=False))
    print("\nCapacidad calibrada por camión:", r["resultado"]["capacidad_camion"])
    print("Tasa de devolución por pool:", r["resultado"]["tasa_devolucion"])
    print("\nDimensionamiento 2026:")
    print(r["dimensionamiento"].round(1).to_string(index=False))
    print(
        f"\nCamiones sugeridos: {r['resultado']['camiones_sugeridos']} | hoy salen "
        f"{r['resultado']['camiones_por_dia_hoy']} por día"
    )


def _imprimir_kpis(nombre: str, k: dict) -> None:
    g = k["global"]
    print(f"\n--- Escenario {nombre} ---")
    print(
        f"Servicio a tiempo : {g['servicio_a_tiempo']['media']:.2%} "
        f"± {g['servicio_a_tiempo']['ic95']:.2%}"
    )
    print(f"Fill rate final   : {g['fill_rate']['media']:.2%}")
    print(f"Venta perdida     : {g['venta_perdida_pallets']['media']:,.0f} pallets/año")
    print(
        f"Backlog promedio  : {g['backlog_promedio_pallets']['media']:.1f} pallets "
        f"(máximo {g['backlog_maximo_pallets']['media']:.0f})"
    )
    print(
        f"Devoluciones      : {g['devolucion_pallets']['media']:,.0f} pallets "
        f"({g['tasa_devolucion']['media']:.2%})"
    )
    print("Por pool:")
    for p, b in k["por_pool"].items():
        f = k["flota"][p]
        print(
            f"  {p:<18} servicio {b['servicio_a_tiempo']['media']:.2%} | "
            f"{f['camiones']} camiones | uso pallets {f['uso_pallets']['media']:.0%} "
            f"paradas {f['uso_paradas']['media']:.0%} | días saturado "
            f"{f['dias_saturado']['media']:.0f}"
        )
    print(
        "Por zona:", {z: f"{b['servicio_a_tiempo']['media']:.2%}" for z, b in k["por_zona"].items()}
    )


def etapa_simular(cfg, log) -> None:
    run_id = nuevo_run_id(f"simulacion_{cfg.corrida.nombre}")
    c = etapas.simular(cfg, log, run_id)
    _imprimir_kpis(c.nombre, c.kpis)
    print("\nMensual:")
    print(c.kpis["mensual"].round(3).to_string(index=False))


def correr_comparacion(args) -> int:
    import pandas as pd

    pd.set_option("display.width", 160, "display.max_columns", 20)
    ov = {"replicas": args.replicas, "semilla": args.semilla}
    r = etapas.comparar(args.escenario, args.disrupcion, ov)
    for nombre, c in r["corridas"].items():
        _imprimir_kpis(nombre, c.kpis)
    comp = r["comparacion"]["comparacion"]
    print("\n================ BASE vs DISRUPCIÓN ================")
    print(
        f"Servicio a tiempo: {comp['servicio_a_tiempo_base']:.2%} -> "
        f"{comp['servicio_a_tiempo_disrupcion']:.2%} "
        f"(pérdida {comp['perdida_servicio_pp']:.2f} pp)"
    )
    print(f"Venta perdida extra: {comp['venta_perdida_extra_pallets']:,.0f} pallets")
    print("\nRecuperación por evento:")
    print(r["comparacion"]["recuperacion"].round(2).to_string(index=False))
    from andesbev_sim.reportes.dashboard import generar

    ruta = generar(r["carpeta"])
    print(f"\nResultados en {r['carpeta']}")
    print(f"Dashboard     : {ruta}")
    return r


def etapa_dashboard(cfg, log, carpeta=None) -> None:
    ruta = etapas.dashboard(cfg, log, carpeta)
    print(f"\nDashboard: {ruta}")


def main(argv: list[str] | None = None) -> int:
    # La consola de Windows puede no estar en UTF-8 (tildes, ×, ±): evita errores al imprimir
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")
    args = _argumentos(argv)
    if args.modo == "gui":
        import uvicorn

        print(
            f"Panel en http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}"
        )
        uvicorn.run("andesbev_sim.web.app:app", host=args.host, port=args.port, log_level="warning")
        return 0

    if args.etapa == "todo":  # pipeline completo, en orden
        for etapa in ["ingesta", "segmentacion", "pronostico", "flota"]:
            codigo = main([*_reenviar(argv), "--etapa", etapa])
            if codigo:
                return codigo
        args.etapa = "comparar"

    if args.etapa == "comparar":
        try:
            correr_comparacion(args)
        except Exception as e:  # el detalle ya quedó en el log del escenario
            print(f"ERROR: {e}")
            return 1
        return 0

    cfg = cargar_config(args.escenario, replicas=args.replicas, semilla=args.semilla)
    run_id = nuevo_run_id(f"{args.etapa}_{cfg.corrida.nombre}")
    tipo_log = cfg.corrida.nombre if args.etapa == "simular" else "preparacion"
    log, rutas = configurar_logging(
        cfg.ruta(cfg.logging.carpeta),
        tipo_log,
        run_id,
        cfg.logging.nivel_consola,
        cfg.logging.nivel_archivo,
    )
    log.info("Inicio | etapa=%s | escenario=%s", args.etapa, args.escenario)
    t0 = time.perf_counter()
    try:
        {
            "ingesta": etapa_ingesta,
            "segmentacion": etapa_segmentacion,
            "pronostico": etapa_pronostico,
            "flota": etapa_flota,
            "simular": etapa_simular,
            "dashboard": lambda c, lg: etapa_dashboard(c, lg, args.corrida),
        }[args.etapa](cfg, log)
    except Exception:
        log.exception("La etapa '%s' terminó con error", args.etapa)
        return 1
    finally:
        log.info("Fin | %.1f s | logs en %s", time.perf_counter() - t0, rutas["log"].parent)
        cerrar_logging()
    return 0


if __name__ == "__main__":
    sys.exit(main())
