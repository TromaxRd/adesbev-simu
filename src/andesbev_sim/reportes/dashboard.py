"""Generación automática del dashboard XHTML de una corrida base vs disrupción.

Lee la carpeta ``outputs/corridas/<run_id>`` (KPIs, series y eventos) y los logs
JSON de cada escenario, arma el contexto y lo vuelca en la plantilla Jinja2
``templates/dashboard.xhtml.j2``. El resultado es UN archivo ``.xhtml``
autocontenido: estilos, datos y la librería Chart.js van incrustados, así que se
abre sin conexión en cualquier navegador y se puede adjuntar como artifact de CI.

XHTML es XML: todo debe quedar bien formado. Por eso los scripts van dentro de
secciones CDATA y los textos que vienen de los logs se escapan automáticamente.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from andesbev_sim.simulacion.red import POOLS

AQUI = Path(__file__).resolve().parent
NOMBRE_POOL = {
    "DEDICADA": "Dedicada (ancla)",
    "CAPILAR_NUCLEO": "Capilar núcleo",
    "CAPILAR_PERIFERIA": "Capilar periferia",
}
NOMBRE_ZONA = {"NUCLEO": "Núcleo", "PERIFERIA": "Periferia", "SIN_GEO": "Sin coordenadas"}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _leer_log(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def resumen_logs(rutas: dict[str, Path]) -> dict:
    """Conteo por nivel, duración y registros más relevantes (WARNING y ERROR)."""
    out = {"escenarios": [], "relevantes": []}
    for escenario, ruta in rutas.items():
        regs = _leer_log(Path(ruta))
        if not regs:
            continue
        niveles = Counter(r["nivel"] for r in regs)
        t0 = datetime.fromisoformat(regs[0]["timestamp"])
        t1 = datetime.fromisoformat(regs[-1]["timestamp"])
        out["escenarios"].append(
            {
                "escenario": escenario,
                "archivo": Path(ruta).name,
                "registros": len(regs),
                "info": niveles.get("INFO", 0),
                "warning": niveles.get("WARNING", 0),
                "error": niveles.get("ERROR", 0),
                "duracion_s": round((t1 - t0).total_seconds(), 2),
                "inicio": regs[0]["timestamp"],
            }
        )
        for r in regs:
            if r["nivel"] in ("WARNING", "ERROR"):
                out["relevantes"].append(
                    {
                        "hora": r["timestamp"][11:19],
                        "escenario": escenario,
                        "nivel": r["nivel"],
                        "mensaje": r["mensaje"],
                    }
                )
    return out


def _pct(x: float, dec: int = 1) -> str:
    return f"{100 * x:.{dec}f} %".replace(".", ",")


def _num(x: float, dec: int = 0) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def construir_contexto(carpeta: Path) -> dict:
    with open(carpeta / "corrida.json", encoding="utf-8") as f:
        meta = json.load(f)
    esc = meta["escenarios"]  # [base, disrupcion]
    nb, nd = esc[0], esc[1]
    k = {e: json.load(open(carpeta / f"{e}_kpis.json", encoding="utf-8")) for e in esc}
    diario = {e: pd.read_csv(carpeta / f"{e}_diario.csv", parse_dates=["FECHA"]) for e in esc}
    mensual = {e: pd.read_csv(carpeta / f"{e}_mensual.csv") for e in esc}
    sectores = {e: pd.read_csv(carpeta / f"{e}_sectores.csv") for e in esc}
    comp = json.load(open(carpeta / "comparacion.json", encoding="utf-8"))
    rec = (
        pd.read_csv(carpeta / "recuperacion.csv")
        if (carpeta / "recuperacion.csv").exists()
        else pd.DataFrame()
    )
    ruta_ev = carpeta / f"{nd}_eventos.csv"
    eventos = pd.read_csv(ruta_ev) if ruta_ev.exists() else pd.DataFrame()

    gb, gd = k[nb]["global"], k[nd]["global"]

    # --- 1. Resumen ejecutivo
    tarjetas = [
        {
            "etiqueta": "Servicio a tiempo · base",
            "valor": _pct(gb["servicio_a_tiempo"]["media"]),
            "nota": f"± {_pct(gb['servicio_a_tiempo']['ic95'], 2)} (IC 95 %)",
        },
        {
            "etiqueta": "Servicio a tiempo · disrupción",
            "valor": _pct(gd["servicio_a_tiempo"]["media"]),
            "nota": f"{_num(comp['perdida_servicio_pp'], 2)} pp menos que la base",
        },
        {
            "etiqueta": "Fill rate final · disrupción",
            "valor": _pct(gd["fill_rate"]["media"], 2),
            "nota": f"{_num(gd['venta_perdida_pallets']['media'])} pallets de venta perdida",
        },
        {
            "etiqueta": "Volumen anual simulado",
            "valor": f"{_num(gb['demanda_pallets']['media'])}",
            "nota": f"pallets en {meta['dias_despacho']} días de despacho",
        },
        {
            "etiqueta": "Factor peak / valle",
            "valor": _num(k[nb]["factor_peak_valle"], 2).replace(",00", ""),
            "nota": "mes más cargado ÷ mes más liviano",
        },
        {
            "etiqueta": "Devoluciones",
            "valor": _pct(gb["tasa_devolucion"]["media"], 2),
            "nota": f"{_num(gb['devolucion_pallets']['media'])} pallets de retorno al año",
        },
    ]
    pools = [
        {
            "pool": NOMBRE_POOL[p],
            "camiones": k[nb]["flota"][p]["camiones"],
            "cap_pallets": _num(k[nb]["flota"][p]["capacidad_pallets_dia"]),
            "cap_paradas": _num(k[nb]["flota"][p]["capacidad_paradas_dia"]),
            "servicio_base": _pct(k[nb]["por_pool"][p]["servicio_a_tiempo"]["media"]),
            "servicio_dis": _pct(k[nd]["por_pool"][p]["servicio_a_tiempo"]["media"]),
            "uso_pallets": _pct(k[nb]["flota"][p]["uso_pallets"]["media"], 0),
            "uso_paradas": _pct(k[nb]["flota"][p]["uso_paradas"]["media"], 0),
            "saturado": _num(k[nb]["flota"][p]["dias_saturado"]["media"]),
            "devolucion": _pct(k[nb]["por_pool"][p]["tasa_devolucion"]["media"], 2),
        }
        for p in POOLS
    ]

    # --- 4. Territorio
    zonas = [
        {
            "zona": NOMBRE_ZONA.get(z, z),
            "demanda": _num(
                k[nb]["por_zona"][z]["demanda_pallets"]["media"] / meta["dias_despacho"], 1
            ),
            "servicio_base": _pct(k[nb]["por_zona"][z]["servicio_a_tiempo"]["media"]),
            "servicio_dis": _pct(k[nd]["por_zona"][z]["servicio_a_tiempo"]["media"]),
            "backlog": _num(k[nd]["por_zona"][z]["backlog_promedio_pallets"]["media"], 1),
        }
        for z in k[nb]["por_zona"]
    ]
    sb = sectores[nb].set_index("SECTOR")
    sd = sectores[nd].set_index("SECTOR")
    orden_sect = sb.sort_values(["ZONA", "demanda_pallets_dia"], ascending=[True, False]).index

    # --- 5. Resiliencia
    recuperacion = [
        {
            "evento": r.nombre.replace("_", " "),
            "tipo": r.tipo.replace("_", " "),
            "eventos": int(r.eventos),
            "duracion": _num(r.duracion_promedio, 1),
            "recuperacion": _num(r.dias_recuperacion, 1),
            "recuperacion_p90": _num(r.dias_recuperacion_p90, 0),
            "perdida": _num(r.perdida_servicio_pp, 1),
            "backlog_extra": _num(r.backlog_maximo_extra, 0),
            "perdida_venta": _num(r.venta_perdida_extra, 1),
        }
        for r in rec.itertuples()
    ]
    ventanas = []
    if not eventos.empty:
        det = eventos[(eventos["determinista"]) & (eventos["replica"] == 0)]
        for e in det.itertuples():
            ventanas.append(
                {
                    "desde": str(e.fecha_inicio),
                    "hasta": str(e.fecha_fin),
                    "nombre": e.nombre.replace("_", " "),
                }
            )

    # --- 6. Auditoría
    rutas_log = {e: Path(v["json"]) for e, v in meta["logs"].items()}
    logs = resumen_logs(rutas_log)
    logs["relevantes"] = logs["relevantes"][:25]

    datos = {
        "devolucion": {
            "pools": [NOMBRE_POOL[p] for p in POOLS],
            "tasa": [
                round(100 * k[nb]["por_pool"][p]["tasa_devolucion"]["media"], 2) for p in POOLS
            ],
        },
        "flota": {
            "pools": [NOMBRE_POOL[p] for p in POOLS],
            "uso_pallets": [
                round(100 * k[nb]["flota"][p]["uso_pallets"]["media"], 1) for p in POOLS
            ],
            "uso_paradas": [
                round(100 * k[nb]["flota"][p]["uso_paradas"]["media"], 1) for p in POOLS
            ],
        },
        "mensual": {
            "meses": [MESES[m - 1] for m in mensual[nb]["MES"]],
            "demanda": mensual[nb]["demanda_pallets_dia"].round(1).tolist(),
            "servicio_base": (100 * mensual[nb]["servicio_a_tiempo"]).round(1).tolist(),
            "servicio_dis": (100 * mensual[nd]["servicio_a_tiempo"]).round(1).tolist(),
            "uso": {
                NOMBRE_POOL[p]: (100 * mensual[nb][f"uso_{p}"]).round(1).tolist() for p in POOLS
            },
        },
        "sectores": {
            "sectores": list(orden_sect),
            "base": (100 * sb.loc[orden_sect, "servicio_a_tiempo"]).round(1).tolist(),
            "dis": (100 * sd.loc[orden_sect, "servicio_a_tiempo"]).round(1).tolist(),
        },
        "diario": {
            "fechas": diario[nb]["FECHA"].dt.strftime("%Y-%m-%d").tolist(),
            "backlog_base": diario[nb]["pendiente"].round(1).tolist(),
            "backlog_dis": diario[nd]["pendiente"].round(1).tolist(),
            "ventanas": ventanas,
        },
    }
    return {
        "titulo": "Simulación downstream AndesBev · piloto Coquimbo–La Serena",
        "meta": meta,
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tarjetas": tarjetas,
        "pools": pools,
        "zonas": zonas,
        "recuperacion": recuperacion,
        "logs": logs,
        "eventos_deterministas": ventanas,
        "n_eventos": len(eventos),
        "comp": comp,
        "factor_peak_valle": _num(k[nb]["factor_peak_valle"], 2),
        "mes_peor": MESES[
            int(mensual[nb].loc[mensual[nb]["servicio_a_tiempo"].idxmin(), "MES"]) - 1
        ],
        "servicio_peor_mes": _pct(mensual[nb]["servicio_a_tiempo"].min()),
        "datos_json": json.dumps(datos, ensure_ascii=False).replace("]]>", "]]\\u003e"),
    }


def generar(carpeta: Path, salida: Path | None = None) -> Path:
    """Genera ``dashboard_<run_id>.xhtml`` dentro de la carpeta de la corrida."""
    carpeta = Path(carpeta)
    ctx = construir_contexto(carpeta)
    chart = (AQUI / "static" / "chart.umd.js").read_text(encoding="utf-8")
    if "]]>" in chart:
        raise ValueError("chart.umd.js contiene ']]>' y rompería la sección CDATA")
    ctx["chartjs"] = chart
    env = Environment(
        loader=FileSystemLoader(AQUI / "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    html = env.get_template("dashboard.xhtml.j2").render(**ctx)
    salida = salida or carpeta / f"dashboard_{ctx['meta']['run_id']}.xhtml"
    salida.write_text(html, encoding="utf-8")
    return salida
