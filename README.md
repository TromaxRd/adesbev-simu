# andesbev-sim

Simulación de la red downstream de AndesBev SpA para el piloto Coquimbo–La Serena.
Lee el Parquet que genera `consolidar_ventas.py` v4 y, etapa por etapa, segmenta la
cartera, pronostica la demanda estacional, calibra la flota con la operación real,
simula 2026 con y sin disrupciones, y publica un dashboard XHTML autocontenido.

```
Parquet v4 -> ingesta -> segmentación -> pronóstico -> flota -> simulación -> dashboard
             (paso 1)    (paso 2)        (paso 3)      (paso 4)  base vs disr.  XHTML
                                                                  (paso 5)     (paso 6)
```

## Ubicación esperada

```
C:\AndesBev\
├── datos_procesados\ventas_consolidado.parquet   <- lo genera consolidar_ventas.py v4
└── andesbev-sim\                                  <- este repositorio
```

Si el Parquet está en otra parte, cambia `datos.parquet` en `config/base.yaml` o
define la variable de entorno `ANDESBEV_PARQUET`.

## Instalación (Windows)

Con pip:

```powershell
cd C:\AndesBev\andesbev-sim
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

O con uv, que instala exactamente las versiones de `uv.lock`:

```powershell
pip install uv
uv sync --extra dev
```

(Con uv, antepone `uv run` a los comandos: `uv run python main.py ...`.)

## Uso

```powershell
python main.py --etapa todo                 # pasos 1 a 6 de corrido
python main.py --modo gui                   # panel web en http://localhost:8000
```

Etapa por etapa:

| Comando | Paso | Qué deja |
| --- | --- | --- |
| `python main.py --etapa ingesta` | 1 | `data/interim/lineas_*.parquet` validados |
| `python main.py --etapa segmentacion` | 2 | ABC/XYZ, canal ancla/capilar, zonas y sectores |
| `python main.py --etapa pronostico` | 3 | modelo estacional + backtest 2025 |
| `python main.py --etapa flota` | 4 | capacidad calibrada y camiones sugeridos |
| `python main.py --etapa simular --escenario config/base.yaml` | 5 | un escenario |
| `python main.py --etapa comparar` | 5-6 | base vs disrupción + dashboard |
| `python main.py --etapa dashboard` | 6 | regenera el dashboard de la última corrida |

Opciones útiles: `--replicas 50`, `--semilla 7`, `--escenario`, `--disrupcion`.
Cada etapa deja tablas para Excel en `outputs/tablas/` (separador `;`, coma
decimal), resultados en `outputs/corridas/<run_id>/` y su registro en `logs/`.

## Escenarios (sin tocar código)

- `config/base.yaml`: todos los parámetros (segmentación, territorio, feriados,
  flota, backlog). Un campo mal escrito detiene la carga con un error claro.
- `config/disrupcion.yaml`: hereda del base y agrega tres disrupciones (bloqueo
  vial en la periferia en febrero, peak de Fiestas Patrias, quiebres de stock
  aleatorios clase A). Tipos: `bloqueo_vial`, `quiebre_stock`, `peak_anomalo`;
  modos `determinista` (con `fecha_inicio`) o `estocastico` (con
  `probabilidad_diaria`).
- Para probar otra flota: `flota.camiones` en el YAML, o los sliders del panel web.

## Logging y trazabilidad

`logs/preparacion/`, `logs/base/` y `logs/disrupcion/` guardan cada corrida en
`.log` (legible) y `.json` (una línea por evento, con `run_id`, `escenario`,
`replica` y `pool`). Niveles: INFO (avance y KPIs por réplica), WARNING
(disrupciones inyectadas, descartes de datos, venta perdida) y ERROR (fallas).

## Pruebas, calidad y seguridad

```powershell
pytest                                  # 40 pruebas: unitarias + integración de punta a punta
ruff check . ; ruff format --check .    # PEP 8
bandit -r src main.py -ll               # seguridad estática del código
pip-audit --skip-editable               # vulnerabilidades conocidas en dependencias
```

La prueba de integración corre todo el pipeline sobre `data/sample/ventas_muestra.parquet`,
una muestra **sintética** con la misma estructura del Parquet v4 (el real no se
sube a GitHub). Se regenera con `python -m andesbev_sim.muestra`.

## Docker

```powershell
docker compose run --rm pipeline    # pipeline completo por consola
docker compose up app               # panel web en http://localhost:8000
```

El Parquet se monta desde `..\datos_procesados` en modo solo lectura; los
resultados quedan en `data\`, `logs\` y `outputs\` del repositorio. En Windows,
Docker Desktop necesita WSL2.

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` corre en cada push:

1. **Estilo y pruebas:** ruff + pytest con cobertura.
2. **SAST:** bandit y pip-audit.
3. **Simulación:** pipeline completo sobre la muestra (`config/ci_*.yaml`); publica
   el dashboard XHTML, las tablas y los logs como *artifacts* descargables.
4. **Docker + DAST:** construye la imagen, levanta el panel y lo escanea con
   OWASP ZAP (baseline).

## Estructura

```
config/            base.yaml, disrupcion.yaml, ci_*.yaml
src/andesbev_sim/
  config.py        carga y validación de YAML (pydantic)
  logging_setup.py logging .log/.json por tipo de corrida
  etapas.py        orquestación de etapas (la usan la CLI y la GUI)
  ingesta/         lectura y validación del Parquet
  demanda/         segmentación ABC/XYZ y pronóstico estacional
  geo/             punto de Weber, zonas y sectores
  simulacion/      red, flota, disrupciones, motor y escenarios
  kpis/            indicadores y recuperación
  reportes/        dashboard XHTML (Jinja2 + Chart.js incrustado)
  web/             FastAPI + panel XHTML
  muestra.py       generador de la muestra sintética para CI
tests/             pytest
main.py            punto de entrada (--modo cli | gui)
```

## Decisiones de diseño

- **Unidad física: pallets equivalentes.** La demanda mezcla cajas (CJ), unidades
  (CU) y otras (CL); `PALLETS_EQUIV` las vuelve comparables y es la unidad de
  capacidad de un camión.
- **Solo demanda y devoluciones entran al simulador.** Envases, subestándar y
  fleteros ya están analizados en el Frente 6.
- **Canal por tamaño de visita.** Un segmento es ANCLA si su mediana de pallets por
  visita supera 0,6: SUPERMERCADOS, MAYORISTA, E-COMMERCE y CADENAS REGIONALES.
- **Territorio por coordenadas, no por comuna.** El texto de comuna trae la
  dirección de facturación (Las Condes, Quilicura...) aunque el despacho sea local.
- **Pronóstico sin distribuciones genéricas.** Nivel × índice mensual × índice de
  día de semana por segmento (canal × ABC × XYZ), con residuos históricos
  remuestreados en dos escalas (mes completo y día). Backtest 2025: WAPE mensual 3,4 %.
- **Capacidad calibrada con la operación real.** Percentil 95 de lo que hizo un
  camión en un día (pallets para la ruta dedicada, paradas para las capilares).
- **Motor de paso diario** con backlog FIFO y venta perdida tras 2 días de atraso;
  todas las réplicas se calculan juntas con NumPy (segundos, no minutos).
- **Números aleatorios comunes:** base y disrupción usan la misma demanda, así la
  diferencia mide solo el efecto de las fallas.
- **Devoluciones:** viajan en el espacio que liberan las entregas; se reportan como
  KPI pero no consumen capacidad de despacho.

## Limitaciones conocidas

- La banda diaria del pronóstico cubrió el 82 % de los días de 2025 (nominal 90 %);
  la mensual cubrió 11 de 12 meses.
- Los feriados de 2026 están en `base.yaml`; los cierres de mes sin despacho
  observados en 2023-2025 no se modelan.
- Dentro de un pool, la capacidad se reparte en proporción a lo pedido por cada
  sector (no hay ruteo explícito).
