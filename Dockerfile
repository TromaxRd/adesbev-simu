# -----------------------------------------------------------------------------
# Imagen del simulador AndesBev. Dos etapas:
#   1) builder: instala las dependencias exactas de uv.lock en un entorno virtual
#   2) final:   copia solo el entorno y el código, y corre con un usuario sin root
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PIP_NO_CACHE_DIR=1
RUN pip install uv==0.11.32
WORKDIR /app
# Primero solo los archivos de dependencias: esta capa se reutiliza si el código cambia
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 andesbev
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY main.py pyproject.toml ./
COPY config ./config
RUN mkdir -p data/interim data/sample logs outputs && chown -R andesbev:andesbev /app
USER andesbev
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/escenarios', timeout=4)"
# Por defecto levanta el panel web; para el pipeline por consola ver docker-compose.yml
CMD ["python", "main.py", "--modo", "gui", "--host", "0.0.0.0", "--port", "8000"]
