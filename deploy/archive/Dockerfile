FROM python:3.9-slim-bookworm

# eccodes for GRIB decoding (cfgrib), proj for pyproj
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes-dev libeccodes-tools libproj-dev proj-data build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# pygrib needs Cython <3 (its .pyx uses Python 2 syntax like xrange/long)
# and numpy must be present for --no-build-isolation to work
RUN pip install --no-cache-dir "Cython<3" numpy==2.0.2 && \
    pip install --no-cache-dir --no-build-isolation pygrib==2.1.6

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps herbie-data==2024.8.0 && \
    pip install --no-cache-dir requests toml

COPY core/ core/
COPY services/ services/
COPY ui/ ui/
COPY templates/ templates/
COPY static/ static/
COPY main.py .

RUN mkdir -p /app/data

# Herbie GRIB cache — use tmpfs, not persistent storage
ENV HERBIE_HOME=/tmp/herbie

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8050/api/health'); exit(0 if r.status_code == 200 else 1)"

CMD ["python", "main.py", "--dashboard"]
