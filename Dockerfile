FROM python:3.11-slim-bookworm

# eccodes for GRIB decoding (Herbie + pygrib), proj for pyproj
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes-dev libeccodes-tools libproj-dev proj-data build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ core/
COPY services/ services/
COPY ui/ ui/
COPY templates/ templates/
COPY main.py .

RUN mkdir -p /app/data

# Herbie GRIB cache — use tmpfs, not persistent storage
ENV HERBIE_HOME=/tmp/herbie

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8050/api/health'); exit(0 if r.status_code == 200 else 1)"

CMD ["python", "main.py", "--dashboard"]
