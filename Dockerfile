FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY pipeline/ pipeline/
COPY monitoring/ monitoring/
COPY api/ api/

# Create required directories
RUN mkdir -p data models logs model_registry figures

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run API server
CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
