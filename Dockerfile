FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/scipy and curl for healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libopenblas-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir faiss-cpu

# App code
COPY src /app/src
COPY scripts /app/scripts
COPY samples /app/samples

ENV PYTHONPATH=/app
EXPOSE 8000

CMD bash -lc "python scripts/preflight.py && python scripts/bootstrap_demo.py && uvicorn src.app:app --host 0.0.0.0 --port 8000"


