FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/models \
    HF_HOME=/models/.cache/huggingface \
    DOCLING_ARTIFACTS_PATH=/models/.cache/docling/models

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

CMD ["python", "-m", "app.main"]
