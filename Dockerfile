# Dockerfile for Sentiment Classifier MLOps Application

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY app/ ./app/
COPY scripts/ ./scripts/


RUN mkdir -p /app/models /app/logs /tmp

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.onnx


ARG MODEL_URL
ARG ENVIRONMENT=dev

ENV MODEL_URL=${MODEL_URL}
ENV ENVIRONMENT=${ENVIRONMENT}
ARG PREDICTIONS_BUCKET
ENV PREDICTIONS_BUCKET=${PREDICTIONS_BUCKET}


RUN if [ -n "$MODEL_URL" ]; then \
    echo "Downloading model during build..." && \
    python scripts/download_model.py || \
    echo "Model download failed during build, will download at runtime"; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

