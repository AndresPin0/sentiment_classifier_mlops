# Dockerfile for Sentiment Classifier MLOps Application

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create directories for model and logs
RUN mkdir -p /app/models /app/logs /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.onnx

# Download model during build (if MODEL_URL is provided)
# Note: This is optional - model can also be downloaded at runtime
ARG MODEL_URL
ARG ENVIRONMENT=dev

ENV MODEL_URL=${MODEL_URL}
ENV ENVIRONMENT=${ENVIRONMENT}
ENV PREDICTIONS_BUCKET=${PREDICTIONS_BUCKET:-}

# Download model if MODEL_URL is provided at build time
RUN if [ -n "$MODEL_URL" ]; then \
    echo "Downloading model during build..." && \
    python scripts/download_model.py || \
    bash scripts/download_model.sh || \
    echo "Model download failed, will download at runtime"; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

