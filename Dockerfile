# Multi-stage build for NVIDIA Stock Prediction API
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models reports data/raw data/processed logs

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command - can be overridden
CMD ["python", "scripts/api.py"]

# Development stage
FROM base as development

# Install additional dev dependencies
RUN pip install --no-cache-dir \
    jupyterlab \
    ipykernel \
    black \
    pytest \
    pytest-cov

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models reports data/raw data/processed logs

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Expose ports for development
EXPOSE 8000 8501 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]