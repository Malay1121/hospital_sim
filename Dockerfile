FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0"

# Copy the full project
COPY . .

# Add project root to PYTHONPATH so `from models import`, `from graders import`
# and `from server.app import` all resolve correctly when running from /app
ENV PYTHONPATH="/app"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
