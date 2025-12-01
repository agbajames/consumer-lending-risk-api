FROM python:3.11-slim

# LightGBM/XGBoost may need OpenMP + build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Healthcheck hits FastAPI /health
HEALTHCHECK CMD curl -fsS http://localhost:8000/health || exit 1

# Run as non-root
USER 1000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]