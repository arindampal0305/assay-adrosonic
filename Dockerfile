# ASSAY Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies if any needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY samples/ ./samples/
COPY scripts/ ./scripts/
COPY README.md .

# Create output and data directories
RUN mkdir -p data/chroma_db data/outputs

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
