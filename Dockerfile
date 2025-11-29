# =========================================================
# TSUKINEKO FaceRecognition — Cloud Run Dockerfile
# =========================================================

FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create workdir
WORKDIR /app

# Copy dependency list
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code (src/, app.py, models, etc.)
COPY . .

# Expose port (Cloud Run uses 8080)
ENV PORT=8080

# Start FastAPI (with uvicorn)
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
