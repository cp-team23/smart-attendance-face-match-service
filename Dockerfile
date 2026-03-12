FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV / InsightFace
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Render
EXPOSE 10000

# Start Flask with production server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:10000", "app:app"]