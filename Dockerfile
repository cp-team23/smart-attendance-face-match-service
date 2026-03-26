FROM python:3.10-slim

WORKDIR /app

# System dependencies
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
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download dlib landmark model
RUN python3 -c "\
import os, urllib.request, bz2; \
DAT_FILE = 'shape_predictor_68_face_landmarks.dat'; \
DAT_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'; \
print('Downloading dlib model...'); \
urllib.request.urlretrieve(DAT_URL, DAT_FILE + '.bz2'); \
data = bz2.open(DAT_FILE + '.bz2').read(); \
open(DAT_FILE, 'wb').write(data); \
os.remove(DAT_FILE + '.bz2'); \
print('Done!') \
"

# Expose port
EXPOSE 10000

# Start gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:10000", "--timeout", "120", "app:app"]