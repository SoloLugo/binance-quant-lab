FROM python:3.11-slim-bookworm

# Evita que Python escriba archivos .pyc y fuerza logs en tiempo real para Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Instalamos libgomp1 (OpenMP), requerido por LightGBM en arquitecturas ARM64 y x86_64
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalamos primero las dependencias para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el código fuente
COPY src/ ./src/

# Creamos los directorios para montar los volúmenes persistentes del NVMe
RUN mkdir -p /app/data/raw /app/models

# Comando por defecto: lanzar el bot en vivo
CMD ["python", "src/live_testnet_bot.py"]