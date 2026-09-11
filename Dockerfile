FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# системные утилиты для healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
    && rm -rf /var/lib/apt/lists/*

# зависимости — только wheel'ы, без сборки из исходников
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    || pip install --no-cache-dir -r requirements.txt

# код и данные
COPY app/ /app/app/
COPY data/ /app/data/

RUN mkdir -p /app/storage

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD wget -qO- http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]