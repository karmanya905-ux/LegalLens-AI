
FROM python:3.11-slim

WORKDIR /app

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data/uploads /app/data/qdrant

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
