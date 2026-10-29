FROM python:3.11-slim

WORKDIR /app

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONDONTWRITEBYTECODE=1

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY training/output/ledgar_tfidf/classifier.joblib ./training/output/ledgar_tfidf/classifier.joblib

RUN mkdir -p /app/data/uploads /app/data/qdrant

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
