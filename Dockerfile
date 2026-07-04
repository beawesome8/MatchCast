FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer;
# code changes then don't trigger a full reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "matchcast.api:app", "--host", "0.0.0.0", "--port", "8000"]
