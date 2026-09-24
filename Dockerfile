FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends iverilog && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
CMD ["gunicorn","--bind","0.0.0.0:10000","--workers","1","--timeout","120","app:app"]
