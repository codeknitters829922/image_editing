FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

# 1. Install dependencies FIRST (This layer is cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy code LAST (Only this layer changes when you edit code)
COPY handler.py .

CMD ["python", "-u", "handler.py"]