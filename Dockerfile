# Use a slim Python 3.10 image to save space
FROM python:3.10-slim

WORKDIR /app

# Install git (required to install diffusers from GitHub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 1. Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Install diffusers from git main (Critical for Flux.2 Klein)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# 3. Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your serverless handler script
COPY handler.py .

# Start the RunPod serverless handler
CMD ["python", "-u", "handler.py"]
