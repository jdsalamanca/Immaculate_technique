# Use a lightweight CUDA image as the base
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
python3 python3-pip
# Set python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    [ -e /usr/bin/pip ] || ln -s /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Ensure build tools
RUN pip install --no-cache-dir packaging setuptools wheel ninja

# Install PyTorch separately (before installing other dependencies)
RUN pip install flash-attn --no-build-isolation

# Copy and install dependencies first (improves build caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code after dependencies are installed
COPY . .

EXPOSE 8080

#Infrastucture disclaimer: Need >32GB RAM CPU and an L4 GPU (>18GB GPU RAM) at least to run this model. 
CMD ["gunicorn", "src.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8080"]
