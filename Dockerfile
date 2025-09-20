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

# Install PyTorch separately (before installing other dependencies)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Copy and install dependencies first (improves build caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code after dependencies are installed
COPY . .

EXPOSE 8000

CMD ["gunicorn", "src.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000"]
