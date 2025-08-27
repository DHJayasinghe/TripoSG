FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (grouped to minimize layers)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    build-essential \
    ninja-build \
    libffi-dev \
    libgl1 \
    libssl-dev \
    libssl3 \
    openssl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3.10-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install latest pip separately (avoids Debian pip conflict)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# Install CUDA 12.6.2
WORKDIR /tmp
RUN wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run && \
    chmod +x cuda_12.6.2_560.35.03_linux.run && \
    ./cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    rm cuda_12.6.2_560.35.03_linux.run

# Install core Python packages
RUN python3.10 -m pip install --no-cache-dir setuptools wheel cffi \
    && python3.10 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Copy requirements first for caching
COPY requirements_api.txt .

# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir -r requirements_api.txt

# Pre-download Hugging Face model weights (no token required)
RUN python3.10 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='VAST-AI/TripoSG', local_dir='pretrained_weights/TripoSG')" && \
    python3.10 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='VAST-AI/TripoSG-scribble', local_dir='pretrained_weights/TripoSG-scribble')" && \
    python3.10 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='briaai/RMBG-1.4', local_dir='pretrained_weights/RMBG-1.4')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Start the application using the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
