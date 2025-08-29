FROM python:3.10

# Install system dependencies and compatible GCC for CUDA
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    libffi-dev \
    libssl-dev \
    libgl1 \
    git \
    ca-certificates \
    wget \
    curl \
    gcc-12 \
    g++-12

# Set environment variables for CUDA build compatibility
ENV CC=gcc-12
ENV CXX=g++-12
ENV CUDAHOSTCXX=/usr/bin/g++-12
ENV NVCC_FLAGS="--allow-unsupported-compiler"

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
