FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    python3-setuptools \
    python3-wheel \
    libffi-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    ca-certificates \
    wget \
    curl \
    gcc-12 \
    g++-12

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ENV CC=gcc-12
ENV CXX=g++-12
ENV CUDAHOSTCXX=/usr/bin/g++-12
ENV NVCC_FLAGS="--allow-unsupported-compiler"
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# Copy requirements first for caching
COPY requirements_api.txt .


# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir -r requirements_api.txt

# Copy and install local diso wheel, then remove it to keep the image small
COPY diso-0.1.4-cp310-cp310-linux_x86_64.whl /tmp/
RUN python3.10 -m pip install --no-cache-dir /tmp/diso-0.1.4-cp310-cp310-linux_x86_64.whl \
    && rm /tmp/diso-0.1.4-cp310-cp310-linux_x86_64.whl

# Install torch and torchvision for CUDA 12.8
RUN python3.10 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Pre-download Hugging Face model weights (no token required)
RUN python3.10 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='VAST-AI/TripoSG', local_dir='pretrained_weights/TripoSG')" && \
    python3.10 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='VAST-AI/TripoSG-scribble', local_dir='pretrained_weights/TripoSG-scribble')" && \
    python3.10 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='briaai/RMBG-1.4', local_dir='pretrained_weights/RMBG-1.4')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

CMD ["python3", "start_api_simple.py"]