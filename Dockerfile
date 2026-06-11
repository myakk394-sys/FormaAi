# Use official PyTorch image with CUDA 12.1 and compiler toolchain
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install spconv and cumm for CUDA 12.0/12.1
RUN pip install --no-cache-dir spconv-cu120==2.3.6 cumm-cu120==0.4.11

# Install flash-attention using pre-built wheel to prevent build timeouts
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl || \
    pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl || \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Copy requirements file first to cache installation
COPY requirements.txt .

# Install requirements with no-build-isolation for local git packages that depend on torch
RUN pip install --no-cache-dir -r requirements.txt --no-build-isolation

# Copy all project files
COPY . .

# Create cache directory for Hugging Face (since write permissions are needed for weights)
RUN mkdir -p /app/cache/huggingface /app/cache/torch /app/cache/trimesh /app/static/outputs && \
    chmod -R 777 /app/cache /app/static/outputs

ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV HOME=/app

# Expose Uvicorn default port for Hugging Face Spaces
EXPOSE 7860

# Run the server
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]
