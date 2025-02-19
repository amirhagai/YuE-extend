ARG CUDA_VERSION="12.4.1"
ARG CUDNN_VERSION=""
ARG UBUNTU_VERSION="22.04"
ARG DOCKER_FROM=nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION
ARG GRADIO_PORT=7860

FROM $DOCKER_FROM AS base

WORKDIR /

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHON_VERSION=3.12
ENV CONDA_DIR=/opt/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
# ENV NUM_GPUS=1
ENV DOWNLOAD_MODELS="all"

# Install dependencies required for Miniconda
RUN apt-get update -y && \
    apt-get install -y wget bzip2 ca-certificates git curl && \
    apt-get install nodejs -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    ca-certificates \
    cmake \
    curl \
    emacs \
    git \
    jq \
    libcurl4-openssl-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libssl-dev \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    openssh-server \
    openssh-client \
    git-lfs \
    vim \
    zip \
    unzip \
    zlib1g-dev \
    libc6-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
    
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    cmake 


ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH="/opt/conda/envs/pyenv/bin:$PATH"


# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda init bash && \
    $CONDA_DIR/bin/conda create -n pyenv python=3.12 -y && \
    $CONDA_DIR/bin/conda install -n pyenv -c conda-forge openmpi mpi4py -y

# Define PyTorch versions via arguments
ARG PYTORCH="2.5.1"
ARG CUDA="124"

# Install PyTorch with specified version and CUDA
RUN $CONDA_DIR/bin/conda run -n pyenv \
    pip install torch==$PYTORCH torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$CUDA

RUN $CONDA_DIR/bin/conda install -n pyenv nvidia/label/cuda-12.4.1::cuda-nvcc

RUN $CONDA_DIR/bin/conda run -n pyenv pip install setuptools

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Download the exllamav2 wheel file



RUN wget https://github.com/turboderp-org/exllamav2/releases/download/v0.2.8/exllamav2-0.2.8+cu121.torch2.4.0-cp312-cp312-linux_x86_64.whl

# Install the exllamav2 package
RUN $CONDA_DIR/bin/conda run -n pyenv pip install exllamav2-0.2.8+cu121.torch2.4.0-cp312-cp312-linux_x86_64.whl

# Remove the wheel file after installation to keep the image clean
RUN rm exllamav2-0.2.8+cu121.torch2.4.0-cp312-cp312-linux_x86_64.whl

# Install git lfs
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Install nginx
RUN apt-get update && \
    apt-get install -y nginx

COPY docker/default /etc/nginx/sites-available/default

# Add Jupyter Notebook
RUN pip install jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions

RUN pip install -U "huggingface_hub[cli]"


RUN $CONDA_DIR/bin/conda run -n pyenv \
    pip install --no-cache-dir \
    omegaconf \
    einops \
    numpy \
    transformers \
    sentencepiece \
    tqdm \
    tensorboard \
    descript-audiotools>=0.7.2 \
    descript-audio-codec \
    scipy \
    accelerate>=0.26.0 \
    langchain_community \
    tiktoken \
    langchain-openai \
    langchainhub \
    chromadb \
    youtube-transcript-api \
    pytube \
    ragatouille \
    transformers \
    diffusers \
    numpy \
    matplotlib \
    opencv-python \
    pandas \
    keybert \
    ctransformers[cuda] \
    python-dotenv 

RUN apt-get update && apt-get install -y ffmpeg


RUN $CONDA_DIR/bin/conda run -n pyenv pip install -U sentence-transformers==2.2.2
RUN $CONDA_DIR/bin/conda run -n pyenv pip install langchain-huggingface
RUN $CONDA_DIR/bin/conda run -n pyenv pip install pydub
RUN $CONDA_DIR/bin/conda run -n pyenv pip install yt-dlp

RUN apt-get install libsndfile1
RUN /opt/conda/envs/yue/bin/pip install black
RUN /opt/conda/envs/yue/bin/pip install librosa soundfile
RUN /opt/conda/envs/yue/bin/pip install faster_whisper

RUN huggingface-cli download m-a-p/YuE-s1-7B-anneal-en-cot --local-dir workspace/models/YuE-s1-7B-anneal-en-cot
RUN huggingface-cli download m-a-p/YuE-s2-1B-general --local-dir workspace/models/YuE-s2-1B-general  
RUN huggingface-cli download m-a-p/xcodec_mini_infer --local-dir /workspace/YuE-Interface/inference/xcodec_mini_infer

EXPOSE 8888

# Tensorboard
# EXPOSE 6006 

# Debug
# RUN $CONDA_DIR/bin/conda run -n pyenv \
#     pip install debugpy

# EXPOSE 5678


# Copy the entire project
COPY --chmod=755 . /YuE-exllamav2-UI

COPY --chmod=755 docker/initialize.sh /initialize.sh
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh

# Expose the Gradio port
EXPOSE $GRADIO_PORT

CMD [ "/initialize.sh" ]
