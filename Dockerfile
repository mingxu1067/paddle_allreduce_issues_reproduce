FROM nvcr.io/nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
  
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        apt-utils \
        python3 \
        python3-dev \
        vim \
        sudo \
        python3-pip

RUN python3 -m pip install --upgrade pip
RUN pip3 install paddlepaddle-gpu==2.2.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

WORKDIR /workspace
