# DisableDockerDetector "Prevent warnings on 1ES builds"
FROM mcr.microsoft.com/mirror/nvcr/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libc++1 \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip

RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN python -m venv qnn-venv
RUN . qnn-venv/bin/activate && \
    pip install git+https://github.com/microsoft/Olive && \
    pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt && \
    pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps

RUN mkdir -p /root/olive-ws

ADD requirements.txt /root/olive-ws/requirements.txt
RUN python -m venv olive-venv
RUN . olive-venv/bin/activate && \
    pip install git+https://github.com/microsoft/Olive && \
    pip install -r /root/olive-ws/requirements.txt && \
    pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-genai-cuda

ENV VIRTUAL_ENV=/olive-venv
ENV PATH="/olive-venv/bin:$PATH"

WORKDIR /root/olive-ws
