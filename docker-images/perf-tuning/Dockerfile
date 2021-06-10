# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
FROM nvcr.io/nvidia/tensorrt:20.10-py3

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y python3.7 python3.7-dev libgomp1 python3-pip unattended-upgrades
RUN unattended-upgrade
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install setuptools
RUN python3.7 -m pip install psutil GPUtil numpy onnx packaging sympy

ARG OPENVINO_VERSION=2021.3
ENV INTEL_OPENVINO_DIR /opt/intel/openvino_${OPENVINO_VERSION}.394
ENV LD_LIBRARY_PATH $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64:$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH
ENV InferenceEngine_DIR $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share
ENV ngraph_DIR $INTEL_OPENVINO_DIR/deployment_tools/ngraph/cmake
ENV PYTHONPATH $INTEL_OPENVINO_DIR/tools:$PYTHONPATH
ENV IE_PLUGINS_PATH $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64
ENV DEBIAN_FRONTEND=noninteractive

RUN wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
    apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021 && rm GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
    cd /etc/apt/sources.list.d && \
    echo "deb https://apt.repos.intel.com/openvino/2021 all main">intel-openvino-2021.list && \
    apt update && \
    apt install -y intel-openvino-dev-ubuntu18-2021.3.394 && \
    cd ${INTEL_OPENVINO_DIR}/install_dependencies && ./install_openvino_dependencies.sh -y

RUN mkdir /perf_tuning /perf_tuning/src
COPY bin /perf_tuning/bin
COPY src/perf_tuning.py /perf_tuning/src
COPY src/monitor.py /perf_tuning/src
COPY src/maps.py /perf_tuning/src

WORKDIR /

ENTRYPOINT [ "python3.7", "/perf_tuning/src/perf_tuning.py"]