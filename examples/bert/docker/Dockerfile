# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# mcr image https://github.com/microsoft/mcr
# tag list https://mcr.microsoft.com/v2/azureml/openmpi4.1.0-ubuntu20.04/tags/list
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

ARG OPENVINO_VERSION=2023.3.0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && ACCEPT_EULA=Y apt-get -y upgrade
RUN apt-get install -y --no-install-recommends wget gnupg

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install pandas \
            plotly \
            psutil \
            datasets \
            transformers \
            optimum \
            onnxruntime-openvino \
            "numpy<2.0" \
            evaluate \
            scikit-learn==1.5.2 \
            git+https://github.com/microsoft/Olive.git \
            --no-cache-dir

ENV INTEL_OPENVINO_DIR /opt/intel/openvino_${OPENVINO_VERSION}
ENV LD_LIBRARY_PATH $INTEL_OPENVINO_DIR/runtime/lib/intel64:$INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH
ENV InferenceEngine_DIR $INTEL_OPENVINO_DIR/runtime/cmake
ENV ngraph_DIR $INTEL_OPENVINO_DIR/runtime/cmake
ENV IE_PLUGINS_PATH $INTEL_OPENVINO_DIR/runtime/lib/intel64

# Install OpenVINO
RUN cd /opt && mkdir -p intel && cd intel && \
    wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz && \
    tar xzf l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz && rm -rf l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz && \
    mv l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64 openvino_2023.3.0 && \
    cd $INTEL_OPENVINO_DIR/install_dependencies && ./install_openvino_dependencies.sh -y

WORKDIR /root

RUN wget "https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-gmmlib_21.3.3_amd64.deb" && \
    wget "https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.9441/intel-igc-core_1.0.9441_amd64.deb" && \
    wget "https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.9441/intel-igc-opencl_1.0.9441_amd64.deb" && \
    wget "https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-opencl-icd_21.48.21782_amd64.deb" && \
    wget "https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-level-zero-gpu_1.2.21782_amd64.deb" && \
    dpkg -i *.deb && rm -rf *.deb

WORKDIR /olive
