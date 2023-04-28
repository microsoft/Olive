FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN pip install azure-ai-ml \
            azure-identity \
            onnxruntime \
            datasets \
            transformers \
            onnxconverter_common \
            git+https://github.com/microsoft/Olive.git

# need to be removed later
RUN pip install git+https://github.com/microsoft/onnxruntime-extensions.git
RUN pip uninstall -y onnxruntime
RUN pip install ort-nightly==1.15.0.dev20230426002 --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/

WORKDIR /olive
