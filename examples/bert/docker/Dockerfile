FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN pip install onnxruntime \
            datasets \
            transformers \
            onnxconverter_common \
            git+https://github.com/microsoft/Olive.git

WORKDIR /olive
