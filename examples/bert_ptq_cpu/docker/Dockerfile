FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN pip install azure-ai-ml \
            azure-identity \
            onnxruntime \
            datasets \
            transformers \
            onnxconverter_common \
            git+https://github.com/microsoft/Olive.git@jambayk/model-info

WORKDIR /olive
