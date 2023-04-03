FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN pip install azure-ai-ml \
            azure-identity \
            onnxruntime \
            datasets \
            transformers \
            onnxconverter_common \
            olive-ai==0.1.0

WORKDIR /olive
