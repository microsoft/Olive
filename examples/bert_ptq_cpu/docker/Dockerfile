FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=$PIP_INDEX_URL

RUN pip install azure-ai-ml \
            azure-identity \
            onnxruntime \
            datasets \
            transformers \
            olive-ai==0.1.0

WORKDIR /olive
