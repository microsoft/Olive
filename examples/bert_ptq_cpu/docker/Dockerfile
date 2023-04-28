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
RUN pip install --index-url https://olivemodels.azureedge.net/olive2/wheels onnxruntime

WORKDIR /olive
