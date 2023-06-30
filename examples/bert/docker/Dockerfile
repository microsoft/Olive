FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN pip install onnxruntime \
            datasets \
            evaluate \
            pydantic<2.0.0 \
            scikit-learn \
            transformers \
            git+https://github.com/microsoft/Olive.git

WORKDIR /olive
