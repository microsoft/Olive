FROM python:3.6

RUN apt-get -y update && apt-get -y upgrade
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy onnx psutil coloredlogs sympy onnxconverter_common docker==5.0.0 six

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN python -m pip install --index-url https://olivewheels.azureedge.net/test onnxruntime-olive==0.1.0
WORKDIR /mnt
