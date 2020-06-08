# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
FROM nvidia/cuda:10.2-runtime

RUN apt-get update && apt-get install -y python3.7 python3.7-dev libgomp1 python3-pip 
RUN python3.7 -m pip install psutil gputil numpy onnx packaging sympy

RUN mkdir /perf_tuning /perf_tuning/src
COPY bin /perf_tuning/bin
COPY src/perf_tuning.py /perf_tuning/src
COPY src/monitor.py /perf_tuning/src
COPY src/maps.py /perf_tuning/src

WORKDIR /

ENTRYPOINT [ "python3.7", "/perf_tuning/src/perf_tuning.py"]