# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
FROM nvidia/cuda:10.0-base

RUN apt-get update && apt-get install -y python3.7 python3.7-dev libgomp1 cuda-cublas-10-0 python3-pip 
RUN python3.7 -m pip install psutil gputil numpy onnx packaging

RUN mkdir /perf_tuning
COPY bin /perf_tuning/bin
COPY src/perf_tuning.py /perf_tuning
COPY src/monitor.py /perf_tuning
COPY src/maps.py /perf_tuning

WORKDIR /

ENTRYPOINT [ "python3.7", "/perf_tuning/perf_tuning.py"]