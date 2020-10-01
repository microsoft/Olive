# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
FROM python:3.6

COPY requirements.txt requirements.txt
COPY src/onnx_converter.py src/onnx_converter.py
COPY src/check_model.py src/check_model.py
COPY src/create_input.py src/create_input.py

# Install OpenMp dependency
# ARG OPENMPI_VERSION=1.10.7
# RUN wget -q -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/openmpi/$OPENMPI_VERSION/openmpi-$OPENMPI_VERSION.tar.gz | tar -xzf - && \
#     cd openmpi-${OPENMPI_VERSION} && \
#     apt-get -y update && \
#     apt-get -y -f install && \
#     apt-get -y install libsysfs2 libsysfs-dev && \
#     ./configure --prefix=/usr/local/mpi && \
#     make -j $(nproc) install && \
#     cd .. && \
#     rm -rf openmpi-${OPENMPI_VERSION}

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y -f install && \
    apt-get -y install openmpi-bin

# Install ImageMagick
 #apt-get install build-essential libx11-dev libxext-dev zlib1g-dev libjpeg-dev libfreetype6-dev libxml2-dev && \
RUN wget -O - http://www.imagemagick.org/download/ImageMagick.tar.gz | tar -xzf - && \
    cd ImageMagick-* && \
    ./configure --prefix=/usr/local/imagemagick && \
    make -j $(nproc) && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf ImageMagick-*

RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.40 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libmpi.so.40.10.3 /usr/lib/x86_64-linux-gnu/libmpi.so.12
ENV PATH /usr/local/mpi/bin:$PATH
ENV PATH /usr/local/imagemagick/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/mpi/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install Python Requirements
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

EXPOSE 8888

# Start program
ENTRYPOINT ["python", "src/onnx_converter.py"]