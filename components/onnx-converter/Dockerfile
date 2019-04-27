FROM python:3.6

COPY requirements.txt requirements.txt
COPY src/onnx-converter.py src/onnx-converter.py

# Install Python Requirements
RUN pip3 install -r requirements.txt

RUN git clone --recursive https://github.com/cjlin1/libsvm libsvm
RUN cd libsvm
RUN make lib
RUN cd ..

EXPOSE 8888

# Launch Jupyter notebook
ENTRYPOINT ["python", "src/onnx-converter.py"]