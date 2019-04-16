FROM python:3.6

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY requirements.txt requirements.txt

# Install Python Requirements
RUN pip3 install -r requirements.txt

EXPOSE 8888

# Call tf2onnx to convert Tensorflow model to ONNX
ENTRYPOINT ["python", "src/tf_onnx.py"]