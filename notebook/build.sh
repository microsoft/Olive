# # login to docker
# docker login ziylregistry.azurecr.io -u ziylregistry
# pull 2 docker images
docker pull mcr.microsoft.com/onnxruntime/onnx-converter && docker pull mcr.microsoft.com/onnxruntime/perf-tuning
