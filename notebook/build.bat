# login to docker
docker login ziylregistry.azurecr.io -u ziylregistry
# pull 2 docker images
docker pull ziylregistry.azurecr.io/onnx-converter && docker pull ziylregistry.azurecr.io/perf-test