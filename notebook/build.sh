apt install docker.io
# login to docker
docker login ziylregistry.azurecr.io -u ziylregistry -p FCbWbDfHTP86p=DZJwPXXs7/q8iZPkI8
# pull 2 docker images
docker pull ziylregistry.azurecr.io/onnx-converter && docker pull ziylregistry.azurecr.io/perf-test