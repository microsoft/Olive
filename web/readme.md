Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# OLive (ONNX Go Live) Web App

This repository shows how to use ONNX pipeline by a web interface in built local server. Note: Job scheduling is not supported yet. 

# Prerequisites
- Install [Docker](https://docs.docker.com/install/).

- Install [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)

    Note that NVIDIA docker 2 is needed to support `runtime=nvidia` API from [Docker python SDK](https://github.com/docker/docker-py), which this web app depends on. `--runtime=nvidia` has been replaced by `--gpus all` since Docker 19.03 and as of now the Docker python SDK hasn't reflected this change. 

- Install project dependencies by running 
## Windows
```bash
build.sh
```

## Linux
```bash
sh build.sh
```

# Start the Web App
First make sure your docker daemon is running. Then, 

### Windows
On Windows you can run to start both frontend and backend servers
```
start-windows.sh
```
You can then access the web app at http://localhost:8000/ 

### Linux
To start, run
```
sudo sh start-linux.sh
```
This will start both backend and frontend servers in the background. You then can access the web app at http://localhost:8000/

To stop the servers, run
```
sudo sh stop-linux.sh
```


