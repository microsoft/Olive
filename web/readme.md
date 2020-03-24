Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# OLive (ONNX Go Live) Web App

This repository shows how to use ONNX pipeline by a web interface in built local server. Note: Job scheduling is not supported yet. 

# Prerequisites
- Install [Docker](https://docs.docker.com/install/).

    Note if you're running Docker 19.03 and up, and would like to use GPU, run 

    ```bash
    apt-get install nvidia-docker2
    ``` 
    
    to support `runtime=nvidia` API from [docker python SDK](https://github.com/docker/docker-py). `--runtime=nvidia` has been replaced by `--gpus all` since Docker 19.03 and as of now the Docker python SDK hasn't reflected this change. 

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


