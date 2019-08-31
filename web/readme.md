Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# OLive (ONNX Go Live) Web App

This repository shows how to use ONNX pipeline by a web interface in built local server. Note: Job scheduling is not supported yet. 

# Prerequisites
- Install [Docker](https://docs.docker.com/install/).

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
### Windows
On Windows you can run to start both frontend and backend servers
```
start-windows.sh

```

### Linux
Open front-end server.
```
npm run --prefix frontend serve
```

In a separate command prompt, open back-end server.
```
sudo python backend/app.py
```

Then the local server has been built on http://localhost:8000/
