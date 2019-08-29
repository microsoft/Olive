Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# OLive (ONNX Live) Web App

This repository shows how to use ONNX pipeline by a web interface in built local server.

# Prerequisites
Install [Docker](https://docs.docker.com/install/).

Install project dependencies by running 
```bash
sh build.sh
```

# Start the Web App
Open front-end server.
```
npm run --prefix frontend serve
```

In a separate command prompt, open back-end server.
```
python backend/app.py
```

Alternatively, on Windows you can run 
```
sh start-windows.sh
```

Then the local server has been built on http://localhost:8000/
