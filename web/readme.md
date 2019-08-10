Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.


# OLive (ONNX Live)

This repository shows how to use ONNX pipeline by a web interface in built local server.

# Prerequisites
Install [Docker](https://docs.docker.com/install/).
### For Windows
```bash
build.sh
```

### For Linux
```bash
sh build.sh
```


Open front-end server.
```
cd frontend
npm run dev
```

Open back-end server.
```
cd backend
python app.py
```

Then the local server has been built. Use ONNX pipeline via the website as:
http://localhost:1223/
