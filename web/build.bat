:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.

:: install vue packages
cd frontend
call npm install
cd ..
:: pull docker images and install onnxpipeline dependencies
call ../utils/build.bat
:: install python packages
pip install flask flask-cors redis celery[redis] flower gevent
