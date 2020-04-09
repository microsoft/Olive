:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.
:: pkill -9 -f 'celery worker'
:: ps auxww | awk '/celery worker/ {print $2}' | xargs kill
FOR /F "tokens=5 delims= " %%P IN ('netstat -a -n -o ^| findstr :8000') DO taskkill /PID %%P /F
FOR /F "tokens=5 delims= " %%P IN ('netstat -a -n -o ^| findstr :5555') DO taskkill /PID %%P /F
docker container stop redis
docker container rm redis