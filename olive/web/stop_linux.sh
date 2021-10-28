#!/bin/sh

docker container stop redis
docker container rm redis
pkill -9 -f 'celery worker'
kill -9 `sudo lsof -t -i:5000`
kill -9 `sudo lsof -t -i:5555`