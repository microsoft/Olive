# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
docker run -d -p 6379:6379 redis
start python ./backend/app.py
start npm run --prefix frontend serve
cd backend
start celery flower -A app.celery --port=5555
start celery worker -A app.celery -P gevent
cd ..