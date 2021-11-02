#!/bin/sh
# Start backend broker
docker run -d -p 6379:6379 --name redis redis

# Start Celery distributed task queue and monitor
cd backend
celery -A app.celery worker  --pool=solo -l info &>/dev/null &
celery -A  app.celery flower --port=5555 &>/dev/null &

# Start Python backend for OLive
python app.py
