# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Start backend broker
docker run -d -p 6379:6379 --name redis redis 
# Start frontend. Listen on port 8000.
npm run --prefix frontend serve &>/dev/null &
# Start Celery distributed task queue and monitor
cd backend
celery flower -A app.celery --port=5555 &>/dev/null &
celery worker -A app.celery -P gevent &>/dev/null &
cd ..
# Start Python backend for OLive
python ./backend/app.py
