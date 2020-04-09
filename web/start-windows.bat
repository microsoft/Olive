:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.

:: Start backend broker
docker run -d -p 6379:6379 --name redis redis 
:: Start frontend. Listen on port 8000.
start npm run --prefix frontend serve
:: Start Celery distributed task queue and monitor
cd backend
start celery flower -A app.celery --port=5555
start celery worker -A app.celery -P gevent
cd ..

:: Start Python backend for OLive
python ./backend/app.py