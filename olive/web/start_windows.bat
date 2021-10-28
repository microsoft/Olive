:: Start backend broker
docker run -d -p 6379:6379 --name redis redis

:: Start Celery distributed task queue and monitor
cd backend
start celery -A app.celery worker  --pool=solo -l info
start celery -A  app.celery flower --port=5555

python app.py