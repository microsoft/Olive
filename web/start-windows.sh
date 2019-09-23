# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
docker run -d -p 6379:6379 redis
# start python ./backend/worker.py
start python ./backend/app.py
start npm run --prefix frontend serve
