npm install -g forever
forever start -c python backend/app.py
npm run --prefix frontend serve
forever stop backend/app.py