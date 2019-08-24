npm install -g forever
forever start -c python backend/app.py
cd frontend
npm run serve
cd ..
forvever stop backend/app.py