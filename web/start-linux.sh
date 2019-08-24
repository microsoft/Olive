apt install forever
forever start python backend/app.py
cd frontend
npm run serve
cd ..
forvever stop backend/app.py