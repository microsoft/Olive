npm install -g forever
forever start -c "npm run --prefix frontend serve"
python backend/app.py
forever stop "npm run --prefix frontend serve"