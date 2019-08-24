# install vue packages
cd frontend
npm install --only=prod
cd ..
# install python packages
pip install flask flask-cors pandas docker netron
# pull docker images
sh ../notebook/build.sh