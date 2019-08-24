# install vue packages
cd frontend
npm install --only=prod
cd ..
# install python packages
pip install flask docker netron
# pull docker images
sh ../notebook/build.sh