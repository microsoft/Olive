# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install vue packages
cd frontend
npm install
cd ..
# install python packages
pip install flask flask-cors pandas docker netron redis rq rq_dashboard
# pull docker images
sh ../utils/build.sh