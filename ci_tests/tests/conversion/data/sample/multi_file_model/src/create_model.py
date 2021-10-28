import os
from ci_tests.tests.conversion.data.sample.multi_file_model.src.multi_file_model import save_model

# execute this script following these instructions:
# 1. python
# 2. import sys; sys.path.append('./ci_tests/tests/conversion/data/sample/multi_file_model/src'); import create_model
# 3. find pth file in data folder
save_model(os.path.join(os.path.dirname(__file__), '..', 'data', 'multi_file_model.pth'))
