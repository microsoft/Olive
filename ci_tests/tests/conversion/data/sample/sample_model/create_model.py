import os
from ci_tests.tests.conversion.data.sample.sample_model.sample_model import save_entire_model

save_entire_model(os.path.join(os.path.dirname(__file__), 'sample_model.pth'))
