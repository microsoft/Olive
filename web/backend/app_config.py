import os


# is Windows
if os.name == 'nt':
    STATIC_DIR = '..\\frontend\\static'
else:
    STATIC_DIR = '../frontend/static'


INPUT_DIR = 'test_data_set_0'
COMPRESS_NAME = 'input.tar.gz'
DOWNLOAD_DIR = 'download'