import os


# is Windows
if os.name == 'nt':
    STATIC_DIR = 'static'
else:
    STATIC_DIR = 'static'


INPUT_DIR = 'test_data_set_0'
COMPRESS_NAME = 'input.tar.gz'
DOWNLOAD_DIR = 'download'