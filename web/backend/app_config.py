# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

# is Windows
if os.name == 'nt':
    STATIC_DIR = 'static'
else:
    STATIC_DIR = 'static'

FILE_INPUTS_DIR = 'io_files/inputs'
TEST_DATA_DIR = 'test_data_set_0'
COMPRESS_NAME = 'input.tar.gz'
DOWNLOAD_DIR = 'static/download'
CONVERT_RES_DIR = 'io_files/convert'
PERF_RES_DIR = 'io_files/perf'