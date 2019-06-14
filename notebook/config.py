import os.path as osp

CONTAINER_NAME = 'ziylregistry.azurecr.io/'

CONVERTED_MODEL_NAME = 'model.onnx' # need to be under a clean directory for perf_test
TEST_DIRECTORY = 'test'

#CONVERTED_MODEL = osp.join(TEST_DIRECTORY, CONVERTED_MODEL_NAME) # need to be under a clean directory for perf_test

RESULT_FILENAME = 'result.txt'

MOUNT_PATH = '/mnt/model'

#MOUNT_MODEL = osp.join(MOUNT_PATH, CONVERTED_MODEL)
#PERF_TEST_RESULT = osp.join(MOUNT_PATH, RESULT_FILENAME)


FUNC_NAME = {
    'onnx_converter': 'onnx-converter',
    'perf_test':'perf_test'
}

def arg(flag, var):
    return ' --' + flag + ' ' + var