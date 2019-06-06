import os.path as osp

CONTAINER_NAME = 'ziylregistry.azurecr.io/'

CONVERTED_MODEL = 'model.onnx'
RESULT_FILENAME = 'result.txt'

MOUNT_PATH = '/mnt/model'

MOUNT_MODEL = osp.join(MOUNT_PATH, CONVERTED_MODEL)
PERF_TEST_RESULT = osp.join(MOUNT_PATH, RESULT_FILENAME)


FUNC_NAME = {
    'onnx_converter': 'onnx-converter',
    'perf_test':'perf_test'
}

def arg(flag, var):
    return ' --' + flag + ' ' + var