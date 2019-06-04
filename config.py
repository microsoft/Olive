import os.path as osp

CONTAINER_NAME = 'ziylregistry.azurecr.io/'
#CONVERTED_MODEL = 'converted_model.onnx'
CONVERTED_MODEL = 'model.onnx'
MOUNT_PATH = '/mnt/model'
MOUNT_MODEL = osp.join(MOUNT_PATH, CONVERTED_MODEL)


FUNC_NAME = {
    'onnx_converter': 'onnx-converter',
    'create_input': 'create-input',
    'perf_test':'perf_test'
}

def arg(flag, var):
    return ' --' + flag + ' ' + var