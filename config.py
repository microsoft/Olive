CONTAINER_NAME = 'ziylregistry.azurecr.io/'
#CONVERTED_MODEL = 'converted_model.onnx'
CONVERTED_MODEL = 'model.onnx'


MOUNT_PATH = '/mnt/model'

FUNC_NAME = {
    'onnx_convert': 'onnx-converter',
    'create_input': 'create-input',
    'perf_test':'perf_test'
}