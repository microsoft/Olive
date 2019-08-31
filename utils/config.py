import os.path as osp

CONTAINER_NAME = 'mcr.microsoft.com/onnxruntime/'

CONVERTED_MODEL_NAME = 'model.onnx' # need to be under a clean directory for perf_tuning
TEST_DIRECTORY = 'test'

# it shoud be a directory
RESULT_FILENAME = 'result'

MOUNT_PATH = '/mnt/model'

OUTPUT_JSON = 'output.json'

LATENCIES_TXT = 'latencies.txt'
LATENCIES_JSON = 'latencies.json'

FUNC_NAME = {
    'onnx_converter': 'onnx-converter',
    'perf_tuning':'perf-tuning'
}

def arg(flag, var):
    # Careful, this is for perf-tuning
    if len(flag) == 1:
        return ' -' + flag + ' ' + str(var)
    else:
        return ' --' + flag + ' ' + str(var)

