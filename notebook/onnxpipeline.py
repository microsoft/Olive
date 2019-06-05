import os
import os.path as osp
import docker
import config

class Onnxpip:
    def __init__(self, directory=None):
        if directory is None:
            raise RuntimeError('Please provide the path for mounting  in local.')

        self.path = directory if directory[0] == '/' else osp.join(os.getcwd(), directory)
        self.client = docker.from_env()
    
    def convert_model(self, output_onnx_path=None, model_type=None):
        if model_type is None:
            raise RuntimeError('The conveted model type needs to be provided.')
        if output_onnx_path is None:
            output_onnx_path = config.MOUNT_MODEL

        img_name = (config.CONTAINER_NAME + 
            config.FUNC_NAME['onnx_converter'] + ':latest')
        arguments = (config.arg('model', config.MOUNT_PATH) +
            config.arg('output_onnx_path', config.MOUNT_MODEL) +
            config.arg('model_type', model_type)
            )

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': config.MOUNT_PATH, 'mode': 'rw'}},
            detach=True)
        self.print_docker_logs(stream)

    def perf_test(self, result=None):
        if result is None:
            result = config.PERF_TEST_RESULT

        img_name = (config.CONTAINER_NAME + 
            config.FUNC_NAME['perf_test'] + ':latest')
        arguments = config.MOUNT_MODEL + ' ' + result

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': config.MOUNT_PATH, 'mode': 'rw'}},
            detach=True)
        self.print_docker_logs(stream)

    def print_docker_logs(self, stream):
        logs = stream.logs(stream=True)
        for line in logs:
            print(line)
