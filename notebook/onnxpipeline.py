import os
import os.path as osp
import docker
import config
import IPython

class Pipeline:
    def __init__(self, local_directory=None, mount_path=config.MOUNT_PATH, print_logs=True):
        
        if local_directory is not None and not os.path.isdir(local_directory):
            raise RuntimeError('local_directory needs to be a directory for volume.')
        elif local_directory is None:
            self.path = os.getcwd()
        elif local_directory[0] == '/':
            self.path = local_directory
        else:
            self.path = osp.join(os.getcwd(), local_directory)
        self.client = docker.from_env()
        self.print_logs = print_logs
        self.mount_path = mount_path
        self.result = config.RESULT_FILENAME
    
    def convert_model(self, model_type=None, output_onnx_path=config.MOUNT_MODEL, 
        model="", model_inputs=None, model_outputs=None, model_params=None,
        model_input_shapes=None, target_opset=None, caffe_model_prototxt=None,
        initial_types=None, input_json=None):

        if model_type is None:
            raise RuntimeError('The conveted model type needs to be provided.')
        img_name = (config.CONTAINER_NAME + 
            config.FUNC_NAME['onnx_converter'] + ':latest')
        
        # --model
        model = osp.join(self.mount_path, model)
        # --caffe_model_prototxt
        if caffe_model_prototxt is not None:
            caffe_model_prototxt = osp.join(self.mount_path, caffe_model_prototxt)
        # --initial_types
        if initial_types is not None:
            initial_types = '"[(\'' + initial_types[0] + '\','+initial_types[1]+')]\"'

        # --input_json
        if input_json is not None:
            input_json = osp.join(self.mount_path, input_json)

        # create test directory for output
        if config.TEST_DIRECTORY is not None:
            test_path = osp.join(self.path, config.TEST_DIRECTORY)
            if not os.path.exists(test_path):
                os.makedirs(test_path)


        arguments = config.arg('model', model)
        argu_dict = locals()
        parameters = self.convert_model.__code__.co_varnames[1:self.convert_model.__code__.co_argcount]
        for p in parameters:
            if argu_dict[p] is not None:
                arguments += config.arg(p, argu_dict[p])

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': self.mount_path, 'mode': 'rw'}},
            detach=True)
        if self.print_logs:
            self.__print_docker_logs(stream)
        return output_onnx_path

    def perf_test(self, model=None, result=None):
        
        if model is None:
            model = config.CONVERTED_MODEL
        model = osp.join(config.MOUNT_PATH, model)

        if result is not None:
            self.result = result
        result = osp.join(config.MOUNT_PATH, self.result)

        img_name = (config.CONTAINER_NAME + 
            config.FUNC_NAME['perf_test'] + ':latest')
        arguments = model + ' ' + result

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': self.mount_path, 'mode': 'rw'}},
            detach=True)
        if self.print_logs:
            self.__print_docker_logs(stream)
        return osp.join(self.path, self.result)

    def __print_docker_logs(self, stream):
        logs = stream.logs(stream=True)
        for line in logs:
            print(line)

    def print_result(self, result=None):
        if result is None:
            result = osp.join(self.path, self.result)
        with open(result, 'r') as f:
            for line in f:  
                print(line)
    
    def config(self):
        print("-----------Config----------------")
        print("           Container information: {}".format(self.client))
        print(" Local directory path for volume: {}".format(self.path))
        print("Volume directory path in dockers: {}".format(self.mount_path))
        print("                     Result path: {}".format(self.result))
        print("        Print logs in the docker: {}".format(self.print_logs))