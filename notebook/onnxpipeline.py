import os
import os.path as osp
import docker
import config
import json

class Pipeline:
    def __init__(self, local_directory=None, mount_path=config.MOUNT_PATH, print_logs=True, 
        convert_directory=config.TEST_DIRECTORY, convert_name=config.CONVERTED_MODEL_NAME, 
        result=config.RESULT_FILENAME):
        
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
        self.result = result
        self.convert_directory = convert_directory
        self.convert_name = convert_name
        self.convert_path = osp.join(self.convert_directory, self.convert_name)
    
    def convert_model(self, model_type=None, output_onnx_path=None, 
        model="", model_inputs=None, model_outputs=None, model_params=None,
        model_input_shapes=None, target_opset=None, caffe_model_prototxt=None,
        initial_types=None, input_json=None, convert_json=False):

        if model_type is None:
            raise RuntimeError('The conveted model type needs to be provided.')
        img_name = (config.CONTAINER_NAME + 
            config.FUNC_NAME['onnx_converter'] + ':latest')

        # --output_onnx_path
        if output_onnx_path is None:
            output_onnx_path = self.convert_path
        output_onnx_path = osp.join(self.mount_path, output_onnx_path)
        
        # --model
        model = osp.join(self.mount_path, model)
        # --caffe_model_prototxt
        if caffe_model_prototxt is not None:
            caffe_model_prototxt = osp.join(self.mount_path, caffe_model_prototxt)
        # --initial_types
        if initial_types is not None:
            if convert_json:
                initial_types = '[(\'' + initial_types[0] + '\','+initial_types[1]+')]'
            else:
                initial_types = '"[(\'' + initial_types[0] + '\','+initial_types[1]+')]"'

        # create test directory for output
        if self.convert_directory is not None:
            test_path = osp.join(self.path, self.convert_directory)
            if not os.path.exists(test_path):
                os.makedirs(test_path)


        arguments = config.arg('model', model)
        argu_dict = locals()
        parameters = self.convert_model.__code__.co_varnames[1:self.convert_model.__code__.co_argcount]
        for p in parameters:
            if argu_dict[p] is not None:
                if p not in ('convert_json'): # not converter parameters
                    arguments += config.arg(p, argu_dict[p])
        
        json_filename = input_json

        # --input_json
        if input_json is not None:
            input_json = osp.join(self.mount_path, input_json)

        if convert_json:
            self.__convert_input_json(arguments, json_filename)
            arguments = config.arg('input_json', input_json)

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': self.mount_path, 'mode': 'rw'}},
            detach=True)
        if self.print_logs:
            self.__print_docker_logs(stream)
        return output_onnx_path

    def perf_test(self, model=None, result=None):
        
        if model is None:
            model = self.convert_path
        model = osp.join(self.mount_path, model)

        if result is not None:
            self.result = result
        result = osp.join(self.mount_path, self.result)

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

    def __convert_input_json(self, arguments, input_json):
        args = arguments.split('--')
        dictionary = {}
        for argv in args:
            argv = argv.split(' ')
            if argv[0] != "" and argv[0] != 'input_json':
                dictionary[argv[0]] = argv[1]
        json_path = osp.join(self.path, input_json)
        json_string = json.dumps(dictionary)
        with open(json_path, 'w') as f:
            f.write(json_string)
        
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
        print("        Converted directory path: {}".format(self.convert_directory))
        print("        Converted model filename: {}".format(self.convert_name))
        print("            Converted model path: {}".format(self.convert_path))
        print("        Print logs in the docker: {}".format(self.print_logs))
