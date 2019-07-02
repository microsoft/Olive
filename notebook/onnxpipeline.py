import os
import os.path as osp
import docker
import config as docker_config
import json
import posixpath
import pandas as pd


class Pipeline:
    def __init__(self, local_directory=None, mount_path=docker_config.MOUNT_PATH, print_logs=True, 
        convert_directory=docker_config.TEST_DIRECTORY, convert_name=docker_config.CONVERTED_MODEL_NAME, 
        result=docker_config.RESULT_FILENAME):
        
        if local_directory is not None and not os.path.isdir(local_directory):
            raise RuntimeError('local_directory needs to be a directory for volume.')
        elif local_directory is None:
            self.path = os.getcwd()
        elif local_directory[0] == '/':
            self.path = local_directory
        else:
            self.path = posixpath.join(os.getcwd(), local_directory)
        self.client = docker.from_env()
        self.print_logs = print_logs
        self.mount_path = mount_path
        self.result = result
        self.convert_directory = convert_directory
        self.convert_name = convert_name
        self.convert_path = posixpath.join(self.convert_directory, self.convert_name)
    
    def convert_model(self, model_type=None, output_onnx_path=None, 
        model="", model_params=None, model_input_shapes=None, target_opset=None, 
        caffe_model_prototxt=None, initial_types=None, input_json=None, convert_json=False,
        model_inputs_names=None, model_outputs_names=None):

        if model_type is None:
            raise RuntimeError('The conveted model type needs to be provided.')
        img_name = (docker_config.CONTAINER_NAME + 
            docker_config.FUNC_NAME['onnx_converter'] + ':latest')

        # --output_onnx_path
        if output_onnx_path is None:
            output_onnx_path = self.convert_path
        output_onnx_path = posixpath.join(self.mount_path, output_onnx_path)

        # --model
        model = posixpath.join(self.mount_path, model)
        # --caffe_model_prototxt
        if caffe_model_prototxt is not None:
            caffe_model_prototxt = posixpath.join(self.mount_path, caffe_model_prototxt)
        # --initial_types
        if initial_types is not None:
            if convert_json:
                initial_types = '[(\'' + initial_types[0] + '\','+initial_types[1]+')]'
            else:
                initial_types = '"[(\'' + initial_types[0] + '\','+initial_types[1]+')]"'

        # create test directory for output
        if self.convert_directory is not None:
            test_path = posixpath.join(self.path, self.convert_directory)
            if not os.path.exists(test_path):
                os.makedirs(test_path)

        json_filename = input_json

        # --input_json
        if input_json is not None:
            local_input_json = input_json
            input_json = posixpath.join(self.mount_path, input_json)

        #arguments = docker_config.arg('model', model)
        arguments = ""
        argu_dict = locals()
        parameters = self.convert_model.__code__.co_varnames[1:self.convert_model.__code__.co_argcount]
        for p in parameters:
            if argu_dict[p] is not None:
                if p not in ('convert_json'): # not converter parameters
                    arguments += docker_config.arg(p, argu_dict[p])
        
        if convert_json:
            self.__convert_input_json(arguments, json_filename)
            arguments = docker_config.arg('input_json', input_json)
        elif input_json is not None:
            with open(posixpath.join(self.path, local_input_json)) as F:
                json_data = json.load(F)
                if 'output_onnx_path' in json_data:
                    output_onnx_path = json_data['output_onnx_path']
                    self.convert_path = output_onnx_path

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': self.mount_path, 'mode': 'rw'}},
            detach=True)
        if self.print_logs:
            self.__print_docker_logs(stream)

        return output_onnx_path

    def perf_test(self, model=None, result=None, config=None, m=None, e=None,
        r=None, t=None, x=None, n=None, s=None, runtime=True):
        
        if model is None:
            model = self.convert_path
        model = posixpath.join(self.mount_path, model)

        if result is not None:
            self.result = result

        runtime = 'nvidia' if runtime else ''
        result = posixpath.join(self.mount_path, self.result)

        img_name = (docker_config.CONTAINER_NAME + 
            docker_config.FUNC_NAME['perf_test'] + ':latest')

        arguments = ""
        argu_dict = locals()
        parameters = self.perf_test.__code__.co_varnames[1:self.perf_test.__code__.co_argcount]
        for p in parameters:
            if argu_dict[p] is not None:
                if p not in ('runtime'): # not perf-test parameters
                    arguments += docker_config.arg(p, argu_dict[p])

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': self.mount_path, 'mode': 'rw'}},
            runtime=runtime,
            detach=True)
        if self.print_logs:
            self.__print_docker_logs(stream)
        return posixpath.join(self.path, self.result)

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
        json_path = posixpath.join(self.path, input_json)
        json_string = json.dumps(dictionary)
        with open(json_path, 'w') as f:
            f.write(json_string)
        
    def print_performance(self, result=None):
        if result is None:
            result = posixpath.join(self.path, self.result)
        with open(posixpath.join(result, docker_config.LATENCIES_TXT), 'r') as f:
            for line in f:  
                print(line)
    
    def get_result(self, result=None):
        if result is None:
            result = posixpath.join(self.path, self.result)
        return Pipeline.Result(result)
    
    def config(self):
        print("-----------config----------------")
        print("           Container information: {}".format(self.client))
        print(" Local directory path for volume: {}".format(self.path))
        print("Volume directory path in dockers: {}".format(self.mount_path))
        print("                     Result path: {}".format(self.result))
        print("        Converted directory path: {}".format(self.convert_directory))
        print("        Converted model filename: {}".format(self.convert_name))
        print("            Converted model path: {}".format(self.convert_path))
        print("        Print logs in the docker: {}".format(self.print_logs))

    class Result:
        def __init__(self, result_directory):
            latency_json = osp.join(result_directory, docker_config.LATENCIES_JSON)
            with open(latency_json) as json_file:  
                self.latency = json.load(json_file)            
            self.profiling_max = 5
            self.profiling = []
            for i in range(self.profiling_max):
                profiling_name = "profile_" + self.latency[i]["name"] + ".json"
                with open(osp.join(result_directory, profiling_name)) as json_file:  
                    self.profiling.append(json.load(json_file))

        def __print_json(self, json_data, orient):
            data = json.dumps(json_data)
            return pd.read_json(data, orient=orient, precise_float=True)
            
        def __check_profiling_index(self, i):
            if i > self.profiling_max:
                raise ValueError('Only provide top {} profiling.'.format(self.profiling_max))

        def prints(self, top=5, orient='colums'):
            return self.__print_json(self.latency[:top], orient)
        
        def print_profiling(self, i, orient='colums'):
            self.__check_profiling_index(i)
            unfold_profiling_list = []
            for p in self.profiling[i]:
                unfold_profiling = {}
                for key in p:
                    if key == 'args':
                        for args_key in p[key]:
                            unfold_profiling[args_key] = p[key][args_key]
                    else:
                        unfold_profiling[key] = p[key]
                unfold_profiling_list.append(unfold_profiling)
            return self.__print_json(unfold_profiling_list, orient)
        """
        def print_profiling_args(self, i, orient='index'):
            self.__check_profiling_index(i)
            return self.__print_json([p['args'] for p in self.profiling[i]], orient)
        """

        def print_environment(self, i, orient='index'):
            return self.__print_json([self.latency[i]['code_snippet']['environment_variables']], orient)

        def get_code(self, i):
            code = self.latency[i]['code_snippet']['code']
            refined_code = code.replace('                 ', '\n').replace('                ', '\n') # 4 tabs
            return refined_code
            