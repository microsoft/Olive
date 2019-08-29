import os
import os.path as osp
import docker
import config as docker_config
import json
import posixpath
import pandas as pd
import functools 
from collections import OrderedDict
import csv

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
        self.convert_path = posixpath.join(self.convert_directory, 
                            self.convert_name)
        self.none_params = {'convert_json', 'runtime', 'windows'} 
                            # no need to write into json
        self.output = ""
    
    def __join_with_mount(self, path):
        if path[:len(self.mount_path)] == self.mount_path: return path
        if path: path = self.win_path_to_linux_relative(osp.relpath(path))
        return posixpath.join(self.mount_path, path)

    def __params2args(self, argu_dict, params):
        arguments = ""
        for p in params:
            if argu_dict[p] is not None:
                if p not in self.none_params:
                    arguments += docker_config.arg(p, argu_dict[p])
        return arguments

    def convert_model(self, model_type=None, output_onnx_path=None, 
        model="", model_params=None, model_input_shapes=None, target_opset=None, 
        caffe_model_prototxt=None, initial_types=None, model_inputs_names=None, model_outputs_names=None,
        input_json=None, convert_json=False, windows=False):

        # is Windows
        if os.name == 'nt':
            windows = True

        def mount_parameters(output_onnx_path, model, caffe_model_prototxt, input_json):
            # --output_onnx_path
            if output_onnx_path is None or output_onnx_path == '':
                output_onnx_path = self.convert_path

            output_onnx_path = self.__join_with_mount(output_onnx_path)

            # --model
            model = self.__join_with_mount(model)
            # --caffe_model_prototxt
            if caffe_model_prototxt is not None:
                caffe_model_prototxt = self.__join_with_mount(caffe_model_prototxt)
                
            if input_json is not None:
                input_json = self.__join_with_mount(input_json)
            return output_onnx_path, model, caffe_model_prototxt, input_json
        
        if model_type is None and input_json is None:
            raise RuntimeError('The conveted model type needs to be provided.')

        img_name = (docker_config.CONTAINER_NAME + 
            docker_config.FUNC_NAME['onnx_converter'] + ':latest')

        # --input_json
        if input_json is not None:
            local_input_json = input_json

        output_onnx_path, model, caffe_model_prototxt, input_json = mount_parameters(
            output_onnx_path, model, caffe_model_prototxt, input_json)


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


        parameters = self.convert_model.__code__.co_varnames[1:self.convert_model.__code__.co_argcount]
        arguments = self.__params2args(locals(), parameters)
        
        # convert the input parameters into json file
        if convert_json:
            self.__convert_input_json(arguments, json_filename)
            arguments = docker_config.arg('input_json', input_json)
        # load by JSON file
        elif input_json is not None:
            with open(posixpath.join(self.path, local_input_json), 'r') as f:
                json_data = json.load(f)

                if 'output_onnx_path' in json_data and (json_data['output_onnx_path'] is not None and json_data['output_onnx_path'] != ''):
                    output_onnx_path = json_data['output_onnx_path']
                    self.convert_path = output_onnx_path
                    params = mount_parameters(
                        output_onnx_path, model, caffe_model_prototxt, input_json)
                    output_onnx_path = params[0]
                if 'model' in json_data:
                    model = json_data['model']
                if 'caffe_model_prototxt' in json_data:
                    caffe_model_prototxt = json_data['caffe_model_prototxt']
                # convert to mount path
                _, model, caffe_model_prototxt, _ = mount_parameters(
                    output_onnx_path, model, caffe_model_prototxt, input_json)
                # write back to JSON
                json_data['output_onnx_path'] = output_onnx_path
                if 'model' in json_data:
                    json_data['model'] = model
                if 'caffe_model_prototxt' in json_data:
                    json_data['caffe_model_prototxt'] = caffe_model_prototxt


            with open(posixpath.join(self.path, local_input_json), 'w') as f:
                json.dump(json_data, f)

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': self.mount_path, 'mode': 'rw'}},
            detach=True)
        if self.print_logs: self.__print_docker_logs(stream, windows)
        
        return output_onnx_path

    def perf_test(self, model=None, result=None, config=None, mode=None, execution_provider=None,
        repeated_times=None, duration_times=None, threadpool_size=None, num_threads=None, top_n=None, 
        parallel=None, runtime=True, input_json=None, convert_json=False, windows=False):
        
        # is Windows, there is no runtime
        if os.name == 'nt':
            runtime = False
            windows = True

        def mount_parameters(model, result, input_json):
            # --model
            if model is None:
                model = self.convert_path
            model = self.__join_with_mount(model)

            result = self.__join_with_mount(self.result)

            # --input_json
            if input_json is not None:
                input_json = self.__join_with_mount(input_json)

            return model, result, input_json

        json_filename = input_json
        # --input_json
        if input_json is not None:
            local_input_json = input_json
        
        if result is not None:
            self.result = result

        model, result, input_json = mount_parameters(model, result, input_json)



        img_name = (docker_config.CONTAINER_NAME + 
            docker_config.FUNC_NAME['perf_test'] + ':latest')


        parameters = self.perf_test.__code__.co_varnames[1:self.perf_test.__code__.co_argcount]
        arguments = self.__params2args(locals(), parameters)
        


        # convert the input parameters into json file
        if convert_json:
            self.__convert_input_json(arguments, json_filename)
            arguments = docker_config.arg('input_json', input_json)
        # load by JSON file
        elif input_json is not None:
            with open(posixpath.join(self.path, local_input_json)) as f:
                json_data = json.load(f)
                if 'result' in json_data:
                    result = json_data['result']
                    if result[:len(self.mount_path)] == self.mount_path:
                        self.result = str(result[len(self.mount_path)+1:])
                    else:
                        self.result = result

                    params = mount_parameters(
                        model, None, input_json)
                    result = params[1]
                if 'runtime' in json_data:
                    runtime = json_data['runtime']
                    
                if 'model' in json_data:
                    model = json_data['model']

                # convert to mount path
                model, _, _ = mount_parameters(
                    model, result, input_json)
                # write back to JSON
                json_data['result'] = result

                if 'model' in json_data:
                    json_data['model'] = model
            with open(posixpath.join(self.path, local_input_json), 'w') as f:
                json.dump(json_data, f)

        runtime = 'nvidia' if runtime else ''

        stream = self.client.containers.run(image=img_name, 
            command=arguments, 
            volumes={self.path: {'bind': self.mount_path, 'mode': 'rw'}},
            runtime=runtime,
            detach=True)
        if self.print_logs: self.__print_docker_logs(stream, windows)

        return posixpath.join(self.path, self.result)

    def __print_docker_logs(self, stream, windows=False):
        logs = stream.logs(stream=True)
        self.output = ""
        for line in logs:
            if type(line) is not str:
                line = line.decode(encoding='UTF-8')
            if windows:
                line = line.replace('\n', '\r\n')
            self.output += line
            if self.print_logs: print(line)


    def __convert_input_json(self, arguments, input_json):
        args = arguments.split('--')
        dictionary = {}
        for argv in args:
            argv = argv.split(' ')
            if argv[0] != "" and argv[0] not in self.none_params:
                dictionary[argv[0]] = argv[1]
        json_path = posixpath.join(self.path, input_json)
        json_string = json.dumps(dictionary)
        with open(json_path, 'w') as f:
            f.write(json_string)
        
    def print_performance(self, result=None):
        if result is None:
            result = posixpath.join(self.path, self.result)
        latency_json = posixpath.join(result, docker_config.LATENCIES_TXT)
        if osp.exists(latency_json):
            with open(latency_json, 'r') as f:
                for line in f:  
                    print(line)
        else:
            raise RuntimeError('Cannot find result directory.')
    
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

    def win_path_to_linux_relative(self, path):
        return osp.relpath(path).replace("\\", "/")


    class Result:
        def __init__(self, result_directory):
            latency_json = osp.join(result_directory, docker_config.LATENCIES_JSON)
            if osp.exists(latency_json):
                with open(latency_json) as json_file:  
                    self.latency = json.load(json_file, object_pairs_hook=OrderedDict) 
            else:
                raise RuntimeError('Cannot find result directory.')
            
            self.profiling = []
            self.latency.pop("failed", None)
            # print(self.latency)
            # Print profiling results for every execution provider
            for ep_name in self.latency.keys():
                # print("key ", ep_name)
                for pf in self.latency[ep_name]:
                    profiling_name = "profile_" + pf["name"] + ".json"
                    profiling_path = osp.join(result_directory, profiling_name)
                    if osp.exists(profiling_path):
                        with open(profiling_path) as json_file:
                            self.profiling.append(json.load(json_file))
            self.profiling_max = 7
            # for i in range(self.profiling_max):
            #     profiling_name = "profile_" + self.latency[i]["name"] + ".json"
            #     profiling_path = osp.join(result_directory, profiling_name)
            #     if osp.exists(profiling_path):
            #         with open(profiling_path) as json_file:
            #             self.profiling.append(json.load(json_file))
            self.profiling_ops = self.__filter_ops()

        def __filter_ops(self):
            profiling_ops = []
            for index in range(self.profiling_max):
                ops = []
                if index < len(self.profiling):
                    for p in self.profiling[index]:
                        if p['cat'] == 'Node':
                            filtered_op = p
                            filtered_op['name'] = p['name'].replace('_kernel_time', '')
                            ops.append(filtered_op)
                    ops.sort(key=lambda x: x['dur'], reverse=True)
                    profiling_ops.append(ops)
            return profiling_ops


        def __print_json(self, json_data, orient):
            data = json.dumps(json_data)
            return pd.read_json(data, orient=orient, precise_float=True)
            
        def __check_profiling_index(self, i):
            if i > self.profiling_max:
                raise ValueError('Only provide top {} profiling.'.format(self.profiling_max))
        
        def __json_to_csv(self, json_data, erase_keys):
    
            csv_name = 'temp.csv'
            with open(csv_name, 'w') as f:
                writer = csv.writer(f)
                keys = []
                for key in json_data[0]:
                    if key not in erase_keys:
                        keys.append(key)
                writer.writerow(keys)
                for j in json_data:
                    values = []
                    for key in j:
                        if key not in erase_keys:
                            values.append(j[key])
                    writer.writerow(values)
            return csv_name

        def prints(self, top=5, orient='table'):
            json_data = []
            for ep_name in self.latency.keys():
                json_data.append(self.latency[ep_name][0])
            csv_file = self.__json_to_csv(json_data, {'command'})
            
            return pd.read_csv(csv_file) 
        
        def print_profiling(self, index=0, top=10, orient='colums'):
            self.__check_profiling_index(index)
            return self.__print_json(self.profiling_ops[index][:top], orient)

        def print_environment(self, ep, index, orient='index'):
            return self.__print_json([self.latency[ep][index]['code_snippet']['environment_variables']], orient)

        def get_code(self, ep, index=0):
            code = self.latency[ep][index]['code_snippet']['code']
            refined_code = code.replace('                 ', '\n').replace('                ', '\n') # 4 tabs
            return refined_code
