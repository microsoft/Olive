# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import uuid, sys, json
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../utils'))
import onnxpipeline
from werkzeug.utils import secure_filename
import netron
import posixpath
import tarfile
import app_config 
from shutil import copyfile, rmtree
from datetime import datetime


# app_configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app)

# reserve an base folder
# BASE_DIR = os.path.join(app.root_path, app_config.STATIC_DIR)

def get_params(request, convert_output_path, mount_path):

    temp_json = 'temp.json'
    model_name = None
    # Get input parameters from request and save them to json file
    if os.path.exists(temp_json):
        os.remove(temp_json)
    request.files['metadata'].save(temp_json)
    with open(temp_json, 'r') as f:
        json_data = json.load(f)

    file_input_dir = os.path.join(app_config.FILE_INPUTS_DIR, get_timestamp())
    if len(convert_output_path) == 0:
        file_input_dir = get_local_mounted_path(json_data['model'], mount_path)
    
    if not os.path.exists(file_input_dir):
        os.makedirs(file_input_dir)

    # Save files passed from request
    if 'file' in request.files:
        temp_model = request.files['file']
        model_name = os.path.join(file_input_dir, temp_model.filename)
        request.files['file'].save(model_name)
        json_data['model'] = model_name
    if 'test_data[]' in request.files:
        # Create folder to hold test data
        test_data_dir = os.path.join(file_input_dir, 'test_data_set_0')
        if os.path.exists(test_data_dir):
            rmtree(test_data_dir)
        os.mkdir(test_data_dir)
        # Upload test data
        for td in request.files.getlist('test_data[]'):
            td.save(os.path.join(test_data_dir, td.filename))
    if 'savedModel[]' in request.files:
        # Upload test data
        variables_dir = os.path.join(file_input_dir, 'variables')
        if not os.path.exists(variables_dir):
            os.mkdir(variables_dir)
        for vf in request.files.getlist('savedModel[]'):
            vf.save(os.path.join(variables_dir, vf.filename))
        json_data['model'] = os.path.dirname(os.path.abspath(json_data['model']))
    
    # remove empty input folder
    if len(os.listdir(file_input_dir)) == 0:
        os.rmdir(file_input_dir)

    with open(temp_json, 'w') as f:
        json.dump(json_data, f)
    return model_name, temp_json

def get_timestamp():
    ts = int(datetime.now().timestamp())
    return str(ts)

# Get the local path from a mounted path. e.g. mnt/model/path -> path
def get_local_mounted_path(path, mount_path):
    if len(path) < len(mount_path):
        return path
    return os.path.dirname(path[len(mount_path) + 1:])

# Keep a maximum number of contents in the given directory. Remove the oldest if greater than maximum.
def garbage_collect(dir, max=5):
    if not os.path.exists(dir):
        return
    num_folders = len(os.listdir(dir))
    if num_folders > max:
        mtime = lambda f: os.stat(os.path.join(dir, f)).st_mtime
        sorted_dir = list(sorted(os.listdir(dir), key=mtime))
        del_dir = sorted_dir[0: num_folders - max]
        for folder in del_dir:
            rmtree(os.path.join(dir, folder))

def clean():
    garbage_collect(app_config.FILE_INPUTS_DIR)
    garbage_collect(app_config.CONVERT_RES_DIR)
    garbage_collect(app_config.PERF_RES_DIR)
    garbage_collect(os.path.join(app.root_path, app_config.DOWNLOAD_DIR))

@app.route('/visualize', methods=['POST'])
def visualize():
    
    if request.method == 'POST':
        response_object = {'status': 'success'}
        temp_model = request.files['file']
        model_name = temp_model.filename

        request.files['file'].save(model_name)

        netron.start(model_name, browse=False, host='0.0.0.0')

    return jsonify(response_object)

@app.route('/convert', methods=['POST'])
def convert():
    response_object = {'status': 'success'}

    cur_ts = get_timestamp()
    # Initiate pipeline object with targeted directory
    pipeline = onnxpipeline.Pipeline(convert_directory=os.path.join(app_config.CONVERT_RES_DIR, cur_ts))
    # may not work in Unix OS cuz the permission
    if os.path.exists(pipeline.convert_directory):
        rmtree(pipeline.convert_directory)
    os.makedirs(pipeline.convert_directory)
        
    model_name, temp_json = get_params(request, pipeline.convert_directory, pipeline.mount_path)

    model = pipeline.convert_model(model=model_name, input_json=temp_json)
    try:
        with open(posixpath.join(pipeline.convert_directory, 'output.json')) as f:
            json_data = json.load(f)
            response_object['output_json'] = json_data

        response_object['logs'] = pipeline.output
        response_object['converted_model'] = model
    except:
        #fail to convert
        pass

    target_dir = app_config.DOWNLOAD_DIR
    input_root =  os.path.join(app.root_path, target_dir)
    # compress input directory
    compress_path = os.path.join(pipeline.convert_directory, app_config.TEST_DATA_DIR)
    input_path = os.path.join(input_root, cur_ts)
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    tar = tarfile.open(os.path.join(input_path, app_config.COMPRESS_NAME), "w:gz")
    try:
        tar.add(compress_path, arcname=app_config.TEST_DATA_DIR)
    except:
        # fail to generate input
        pass

    tar.close()


    # copy converted onnx model
    if os.path.exists(pipeline.convert_path):
        copyfile(pipeline.convert_path, os.path.join(input_root, cur_ts, pipeline.convert_name))

    response_object['input_path'] = posixpath.join(target_dir, cur_ts, app_config.COMPRESS_NAME)
    response_object['model_path'] = posixpath.join(target_dir, cur_ts, pipeline.convert_name)

    clean()
    return jsonify(response_object)

@app.route('/perf_tuning', methods=['POST'])
def perf_tuning():

    response_object = {'status': 'success'}

    pipeline = onnxpipeline.Pipeline()
    if 'file' in request.files:
        _, temp_json = get_params(request, os.path.join(app_config.FILE_INPUTS_DIR), pipeline.mount_path)
    else:
        # If model path is provided, add test data if neccessary to the model directory path
        _, temp_json = get_params(request, '', pipeline.mount_path)

    # create result dir
    result_dir = os.path.join(app_config.PERF_RES_DIR, get_timestamp())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result = pipeline.perf_tuning(input_json=temp_json, result=result_dir)

    response_object['logs'] = pipeline.output
    try:
        r = pipeline.get_result(result)
        response_object['result'] = json.dumps(r.latency)
        response_object['profiling'] = r.profiling_ops
    except RuntimeError:
        pass
    # clean up result dir
    if os.path.exists(result_dir):
        rmtree(result_dir)
    clean()
    return jsonify(response_object)

@app.route('/download/<path:filename>', methods=['POST', 'GET'])
def download(filename):
    try:
        path = os.path.join(app.root_path, app_config.DOWNLOAD_DIR)
        return send_file(os.path.join(path, filename), filename)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0')