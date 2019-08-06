from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid, sys, json
sys.path.append('../../notebook')
import onnxpipeline
from werkzeug.utils import secure_filename
import netron
import posixpath
import tarfile
import app_config 
import os
from shutil import copyfile


# app_configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app)

def get_params(request):

    temp_json = 'temp.json'
    model_name = None

    request.files['metadata'].save(temp_json)
    
    with open(temp_json, 'r') as f:
        json_data = json.load(f)

    if 'file' in request.files:
        temp_model = request.files['file']
        model_name = temp_model.filename
        request.files['file'].save(model_name)
        json_data['model'] = model_name

    with open(temp_json, 'w') as f:
        json.dump(json_data, f)
    return model_name, temp_json

@app.route('/visualize', methods=['POST'])
def visualize():
    response_object = {'status': 'failure'}
    if request.method == 'POST':
        response_object = {'status': 'success'}
        temp_model = request.files['file']
        model_name = temp_model.filename

        request.files['file'].save(model_name)

        netron.start(model_name)

    return jsonify(response_object)

@app.route('/convert', methods=['POST'])
def convert():
    response_object = {'status': 'success'}
    model_name, temp_json = get_params(request)

    pipeline = onnxpipeline.Pipeline()
    model = pipeline.convert_model(model=model_name, input_json=temp_json)
    with open(posixpath.join(pipeline.convert_directory, 'output.json')) as f:
        json_data = json.load(f)
        response_object['output_json'] = json_data

    response_object['logs'] = pipeline.output
    response_object['converted_model'] = model

    target_dir = app_config.DOWNLOAD_DIR
    input_root = os.path.join(app_config.STATIC_DIR, target_dir)
    # compress input directory
    compress_path = os.path.join(pipeline.convert_directory, app_config.INPUT_DIR)
    input_path = os.path.join(input_root, app_config.COMPRESS_NAME)
    tar = tarfile.open(input_path, "w:gz")
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    tar.add(compress_path, arcname=app_config.INPUT_DIR)
    tar.close()

    # copy converted onnx model
    copyfile(pipeline.convert_path, os.path.join(input_root, pipeline.convert_name))

    response_object['input_path'] = posixpath.join(target_dir, app_config.COMPRESS_NAME)
    response_object['model_path'] = posixpath.join(target_dir, pipeline.convert_name)

    return jsonify(response_object)

@app.route('/perf_test', methods=['POST'])
def perf_test():
    response_object = {'status': 'success'}

    _, temp_json = get_params(request)
    
    pipeline = onnxpipeline.Pipeline()
    result = pipeline.perf_test(input_json=temp_json)

    response_object['logs'] = pipeline.output
    r = pipeline.get_result(result)
    response_object['result'] = r.latency
    response_object['profiling'] = r.profiling
    #response_object['profiling'] = []

    return jsonify(response_object)


if __name__ == '__main__':
    app.run()