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
from shutil import copyfile, rmtree


# app_configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app)

# reserve an input folder
RESERVED_INPUT_PATH = './inputs'

def get_params(request):

    temp_json = 'temp.json'
    model_name = None

    request.files['metadata'].save(temp_json)
    
    with open(temp_json, 'r') as f:
        json_data = json.load(f)

    if not os.path.exists(RESERVED_INPUT_PATH):
        os.mkdir(RESERVED_INPUT_PATH)

    print(request.files)
    if 'file' in request.files:
        temp_model = request.files['file']
        model_name = os.path.join(RESERVED_INPUT_PATH, temp_model.filename)
        print(temp_model)
        request.files['file'].save(model_name)
        json_data['model'] = model_name
    if 'test_data' in request.files:
        # Upload test data
        test_data_dir = os.path.join(RESERVED_INPUT_PATH, 'test_data_set_0')
        if not os.path.exists(test_data_dir):
            os.mkdir(test_data_dir)
        for td in request.files['test_data']:
            print(td)
            request.files['test_data'].save(os.path.join(test_data_dir, td.filename))

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

        netron.start(model_name, browse=False)

    return jsonify(response_object)

@app.route('/convert', methods=['POST'])
def convert():
    response_object = {'status': 'success'}
    model_name, temp_json = get_params(request)

    pipeline = onnxpipeline.Pipeline()

    # may not work in Unix OS cuz the permission
    if os.path.exists(pipeline.convert_directory):
        rmtree(pipeline.convert_directory)

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
    input_root = os.path.join(app_config.STATIC_DIR, target_dir)
    # compress input directory
    compress_path = os.path.join(pipeline.convert_directory, app_config.INPUT_DIR)
    input_path = os.path.join(input_root)
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    tar = tarfile.open(os.path.join(input_path, app_config.COMPRESS_NAME), "w:gz")
    try:
        tar.add(compress_path, arcname=app_config.INPUT_DIR)
    except:
        # fail to generate input
        pass

    tar.close()


    # copy converted onnx model
    if os.path.exists(pipeline.convert_path):
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
    try:
        r = pipeline.get_result(result)
        response_object['result'] = json.dumps(r.latency)
        response_object['profiling'] = r.profiling_ops
    except RuntimeError:
        pass

    return jsonify(response_object)


if __name__ == '__main__':
    app.run()