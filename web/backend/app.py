from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid, sys, json
sys.path.append('../../notebook')
import onnxpipeline
from werkzeug.utils import secure_filename
import netron


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app)

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
@app.route('/convert', methods=['POST'])
def convert():
    response_object = {'status': 'success'}
    model_name, temp_json = get_params(request)

    pipeline = onnxpipeline.Pipeline()
    model = pipeline.convert_model(model=model_name, input_json=temp_json)

    response_object['logs'] = pipeline.output
    response_object['converted_model'] = model

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
    return jsonify(response_object)




if __name__ == '__main__':
    app.run()