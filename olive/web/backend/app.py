import os
import sys

from flask import Flask, jsonify, request, send_file, url_for, send_from_directory, safe_join
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import onnxpipeline
import netron
import app_config
from shutil import rmtree
import webbrowser
from datetime import datetime
from celery import Celery
import requests
import json
from zipfile import ZipFile
from tarfile import TarFile, is_tarfile

app = Flask(__name__)

app.config.from_object(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['CELERY_TRACK_STARTED'] = True
app.config['CELERY_TASK_RESULT_EXPIRES'] = 0  # Celery job won't expire

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])
celery.conf.update(app.config)

# enable CORS
CORS(app)


def create_input_dir(conversion=False):
    cur_ts = get_timestamp()
    if conversion:
        file_input_dir = os.path.join(app.root_path, app_config.CONVERT_DIR, cur_ts)
    else:
        file_input_dir = os.path.join(app.root_path, app_config.PERF_DIR, cur_ts)
    if not os.path.exists(file_input_dir):
        os.makedirs(file_input_dir)
    return file_input_dir


def create_temp_json(path, metadata):
    temp_json = os.path.join(path, 'temp.json')
    if os.path.exists(temp_json):
        os.remove(temp_json)
    metadata.save(temp_json)
    return temp_json


def store_file_from_request(request, file_key, save_to_dir):
    if file_key in request.files:
        if os.path.exists(save_to_dir):
            rmtree(save_to_dir)
        os.mkdir(save_to_dir)
        temp_file = request.files[file_key]
        saved_file_path = os.path.join(save_to_dir, temp_file.filename)
        request.files[file_key].save(saved_file_path)
        return saved_file_path
    return ''


def store_files_from_request(request, file_key, save_to_dir):

    if file_key in request.files:
        if os.path.exists(save_to_dir):
            rmtree(save_to_dir)
        os.mkdir(save_to_dir)
        file_list = request.files.getlist(file_key)
        # Upload test data
        for td in file_list:
            saved_file_path = os.path.join(save_to_dir, td.filename)
            td.save(saved_file_path)


def get_convert_json(request):
    file_input_dir = create_input_dir(conversion=True)
    temp_json = create_temp_json(file_input_dir, request.files['metadata'])
    with open(temp_json, 'r') as f:
        json_data = json.load(f)
        model_framework = json_data["model_framework"]
        if model_framework == "pytorch":
            framework_version = json_data.get("pytorch_version")
        if model_framework == "tensorflow":
            framework_version = json_data.get("tensorflow_version")

    json_data['framework_version'] = framework_version

    # Save files passed from request
    model_name = store_file_from_request(request, 'file', file_input_dir)
    if model_name:
        if is_tarfile(model_name):
            with TarFile.open(model_name) as tar_ref:
                tar_ref.extractall(file_input_dir)
            model_dir_name = os.path.basename(model_name).split(".")[0]
            model_name = os.path.join(file_input_dir, model_dir_name)
        elif model_name.endswith(".zip"):
            with ZipFile(model_name) as zip_ref:
                zip_ref.extractall(file_input_dir)
            model_dir_name = os.path.basename(model_name).split(".")[0]
            model_name = os.path.join(file_input_dir, model_dir_name)
        json_data['model_path'] = model_name

    test_data_dir = os.path.join(file_input_dir, app_config.TEST_DATA_DIR)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    sample_input_data_path = store_file_from_request(request, 'sample_input_data_path', test_data_dir)
    if not sample_input_data_path == "":
        json_data['sample_input_data_path'] = sample_input_data_path

    conversion_config = store_file_from_request(request, 'conversion_config', file_input_dir)
    if not conversion_config == "":
        json_data['conversion_config'] = conversion_config

    with open(temp_json, 'w') as f:
        json.dump(json_data, f)

    return model_name, temp_json[len(app.root_path) + 1:], file_input_dir


def get_perf_json(request):
    file_dir = create_input_dir(conversion=False)

    temp_json = create_temp_json(file_dir, request.files['metadata'])
    with open(temp_json, 'r') as f:
        json_data = json.load(f)

    # Save files passed from request
    model_name = store_file_from_request(request, 'file', file_dir)
    if not model_name == "":
        json_data['model_path'] = model_name

    # Store sanple input data file if any
    test_data_dir = os.path.join(file_dir, app_config.TEST_DATA_DIR)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    sample_input_data_path = store_file_from_request(request, 'sample_input_data_path', test_data_dir)
    if not sample_input_data_path == "":
        json_data['sample_input_data_path'] = sample_input_data_path

    optimization_config = store_file_from_request(request, 'optimization_config', file_dir)
    if not optimization_config == "":
        json_data['optimization_config'] = optimization_config

    with open(temp_json, 'w') as f:
        json.dump(json_data, f)

    return temp_json[len(app.root_path) + 1:], file_dir


def get_timestamp():
    ts = int(datetime.now().timestamp())
    return str(ts)


def get_time_from_ts(time):
    dt = datetime.fromtimestamp(time)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f %Z")


def garbage_collect(dir, max=20):
    if not os.path.exists(dir):
        return
    num_folders = len(os.listdir(dir))
    if num_folders > max:
        mtime = lambda f: os.stat(os.path.join(dir, f)).st_mtime
        sorted_dir = list(sorted(os.listdir(dir), key=mtime))
        del_dir = sorted_dir[0:num_folders - max]
        for folder in del_dir:
            rmtree(os.path.join(dir, folder))


def clean(root_path):
    garbage_collect(os.path.join(root_path, app_config.CONVERT_DIR))
    garbage_collect(os.path.join(root_path, app_config.PERF_DIR))


@app.route('/visualize', methods=['POST'])
def visualize():

    if request.method == 'POST':
        response_object = {'status': 'success'}
        temp_model = request.files['file']
        model_name = temp_model.filename

        request.files['file'].save(model_name)

        netron.start(model_name, browse=False)

    return jsonify(response_object)


@celery.task(bind=True)
def convert(self, temp_json, file_input_dir):
    response_object = {'status': 'success'}
    # Initiate pipeline object with targeted directory
    pipeline = onnxpipeline.Pipeline()
    onnx_model = pipeline.convert_model(input_json=temp_json, convert_directory=file_input_dir)
    if onnx_model:
        response_object['converted_model'] = onnx_model
        response_object['conversion_status'] = "SUCCESS"
    else:
        response_object['conversion_status'] = "FAILED"
        raise Exception("conversion failed\n {}".format(pipeline.output))

    response_object['logs'] = pipeline.output
    response_object['converted_model'] = onnx_model
    return response_object


@app.route('/convert', methods=['POST'])
def send_convert_job():
    model_name, temp_json, file_input_dir = get_convert_json(request)
    job_name = "convert." + request.form.get('job_name')
    job = convert.apply_async(args=[temp_json, file_input_dir], shadow=job_name)
    return jsonify({'Location': url_for('convert_status', task_id=job.id), 'job_id': job.id}), 202


@app.route('/convertstatus/<task_id>')
def convert_status(task_id):
    task = convert.AsyncResult(task_id)
    if task.state == 'PENDING' or task.state == 'STARTED':
        # job has not started yet
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'logs': task.info.get('logs', ''),
            'converted_model': task.info.get('converted_model', ''),
            'conversion_status': task.info.get('conversion_status', ''),
            'status': task.info.get('status', '')
        }
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.traceback),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/perf_tuning', methods=['POST'])
def send_perf_job():
    temp_json, file_dir = get_perf_json(request)
    job_name = 'perf_tuning.' + request.form.get('job_name')
    job = perf_tuning.apply_async(args=[temp_json, file_dir], shadow=job_name)
    return jsonify({'Location': url_for('perf_status', task_id=job.id), 'job_id': job.id}), 202


@celery.task(bind=True)
def perf_tuning(self, temp_json, file_dir):
    response_object = {'status': 'success'}

    pipeline = onnxpipeline.Pipeline()

    # create result dir
    result_dir = os.path.join(file_dir, "olive_opt_result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result, optimized_model_path = pipeline.perf_tuning(input_json=temp_json, result_dir=result_dir)
    try:
        r = pipeline.get_result(result, optimized_model_path)
        response_object['result'] = json.dumps(r.__dict__)
    except Exception:
        raise Exception("optimization failed\n {}".format(pipeline.output))

    response_object['logs'] = pipeline.output
    clean(app.root_path)
    return response_object


@app.route('/perfstatus/<task_id>')
def perf_status(task_id):
    task = perf_tuning.AsyncResult(task_id)
    if task.state == 'PENDING' or task.state == 'STARTED':
        # job did not start yet
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'result': task.info.get('result', ''),
            'status': task.info.get('status', ''),
            'logs': task.info.get('logs', ''),
        }
    else:
        response = {
            'state': task.state,
            'traceback': str(task.traceback),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/gettasks')
def get_tasks():
    api_root = 'http://localhost:5555/api'
    task_api = '{}/tasks'.format(api_root)
    resp = requests.get(task_api)
    try:
        reply = resp.json()
        for key in reply:
            reply[key]["received"] = get_time_from_ts(reply[key]["received"])
            reply[key]["started"] = get_time_from_ts(reply[key]["started"])
        return jsonify(reply)
    except:
        pass


@app.route('/getargs/<task_id>')
def get_task_args(task_id):
    api_root = 'http://localhost:5555/api/task/info'
    task_api = '{}/{}'.format(api_root, task_id)
    resp = requests.get(task_api)
    try:
        reply = resp.json()
        args = reply.get('args')
        if len(args) > 0:
            args = args[args.find('{'):len(args) - 1]
            args = args.replace('\\', '\\\\').replace('\'', '"').replace('True', '"true"').replace('False', '"false"')
        return jsonify(json.loads(args))
    except:
        pass
    return jsonify({'args': ''})


@app.route('/getjobname/<task_id>')
def get_job_name(task_id):
    api_root = 'http://localhost:5555/api/task/info'
    task_api = '{}/{}'.format(api_root, task_id)
    resp = requests.get(task_api)
    try:
        reply = resp.json()
        name = reply.get('name', '')
        return jsonify({"name": name})
    except:
        pass
    return jsonify({'name': ''})


@app.route('/download/<path:filename>', methods=['POST', 'GET'])
def download(filename):
    try:
        return send_file(safe_join(app.root_path, filename), filename)
    except Exception as e:
        return str(e)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory(os.path.join(os.getcwd(), 'dist', 'js'), path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory(os.path.join(os.getcwd(), 'dist', 'css'), path)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory(os.path.join(os.getcwd(), 'dist'), 'index.html')


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    webbrowser.open_new(url)
    app.run(host='0.0.0.0', debug=False)

