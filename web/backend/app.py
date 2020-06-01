# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from flask import Flask, jsonify, request, send_file, url_for
from flask_cors import CORS
import uuid, sys, json
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../utils'))
import onnxpipeline
from convert_test_data import convert_data_to_pb
from werkzeug.utils import secure_filename
import netron
import posixpath
import tarfile
import app_config
from shutil import copyfile, rmtree
from datetime import datetime
from celery import Celery
import requests, json

# app_configuration
DEBUG = True

# instantiate the app
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


def create_input_dir():
    """
    Create a folder with unique timestamp to hold user input files such as model file 
    and test data files.
    """
    cur_ts = get_timestamp()
    # Create a folder to hold input files
    file_input_dir = os.path.join(app.root_path, app_config.FILE_INPUTS_DIR, cur_ts)
    if not os.path.exists(file_input_dir):
        os.makedirs(file_input_dir)
    return file_input_dir


def create_temp_json(path, metadata):
    """
    Store metadata from proxy request to JSON. 
    Args: 
        path: The folder to store JSON.
        metadata: Metadata from proxy request.
    Returns:
        temp_json: Path to the created .json file.
    """
    temp_json = os.path.join(path, 'temp.json')
    if os.path.exists(temp_json):
        os.remove(temp_json)
    # Get input parameters from request and save them to json file
    metadata.save(temp_json)
    return temp_json


def store_file_from_request(request, file_key, save_to_dir):
    """
    Retrieve and store file, if any, from request to internal folder.
    Args:
        request: Proxy request
        file_key: The key to retrieve file from request
        save_to_dir: Folder to save file.
    Returns:
        Saved file path. Empty if no file found. 
    """
    if file_key in request.files:
        if os.path.exists(save_to_dir):
            rmtree(save_to_dir)
        os.mkdir(save_to_dir)
        temp_file = request.files[file_key]
        saved_file_path = os.path.join(save_to_dir, temp_file.filename)
        request.files[file_key].save(saved_file_path)
        return saved_file_path
    return ''


def store_files_from_request(request, file_key, save_to_dir, isTestData=False):
    """
    Retrieve and store files, if any, from request to internal folder.
    Args:
        request: Proxy request
        file_key: The key to retrieve file from request
        save_to_dir: Folder to save files.
    Returns:
        Saved file path. Empty if no file found. 
    """
    if file_key in request.files:
        if os.path.exists(save_to_dir):
            rmtree(save_to_dir)
        os.mkdir(save_to_dir)
        file_list = request.files.getlist(file_key)
        # Upload test data
        for td in file_list:
            saved_file_path = os.path.join(save_to_dir, td.filename)
            td.save(saved_file_path)
            if isTestData and saved_file_path.split('.')[1] != 'pb':
                convert_data_to_pb(saved_file_path, saved_file_path)
                # remove the pickle file
                os.remove(saved_file_path)


def get_convert_json(request, test_data_root_path):
    """
    Generate and return a JSON file specifying job specs for onnx-convert job run. 
    Args: 
        request: proxy request from frontend.
        test_data_root_path: folder to store test data internally.
    """
    file_input_dir = create_input_dir()
    temp_json = create_temp_json(file_input_dir, request.files['metadata'])
    with open(temp_json, 'r') as f:
        json_data = json.load(f)

    # Save files passed from request
    model_name = store_file_from_request(request, 'file', file_input_dir)
    json_data['model'] = model_name

    # Store test data files if any
    test_data_dir = os.path.join(test_data_root_path, app_config.TEST_DATA_DIR)
    store_files_from_request(request, 'test_data[]', test_data_dir)

    # Store tensorflow savedmodel files if any
    variables_dir = os.path.join(file_input_dir, 'variables')
    store_files_from_request(request, 'savedModel[]', variables_dir)
    # If using tensorflow saved model, change model path to the directory containing the model files.
    if os.path.exists(variables_dir):
        json_data['model'] = os.path.dirname(os.path.abspath(json_data['model']))

    # remove empty input folder
    if len(os.listdir(file_input_dir)) == 0:
        os.rmdir(file_input_dir)

    with open(temp_json, 'w') as f:
        json.dump(json_data, f)

    return model_name, temp_json[len(app.root_path) + 1:], json_data


def get_perf_json(request):
    """
    Generate and return a JSON file specifying job specs for perf-tuning job run. 
    Args: 
        request: proxy request from frontend.
    Returns:
        model_name: stored model directory
        temp_json: path storing JSON as input for perf-tuning job.
        json_data: JSON object storing the input arguments from frontend user
    """
    file_input_dir = create_input_dir()

    temp_json = create_temp_json(file_input_dir, request.files['metadata'])
    with open(temp_json, 'r') as f:
        json_data = json.load(f)

    if request.form.get('prev_model_path'):
        # Get input files directory from the mounted path if the inputs are already present from previous calls
        local_mounted_path = get_local_mounted_path(request.form.get('prev_model_path'))
        file_input_dir = os.path.join(app.root_path, local_mounted_path)

    # Save files passed from request
    model_name = store_file_from_request(request, 'file', file_input_dir)
    json_data['model'] = model_name

    # Store test data files if any
    test_data_dir = os.path.join(file_input_dir, app_config.TEST_DATA_DIR)
    store_files_from_request(request, 'test_data[]', test_data_dir)

    with open(temp_json, 'w') as f:
        json.dump(json_data, f)

    return model_name, temp_json[len(app.root_path) + 1:], json_data


def get_timestamp():
    """
    Generate a unique timestamp.
    Returns:
        Unique timestamp as a string.
    """
    ts = int(datetime.now().timestamp())
    return str(ts)


def get_time_from_ts(time):
    """
    Get a readable time from timestamp.
    Args:
        time: Timestamp
    Returns:
        Date and time string.
    """
    dt = datetime.fromtimestamp(time)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f %Z")


# Get the local path from a mounted path. e.g. mnt/model/path -> path
def get_local_mounted_path(path):
    """
    Retrieve local relative path from a docker mount path
    Args:
        path: Mounted path for docker.
    Returns:
        Local path for the mounted path relative to the project root path. 
    """
    if len(path) < len(app_config.MOUNT_PATH):
        return path
    return os.path.dirname(path[len(app_config.MOUNT_PATH) + 2:])


def garbage_collect(dir, max=20):
    """
    Garbage collect any old files cached for job runs. Current logic is to keep 
    a maximum number of run information in the given directory. Remove the oldest 
    if exceeds maximum.
    Args:
        dir: Directory to remove old files
        max: Maximum number of folders allowed in the directory. Default: 20.
    """
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
    """
    Clean up files in specific directory.
    Args:
        root_path: The project absolute root path.
    """
    garbage_collect(os.path.join(root_path, app_config.FILE_INPUTS_DIR))
    garbage_collect(os.path.join(root_path, app_config.CONVERT_RES_DIR))
    garbage_collect(os.path.join(root_path, app_config.PERF_RES_DIR))
    garbage_collect(os.path.join(root_path, app_config.DOWNLOAD_DIR))


@app.route('/visualize', methods=['POST'])
def visualize():
    """
    Start netron with model from request.
    Returns:
        JSON response indicting success.
    """
    if request.method == 'POST':
        response_object = {'status': 'success'}
        temp_model = request.files['file']
        model_name = temp_model.filename

        request.files['file'].save(model_name)

        netron.start(model_name, browse=False, host='0.0.0.0')

    return jsonify(response_object)


@celery.task(bind=True)
def convert(self, model_name, temp_json, cur_ts, root_path, input_params):
    """
    Define a celery job which runs OLive pipeline onnx-converter.
    Args:
        model_name: Model path to run onnx-convert.
        temp_json: Input specs for onnx-convert job.
        cur_ts: Unique timestamp for the job.
        root_path: Project app absolute root path.
        input_params: JSON object storing the input arguments from frontend user
    Return:
        JSON response with run status and results.
    """
    response_object = {'status': 'success'}
    # Initiate pipeline object with targeted directory
    pipeline = onnxpipeline.Pipeline(convert_directory=os.path.join(app_config.CONVERT_RES_DIR, cur_ts))
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
    input_root = os.path.join(root_path, target_dir)
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

    clean(root_path)

    return response_object


@app.route('/convert', methods=['POST'])
def send_convert_job():
    """
    Send onnx-convert job to celery task queue.
    Returns: 
        JSON object with job status.
    """
    cur_ts = get_timestamp()
    convert_directory = os.path.join(app.root_path, app_config.CONVERT_RES_DIR, cur_ts)

    # may not work in Unix OS cuz the permission
    if os.path.exists(convert_directory):
        rmtree(convert_directory)
    os.makedirs(convert_directory)

    model_name, temp_json, input_params = get_convert_json(request, convert_directory)
    job_name = "convert." + request.form.get('job_name')
    job = convert.apply_async(args=[model_name, temp_json, cur_ts, app.root_path, input_params], shadow=job_name)
    return jsonify({'Location': url_for('convert_status', task_id=job.id), 'job_id': job.id}), 202


@app.route('/convertstatus/<task_id>')
def convert_status(task_id):
    """
    Get onnx-converter task status from celery.
    Returns:
        JSON object with task status. If "SUCCESS", return the task result. 
        If "ERROR", return error message.
    """
    task = convert.AsyncResult(task_id)
    if task.state == 'PENDING' or task.state == 'STARTED':
        # job has not started yet
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'output_json': task.info.get('output_json', ''),
            'input_path': task.info.get('input_path', ''),
            'model_path': task.info.get('model_path', ''),
            'converted_model': task.info.get('converted_model', ''),
            'status': task.info.get('status', '')
        }
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/perf_tuning', methods=['POST'])
def send_perf_job():
    """
    Send perf-tuning job to celery task queue.
    Returns: 
        JSON object with job status.
    """
    _, temp_json, input_params = get_perf_json(request)
    job_name = 'perf_tuning.' + request.form.get('job_name')
    job = perf_tuning.apply_async(args=[temp_json, input_params], shadow=job_name)
    return jsonify({'Location': url_for('perf_status', task_id=job.id), 'job_id': job.id}), 202


@celery.task(bind=True)
def perf_tuning(self, temp_json, input_params):
    """
    Define a celery job which runs OLive pipeline perf-tuning.
    Args:
        temp_json: Input specs for perf-tuning job.
        input_params: JSON object storing the input arguments from frontend user.
    Return:
        JSON response with run status and results.
    """
    response_object = {'status': 'success'}

    pipeline = onnxpipeline.Pipeline()

    # create result dir
    result_dir = os.path.join(app_config.PERF_RES_DIR, get_timestamp())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result = pipeline.perf_tuning(input_json=temp_json, result=result_dir)
    try:
        r = pipeline.get_result(result)
        response_object['result'] = json.dumps(r.latency)
        response_object['profiling'] = r.profiling_ops
    except RuntimeError:
        pass

    response_object['logs'] = pipeline.output
    # clean up result dir
    # if os.path.exists(result_dir):
    #    rmtree(result_dir)
    clean(app.root_path)
    return response_object


@app.route('/perfstatus/<task_id>')
def perf_status(task_id):
    """
    Get perf-tuning task status from celery.
    Returns:
        JSON object with task status. If "SUCCESS", return the task result. 
        If "ERROR", return error message.
    """
    task = perf_tuning.AsyncResult(task_id)
    if task.state == 'PENDING' or task.state == 'STARTED':
        # job did not start yet
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'result': task.info.get('result', ''),
            'profiling': task.info.get('profiling', ''),
            'status': task.info.get('status', ''),
            'logs': task.info.get('logs', ''),
        }
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/gettasks')
def get_tasks():
    """
    Get all queued tasks from Celery. 
    Returns:
        All tasks and their information in a JSON object. 
        {
            job_id: {
                "name":
                "state":
                "type":
                "received":
                "started":
                ...
            }
            ...
        }
    """
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
    """
    Get task user input argments given a task id.
    Args:
        task_id: Unique task id generated by celery to retrieve jobs.
    Returns:
        JSON object with user arguments.
    """
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
    """
    Get task job name as defined by user in frontend given a task id.
    Args:
        task_id: Unique task id generated by celery to retrieve jobs.
    Returns:
        JSON object {name: }.
    """
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
    """
    Send a downloadable file to frontend.
    Args:
        filename: The file to download.
    Returns:
        Downloadable file. 
    """
    try:
        path = os.path.join(app.root_path, app_config.DOWNLOAD_DIR)
        return send_file(os.path.join(path, filename), filename)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
