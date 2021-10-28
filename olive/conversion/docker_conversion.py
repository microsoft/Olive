import json
import os
import posixpath
from shutil import copy
import subprocess
import time

import docker

import olive
from ..constants import PYTORCH_VERSION, TENSORFLOW_VERSION, MODEL_MOUNT_DIR, CVT_IMG_NAME, ONNX_MODEL_PATH, OLIVE_MOUNT_DIR, MCR_PREFIX


def convert_in_docker(args):
    ts = str(time.time()).split(".")[0]
    image_name = pull_conversion_image(args)

    model_path, sample_input_data_path, onnx_model_path = get_local_path(args)
    local_model_dir, local_onnx_model_dir = get_local_dir(model_path, onnx_model_path)

    duplicated_sample_input_data_path = os.path.join(local_model_dir, "sample_data_{}.npz".format(ts)) if sample_input_data_path else None
    duplicated_onnx_model_path = os.path.join(local_model_dir, "onnx_model_{}.onnx".format(ts))
    duplicated_config_path = os.path.join(local_model_dir, "cvt_config_{}.json".format(ts)) if args.conversion_config else None
    if sample_input_data_path:
        copy(sample_input_data_path, duplicated_sample_input_data_path)

    update_arguments(args, model_path, duplicated_sample_input_data_path, duplicated_onnx_model_path, duplicated_config_path)
    cvt_args_str = generate_docker_arguments(args)
    olive_package_dir = olive.__path__[0]
    volumes_bind_dict = {
        local_model_dir: {"bind": MODEL_MOUNT_DIR, "mode": "rw"},
        olive_package_dir: {"bind": OLIVE_MOUNT_DIR, "mode": "rw"}
    }
    run_conversion_container(image_name, cvt_args_str, volumes_bind_dict)

    copy(duplicated_onnx_model_path, os.path.join(local_onnx_model_dir, onnx_model_path))
    try:
        remove_duplicated_files(duplicated_onnx_model_path, duplicated_sample_input_data_path, duplicated_config_path)
    except PermissionError:
        pass


def remove_duplicated_files(duplicated_onnx_model_path, duplicated_sample_input_data_path, duplicated_config_path):
    os.remove(duplicated_onnx_model_path)
    if duplicated_sample_input_data_path:
        os.remove(duplicated_sample_input_data_path)
    if duplicated_config_path:
        os.remove(duplicated_config_path)


def run_conversion_container(image_name, cvt_args_str, volumes_bind_dict):
    client = docker.from_env()

    stream = client.containers.run(image=image_name,
                                   command=cvt_args_str,
                                   volumes=volumes_bind_dict,
                                   detach=True)
    logs = stream.logs(stream=True)
    for line in logs:
        if type(line) is not str:
            line = line.decode(encoding='UTF-8')
        line = line.replace('\n', '\r\n')
        print(line)
    
    stream.remove(force=True)


def generate_docker_arguments(args):
    cvt_args_str = "python -m olive convert "
    for key in args.__dict__.keys():
        if key not in ["use_conda", "use_docker", "func"]:
            if args.__dict__[key]:
                cvt_args_str = cvt_args_str + "--{} {} ".format(key, args.__dict__[key])
    return cvt_args_str


def update_arguments(args, model_path, duplicated_sample_input_data_path, duplicated_onnx_model_path, duplicated_config_path):
    mount_model_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(model_path))
    mount_sample_input_data_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(duplicated_sample_input_data_path)) if duplicated_sample_input_data_path else None
    mount_onnx_model_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(duplicated_onnx_model_path))

    if args.conversion_config:
        with open(args.conversion_config, 'r') as f:
            config_dict = json.load(f)
        config_dict["model_path"] = mount_model_path
        config_dict["onnx_model_path"] = mount_onnx_model_path
        if mount_sample_input_data_path:
            config_dict["sample_input_data_path"] = mount_sample_input_data_path
        with open(duplicated_config_path, 'w') as f:
            json.dump(config_dict, f)
        mount_config_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(duplicated_config_path))
        args.conversion_config = mount_config_path

    else:
        args.model_path = mount_model_path
        args.onnx_model_path = mount_onnx_model_path
        if mount_sample_input_data_path:
            args.sample_input_data_path = mount_sample_input_data_path


def get_local_dir(model_path, onnx_model_path):
    local_model_dir = os.path.dirname(os.path.abspath(model_path))
    local_onnx_model_dir = os.path.dirname(os.path.abspath(onnx_model_path))
    return local_model_dir, local_onnx_model_dir


def pull_conversion_image(args):
    if args.conversion_config:
        with open(args.conversion_config, 'r') as f:
            config_dict = json.load(f)
        model_framework = config_dict.get("model_framework")
        framework_version = config_dict.get("framework_version")
    else:
        model_framework = args.model_framework.lower()
        framework_version = args.framework_version

    if not framework_version:
        if model_framework == "pytorch":
            framework_version = PYTORCH_VERSION
        else:
            framework_version = TENSORFLOW_VERSION

    image_name = "{}/{}:{}_{}".format(MCR_PREFIX, CVT_IMG_NAME, model_framework, framework_version)
    subprocess.run("docker pull {}".format(image_name), shell=True)
    return image_name


def get_local_path(args):
    if args.conversion_config:
        with open(args.conversion_config, 'r') as f:
            config_dict = json.load(f)
        model_path = config_dict.get("model_path")
        sample_input_data_path = config_dict.get("sample_input_data_path")
        onnx_model_path = config_dict.get("onnx_model_path", ONNX_MODEL_PATH)
    else:
        model_path = args.model_path
        sample_input_data_path = args.sample_input_data_path
        onnx_model_path = args.onnx_model_path
    return model_path, sample_input_data_path, onnx_model_path