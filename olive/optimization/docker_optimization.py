import json
import os
import posixpath
from shutil import copy, rmtree
import subprocess
import time
import sys

import docker

import olive
from ..constants import ONNXRUNTIME_VERSION, MODEL_MOUNT_DIR, OPT_IMG_NAME, OLIVE_RESULT_PATH, OLIVE_MOUNT_DIR, MCR_PREFIX


def optimize_in_docker(args):
    ts = str(time.time()).split(".")[0]
    image_name = pull_optimization_image(args)

    model_path, sample_input_data_path, result_path = get_local_path(args)
    local_model_dir, local_result_dir = get_local_dir(model_path, result_path)
    duplicated_sample_input_data_path = os.path.join(local_model_dir, "sample_data_{}.npz".format(ts)) if sample_input_data_path else None
    duplicated_result_folder = os.path.join(local_model_dir, "opt_result_{}".format(ts))
    duplicated_config_path = os.path.join(local_model_dir, "opt_config_{}.json".format(ts)) if args.optimization_config else None

    if sample_input_data_path:
        copy(sample_input_data_path, duplicated_sample_input_data_path)
    update_arguments(args, model_path, duplicated_sample_input_data_path, duplicated_result_folder, duplicated_config_path)

    opt_args_str = generate_docker_arguments(args)

    olive_package_dir = olive.__path__[0]

    volumes_bind_dict = {
        local_model_dir: {"bind": MODEL_MOUNT_DIR, "mode": "rw"},
        olive_package_dir: {"bind": OLIVE_MOUNT_DIR, "mode": "rw"}
    }

    run_optimization_container(args, image_name, opt_args_str, volumes_bind_dict)

    for f in os.listdir(duplicated_result_folder):
        copy(os.path.join(duplicated_result_folder, f), local_result_dir)

    try:
        remove_duplicated_files(duplicated_result_folder, duplicated_sample_input_data_path, duplicated_config_path)
    except PermissionError:
        pass


def remove_duplicated_files(duplicated_result_folder, duplicated_sample_input_data_path, duplicated_config_path):
    rmtree(duplicated_result_folder)
    if duplicated_sample_input_data_path:
        os.remove(duplicated_sample_input_data_path)
    if duplicated_config_path:
        os.remove(duplicated_config_path)


def run_optimization_container(args, image_name, opt_args_str, volumes_bind_dict):
    client = docker.from_env()
    if args.use_gpu:
        if sys.platform.startswith('win'):
            raise Exception("OLive does't support GPU optimization with Docker on Windows machine")

        stream = client.containers.run(image=image_name,
                                       command=opt_args_str,
                                       volumes=volumes_bind_dict,
                                       detach=True,
                                       device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])])
    else:
        stream = client.containers.run(image=image_name,
                                       command=opt_args_str,
                                       volumes=volumes_bind_dict,
                                       detach=True)
    logs = stream.logs(stream=True)
    for line in logs:
        if type(line) is not str:
            line = line.decode(encoding='UTF-8')
        line = line.replace('\n', '\r\n')
        print(line)

    stream.remove(force=True)


def pull_optimization_image(args):
    if args.use_gpu:
        image_name = "{}/{}:{}_{}".format(MCR_PREFIX, OPT_IMG_NAME, ONNXRUNTIME_VERSION, "gpu")
    else:
        image_name = "{}/{}:{}_{}".format(MCR_PREFIX, OPT_IMG_NAME, ONNXRUNTIME_VERSION, "cpu")
    subprocess.run("docker pull {}".format(image_name), shell=True)
    return image_name


def get_local_path(args):
    if args.optimization_config:
        with open(args.optimization_config, 'r') as f:
            config_dict = json.load(f)
        model_path = config_dict.get("model_path")
        sample_input_data_path = config_dict.get("sample_input_data_path")
        result_path = config_dict.get("result_path", OLIVE_RESULT_PATH)
    else:
        model_path = args.model_path
        sample_input_data_path = args.sample_input_data_path
        result_path = args.result_path

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    return model_path, sample_input_data_path, result_path


def get_local_dir(model_path, result_path):
    local_model_dir = os.path.dirname(os.path.abspath(model_path))
    local_result_dir = os.path.abspath(result_path)
    return local_model_dir, local_result_dir


def update_arguments(args, model_path, duplicated_sample_input_data_path, duplicated_result_folder, duplicated_config_path):
    mount_model_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(model_path))
    mount_sample_input_data_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(duplicated_sample_input_data_path)) if duplicated_sample_input_data_path else None
    mount_result_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(duplicated_result_folder))

    if args.optimization_config:
        with open(args.optimization_config, 'r') as f:
            config_dict = json.load(f)
        config_dict["model_path"] = mount_model_path
        config_dict["result_path"] = mount_result_path
        if mount_sample_input_data_path:
            config_dict["sample_input_data_path"] = mount_sample_input_data_path

        with open(duplicated_config_path, 'w') as f:
            json.dump(config_dict, f)

        mount_config_path = posixpath.join(MODEL_MOUNT_DIR, os.path.basename(duplicated_config_path))
        args.optimization_config = mount_config_path
    else:
        args.model_path = mount_model_path
        args.result_path = mount_result_path
        if mount_sample_input_data_path:
            args.sample_input_data_path = mount_sample_input_data_path


def generate_docker_arguments(args):
    opt_args_str = "python -m olive optimize ".format(OLIVE_MOUNT_DIR)
    if args.__dict__["optimization_config"]:
        opt_args_str = opt_args_str + "--optimization_config {} ".format(args.__dict__["optimization_config"])
    else:
        for key in args.__dict__.keys():
            if key not in ["use_conda", "use_docker", "use_gpu", "onnxruntime_version", "func"]:
                if args.__dict__[key]:
                    if key in ["quantization_enabled", "transformer_enabled", "trt_fp16_enabled", "throughput_tuning_enabled"]:
                        opt_args_str = opt_args_str + "--{} ".format(key)
                    else:
                        opt_args_str = opt_args_str + "--{} {} ".format(key, args.__dict__[key])
    return opt_args_str
