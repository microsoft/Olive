# How to Define `host` or `target` Systems
A system is a environment concept (OS, hardware spec, device platform, supported EP) that a Pass is run in or a Model is evaluated on.

There are three systems in Olive: **local system**, **Python environment system**, **Docker system**. Each system is categorized in one of two types of systems: **host** and **target**. A **host** is the environment where the Pass is run, and a **target** is the environment where the Model is evaluated. Most of time, the **host** and **target** are the same, but they can be different in some cases. For example, you can run a Pass on a local machine with a CPU and evaluate a Model on a remote machine with a GPU.

## Accelerator Configuration

For each **host** or **target**, it will represent list of accelerators that are supported by the system. Each accelerator is represented by a dictionary with the following attributes:

- `device`: The device type of the accelerator. It could be "cpu", "gpu", "npu", etc.
- `execution_providers`: The execution provider list that are supported by the accelerator. For example, `["CUDAExecutionProvider", "CPUExecutionProvider"]`.

The **host** only use the `device` attribute to run the passes. Instead, the **target** uses both `device` and `execution_providers` attributes to run passes or evaluate models.

## Local System

The local system represents the local machine where the Pass is run or the Model is evaluated and it only contains the accelerators attribute.

```json
{
    "systems": {
        "local_system" : {
            "type": "LocalSystem",
            "accelerators": [
                {
                    "device": "cpu",
                    "execution_providers": ["CPUExecutionProvider"]
                }
            ]
        }
    },
    "engine": {
        "target": "local_system"
    }
}
```

```{Note}
* The accelerators attribute for local system is optional. If not provided, Olive will get the available execution providers installed in the current local machine and infer its `device`.
* For each accelerator, either `device` or ``execution_providers` is optional but not both if the accelerators are specified. If `device` or `execution_providers` is not provided, Olive will infer the `device` or `execution_providers` if possible.

Most of time, the local system could be simplified as below:

    {
        "type": "LocalSystem"
    }

In this case, Olive will infer the `device` and `execution_providers` based on the local machine. Please note the `device` attribute is required for **host** since it will not be inferred for host system.
```

## Python Environment System

The python environment system represents the python virtual environment. The python environment system is configured with the following attributes:

- `accelerators`: The list of accelerators that are supported by the system.
- `python_environment_path`: The path to the python virtual environment, which is required for native python system.
- `environment_variables`: The environment variables that are required to run the python environment system. This is optional.
- `prepend_to_path`: The path that will be prepended to the PATH environment variable. This is optional.

Here are the examples of configuring the general Python Environment System.

```json
{
    "systems"  : {
        "python_system" : {
            "type": "PythonEnvironment",
            "python_environment_path": "/home/user/.virtualenvs/myenv/bin",
            "accelerators": [
                {
                    "device": "cpu",
                    "execution_providers": [
                        "CPUExecutionProvider",
                        "OpenVINOExecutionProvider"
                    ]
                }
            ]
        }
    },
    "engine": {
        "target": "python_system"
    }
}
```

```{Note}
- The python environment must have `olive-ai` installed.
- The accelerators for python system is optional. If not provided, Olive will get the available execution providers installed in current python virtual environment and infer its `device`.
- For each accelerator, either `device` or `execution_providers` is optional but not both if the accelerators are specified. If `device` or `execution_providers` is not provided, Olive will infer the `device` or `execution_providers` if possible.
```

## Docker System

The Docker system refers to the container environment where the Olive workflow is executed. It can only be set to `host`, indicating that the workflow runs inside a Docker container. Once the workflow completes, the output model folder will be mounted back to the same relative path specified in your configuration file. For example, if `output_dir` is set to `xx/yy/zz`, the output will be saved to `xx/yy/zz` on the host machine after the workflow finishes.

If a target system is specified, it can only be `Local` or `PythonEnvironment`.

The docker system is configured with the following attributes:

* `accelerators`: The list of accelerators that are supported by the system.
* `image_name`: The name (and optionally tag) of the Docker image to be built, e.g. `"my-image:latest"`. The default value is `"olive-docker:latest"`
* `build_context_path`: This directory should contain all files required for the build, including the Dockerfile.
* `dockerfile`: The relative path to the Dockerfile within the build context, e.g. `"Dockerfile"` or `"docker/Dockerfile.dev"`.
* `build_args`: A dictionary of build-time variables to pass to the Dockerfile. Keys are argument names and values are their corresponding values.
* `run_params`: A dictionary of parameters to be used when running the container. These correspond to keyword arguments accepted by `docker.containers.run()`.
* `work_dir`: The working directory where the workflow runs and files are mounted. The default value is `/olive-ws`.
* `clean_image`: Whether to remove the Docker image after the workflow finishes. The default value is `True`.


```{Note}
* The docker container must have `olive-ai` installed.
```

### Prerequisites


1. Docker Engine installed on the host machine.
2. docker extra dependencies installed. `pip install olive-ai[docker]` or `pip install docker`


### Native Docker System

```json
{
    "type": "Docker",
    "image_name": "olive",
    "build_context_path": "docker",
    "dockerfile": "Dockerfile"
    "accelerators": [
        {
            "device": "cpu",
            "execution_providers": ["CPUExecutionProvider"]
        }
    ]
}
```
