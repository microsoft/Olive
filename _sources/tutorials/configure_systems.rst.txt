.. _how_to_configure_system:

How To Configure System
=========================

A system is the environment (OS, hardware spec, device platform, supported EP) that a Pass is run in or a Model
is evaluated on. It can thus be the **host** of a Pass or the **target** of an evaluation. This document describes
how to configure the different types of Systems.

Accelerator Configuration
-------------------------
For each **host** or **target**, it could specify multiple accelerators associated with it. Each accelerator could have the following attributes.

* :code:`device`: The device type of the accelerator. It could be "cpu", "gpu", "npu", etc. Please refer to the API documentation for the full list of supported devices.
* :code:`execution_providers`: The execution provider list that are supported by the accelerator. For e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"].

Note:

- The accelerators for local system or python system is optional. If not provided, Olive will get the available execution providers installed in current local machine and infer its device.
- For local system or python system, either device or execution_providers is optional but not both if the accelerators are specified. If device or execution_providers is not provided, Olive will infer the device or execution_providers if possible.
- For docker system and AzureML system, both device and execution_providers are mandatory. Otherwise, Olive will raise an error.

Local System
-------------

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "LocalSystem",
                "config": {
                    "accelerators": [{"device": "cpu"}]
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.local import LocalSystem
            from olive.system.common import Device

            local_system = LocalSystem(
                accelerators=[{"device": Device.CPU}]
            )

Please refer to :ref:`local_system_config` for more details on the config options.

AzureML System
---------------

Prerequisites
^^^^^^^^^^^^^

1. azureml extra dependencies installed.

    .. code-block:: bash

        pip install olive-ai[azureml]

    or

    .. code-block:: bash

        pip install azure-ai-ml azure-identity

2. AzureML Workspace with necessary compute created. Refer to
`this <https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace>`_ for more details. Download
the workspace config json.

System Configuration
^^^^^^^^^^^^^^^^^^^^^

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "AzureML",
                "config": {
                    "aml_compute": "cpu-cluster",
                    "aml_docker_config": {
                        "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                        "conda_file_path": "conda.yaml"
                    }
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.azureml import AzureMLDockerConfig, AzureMLSystem

            docker_config = AzureMLDockerConfig(
                base_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                conda_file_path="conda.yaml",
            )
            aml_system = AzureMLSystem(
                aml_compute="cpu-cluster",
                aml_docker_config=docker_config
            )

If you provide a :code:`aml_docker_config`, Olive will create a new Azure ML Environment using the :code:`aml_docker_config` configuration.
Alternatively, you can provide an existing Azure ML Environment using :code:`aml_environment_config`:

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "AzureML",
                "config": {
                    "aml_compute": "cpu-cluster",
                    "aml_environment_config": {
                        "name": "myenv",
                        "version": "1"
                    }
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.azureml import AzureMLDockerConfig, AzureMLSystem

            aml_environment_config = AzureMLEnvironmentConfig(
                name="myenv",
                version="1",
            )
            aml_system = AzureMLSystem(
                aml_compute="cpu-cluster",
                aml_environment_config=aml_environment_config
            )


Olive can also manage the environment by setting :code:`olive_managed_env = True`

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "AzureML",
                "config": {
                    "aml_compute": "cpu-cluster",
                    "accelerators": [
                        {
                            "device": "cpu",
                            "execution_providers": [
                                "CPUExecutionProvider",
                                "OpenVINOExecutionProvider"
                            ]
                        }
                    ],
                    "olive_managed_env": true,
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.azureml import AzureMLSystem

            aml_system = AzureMLSystem(
                aml_compute="cpu-cluster",
                accelerators=[{"device": "cpu", "execution_providers": ["CPUExecutionProvider", "OpenVINOExecutionProvider"]}],
                olive_managed_env=True,
            )


Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/bert/conda.yaml>`__
for :code:`"conda.yaml"`.

.. important::

    The AzureML environment must have :code:`olive-ai` installed if :code:`olive_managed_env = False`!

Please refer to :ref:`azureml_system_config` for more details on the config options.

AzureML Readymade Systems
^^^^^^^^^^^^^^^^^^^^^^^^^

There are some readymade systems available for AzureML. These systems are pre-configured with the necessary.
    .. code-block:: json

            {
                "type": "AzureNDV2System",
                "config": {
                    "aml_compute": "gpu-cluster",
                    "aml_docker_config": {
                        "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                        "conda_file_path": "conda.yaml"
                    }
                }
            }

Please refer to :ref:`olive_system_alias` for the list of supported AzureML readymade systems.


Docker System
--------------

Prerequisites
^^^^^^^^^^^^^

1. Docker Engine installed on the host machine.

2. docker extra dependencies installed.

    .. code-block:: bash

        pip install olive-ai[docker]

    or

    .. code-block:: bash

        pip install docker

System Configuration
^^^^^^^^^^^^^^^^^^^^^

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "Docker",
                "config": {
                    "local_docker_config": {
                        "image_name": "olive",
                        "build_context_path": "docker",
                        "dockerfile": "Dockerfile"
                    }
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.docker import DockerSystem, LocalDockerConfig

            local_docker_config = LocalDockerConfig(
                image_name="olive",
                build_context_path="docker",
                dockerfile="Dockerfile",
            )
            docker_system = DockerSystem(local_docker_config=local_docker_config)

Olive can manage the environment by setting :code:`olive_managed_env = True`

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "Docker",
                "config": {
                    "accelerators": [
                        {
                            "device": "cpu",
                            "execution_providers": [
                                "CPUExecutionProvider",
                                "OpenVINOExecutionProvider"
                            ]
                        }
                    ],
                    "olive_managed_env": true,
                    "requirements_file": "mnist_requirements.txt"
                    }
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.docker import DockerSystem

            docker_system = DockerSystem(
                accelerators=["cpu"],
                olive_managed_env=True,
                requirements_file="mnist_requirements.txt",
            )

Please refer to this `example <https://github.com/microsoft/Olive/tree/main/examples/bert/docker>`__
for :code:`"docker"` and :code:`"Dockerfile"`.

.. important::

    The docker container must have :code:`olive-ai` installed!

Please refer to :ref:`docker_system_config` for more details on the config options.

Python Environment System
--------------------------

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "PythonEnvironment",
                "config": {
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
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.python_environment import PythonEnvironmentSystem
            from olive.system.common import Device

            python_environment_system = PythonEnvironmentSystem(
                python_environment_path = "/home/user/.virtualenvs/myenv/bin",
                accelerators = [{"device": Device.CPU}]
            )

Olive can also manage the environment by setting :code:`olive_managed_env = True`. This feature works best when used from Conda.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "PythonEnvironment",
                "config": {
                    "accelerators": [{"device": "cpu"}]
                    "olive_managed_env": true,
                }
            }
    .. tab:: Python Class
        .. code-block:: python

            from olive.systems.python_environment import PythonEnvironmentSystem
            from olive.system.common import Device

            python_environment_system = PythonEnvironmentSystem(
                olive_managed_env = True,
                accelerators = [{"device": Device.CPU}]
            )

.. important::

    The python environment must have :code:`olive-ai` installed if :code:`olive_managed_env = False`!

Please refer to :ref:`python_environment_system_config` for more details on the config options.


Isolated ORT System
-------------------
.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "IsolatedORT",
                "config": {
                    "python_environment_path": "/home/user/.virtualenvs/myenv/bin",
                    "accelerators": [{"device": "cpu"}]
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.ort_evironment import IsolatedORTSystem
            from olive.system.common import Device

            python_environment_system = IsolatedORTSystem(
                python_environment_path = "/home/user/.virtualenvs/myenv/bin",
                accelerators = [{"device": Device.CPU}]
            )

IsolatedORTSystem does not support olive_managed_env and can only be used to evaluate ONNX models.

.. important::

    The python environment must have the relevant ONNX runtime package installed!

Please refer to :ref:`isolated_ort_system_config` for more details on the config options.
