.. _how_to_configure_system:

How To Configure System
=========================

A system is a environment concept (OS, hardware spec, device platform, supported EP) that a Pass is run in or a Model is evaluated on.

There are two types of systems: **host** and **target**. A **host** is the environment where the Pass is run, and a **target** is the environment where the Model is evaluated.

There are five systems: **local system**, **Python environment system**, **Docker system**, **AzureML system**, **Isolated ORT system**.

In Olive, the system could be categorized into two kinds of system based on the :code:`olive_managed_env = True`.

* Native System: This is the normal system. All kinds of systems can be configured as native system. The **Isolated ORT system** can only be used as **target** system for model evaluation.
* Managed System: Olive will manage the environment by installing the required packages from the :code:`requirements_file` or Dockerfile in the environment. Only **Python environment system**, **Docker system**, **AzureML system** support managed system.
  The ``device`` and ``execution_providers`` for managed system is mandatory. Otherwise, Olive will raise an error.

Most of time, the **host** and **target** are the same, but they can be different in some cases. For example, you can run a Pass on a local machine with a CPU and evaluate a Model on a remote machine with a GPU.

Accelerator Configuration
-------------------------
For each **host** or **target**, it will represent list of accelerators that are supported by the system. Each accelerator is represented by a dictionary with the following attributes:

.. admonition:: AcceleratorConfig

    * :code:`device`: The device type of the accelerator. It could be "cpu", "gpu", "npu", etc. Please refer to :ref:`device` for the full list of supported devices.
    * :code:`execution_providers`: The execution provider list that are supported by the accelerator. For e.g. ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.

The **host** only use the ``device`` attribute to run the passes. Instead, the **target** uses both ``device`` and ``execution_providers`` attributes to run passes or evaluate models.

Local System
-------------
The local system represents the local machine where the Pass is run or the Model is evaluated and it only contains the accelerators attribute.

.. admonition:: Configuration

    * :code:`accelerators`: The list of accelerators that are supported by the system.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

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

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.local import LocalSystem
            from olive.system.common import Device

            local_system = LocalSystem(
                accelerators=[{"device": Device.CPU}]
            )

.. note::

    * Local system doesn't support :code:`olive_managed_env`.
    * The accelerators attribute for local system is optional. If not provided, Olive will get the available execution providers installed in the current local machine and infer its ``device``.
    * For each accelerator, either ``device`` or ``execution_providers`` is optional but not both if the accelerators are specified. If ``device`` or ``execution_providers`` is not provided, Olive will infer the ``device`` or ``execution_providers`` if possible.

    Most of time, the local system could be simplified as below:

    .. code-block:: json

        {
            "type": "LocalSystem"
        }

    In this case, Olive will infer the ``device`` and ``execution_providers`` based on the local machine. Please note the ``device`` attribute is required for **host** since it will not be inferred for host system.

Please refer to :ref:`local_system_config` for more details on the config options.

Python Environment System
--------------------------

The python environment system represents the python virtual environment. It can be configured as a native system or a managed system. The python environment system is configured with the following attributes:

.. admonition:: Configuration

    * :code:`accelerators`: The list of accelerators that are supported by the system.
    * :code:`python_environment_path`: The path to the python virtual environment, which is required for native python system.
    * :code:`environment_variables`: The environment variables that are required to run the python environment system. This is optional.
    * :code:`prepend_to_path`: The path that will be prepended to the PATH environment variable. This is optional.
    * :code:`olive_managed_env`: A boolean flag to indicate if the environment is managed by Olive. This is optional and defaults to False.
    * :code:`requirements_file`: The path to the requirements file, which is only required and used when :code:`olive_managed_env = True`.

Native Python Environment System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are the examples of configuring the general Python Environment System.

.. tabs::

    .. tab:: Config JSON

        .. code-block:: json

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

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.python_environment import PythonEnvironmentSystem
            from olive.system.common import Device

            python_environment_system = PythonEnvironmentSystem(
                python_environment_path = "/home/user/.virtualenvs/myenv/bin",
                accelerators = [{"device": Device.CPU}]
            )

.. note::

    * The python environment must have :code:`olive-ai` installed if :code:`olive_managed_env = False`!
    * The accelerators for python system is optional. If not provided, Olive will get the available execution providers installed in current python virtual environment and infer its ``device``.
    * For each accelerator, either ``device`` or ``execution_providers`` is optional but not both if the accelerators are specified. If ``device`` or ``execution_providers`` is not provided, Olive will infer the ``device`` or ``execution_providers`` if possible.


Managed Python Environment System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When :code:`olive_managed_env = True`, Olive will manage the python environment by installing the required packages from the :code:`requirements_file`. As the result, the :code:`requirements_file` is required and must be provided.

For managed python environment system, Olive only infer the onnxruntime from the following onnxruntime execution providers:

* CPUExecutionProvider: (*onnxruntime*)
* CUDAExecutionProvider: (*onnxruntime-gpu*)
* TensorrtExecutionProvider: (*onnxruntime-gpu*)
* OpenVINOExecutionProvider: (*onnxruntime-openvino*)
* DmlExecutionProvider: (*onnxruntime-directml*)

.. code-block:: json

    {
        "type": "PythonEnvironment",
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

Please refer to :ref:`python_environment_system_config` for more details on the config options.

.. caution::
    Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/resnet/resnet_multiple_ep.json>`__
    for how to use managed python environment system to optimize the model against different execution providers.

Docker System
--------------

The docker system represents the docker container where the Pass is run or the Model is evaluated. It can be configured as a native system or a managed system. The docker system is configured with the following attributes:

.. admonition:: Configuration

    * :code:`accelerators`: The list of accelerators that are supported by the system.
    * :code:`local_docker_config`: The configuration for the local docker system, which includes the following attributes:

        * :code:`image_name`: The name of the docker image.
        * :code:`build_context_path`: The path to the build context.
        * :code:`dockerfile`: The path to the Dockerfile.

    * :code:`requirements_file`: The path to the requirements file. If provided, Olive will install the required packages from the requirements file in the docker container.
    * :code:`olive_managed_env`: A boolean flag to indicate if the environment is managed by Olive. This is optional and defaults to False.

.. note::

    * the :code:`build_context_path`, :code:`dockerfile` and :code:`requirements_file` cannot be ``None`` at the same time.
    * The docker container must have :code:`olive-ai` installed.
    * The ``device`` and ``execution_providers`` for docker system is mandatory. Otherwise, Olive will raise an error.

Prerequisites
^^^^^^^^^^^^^

1. Docker Engine installed on the host machine.

2. docker extra dependencies installed.

    .. code-block:: bash

        pip install olive-ai[docker]

    or

    .. code-block:: bash

        pip install docker

Native Docker System
^^^^^^^^^^^^^^^^^^^^

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "Docker",
                "local_docker_config": {
                    "image_name": "olive",
                    "build_context_path": "docker",
                    "dockerfile": "Dockerfile"
                },
                "accelerators": [
                    {
                        "device": "cpu",
                        "execution_providers": ["CPUExecutionProvider"]
                    }
                ]
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

Managed Docker System
^^^^^^^^^^^^^^^^^^^^^

When :code:`olive_managed_env = True`, Olive will manage the docker environment by installing the required packages from the :code:`requirements_file` in the docker container if provided.
From the time being, Olive only supports the following base Dockerfiles based on input execution providers:

* CPUExecutionProvider: (*Dockerfile.cpu*)
* CUDAExecutionProvider: (*Dockerfile.gpu*)
* TensorrtExecutionProvider: (*Dockerfile.gpu*)
* OpenVINOExecutionProvider: (*Dockerfile.openvino*)

A typical managed Docker system can be configured by the following example:

.. code-block:: json

    {
        "type": "Docker",
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

AzureML System
---------------

The AzureML system represents the Azure Machine Learning workspace where the Pass is run or the Model is evaluated. It can be configured as a native system or a managed system. The AzureML system is configured with the following attributes:

.. admonition:: Configuration

    * :code:`accelerators`: The list of accelerators that are supported by the system, which is required.
    * :code:`aml_compute`: The name of the AzureML compute, which is required.
    * :code:`azureml_client_config`: The configuration for the AzureML client, which includes the following attributes:

        * :code:`subscription_id`: The subscription id of the AzureML workspace.
        * :code:`resource_group`: The resource group of the AzureML workspace.
        * :code:`workspace_name`: The name of the AzureML workspace.

    * :code:`aml_docker_config`: The configuration for the AzureML docker system, which includes the following attributes:

        * :code:`base_image`: The base image for the AzureML environment.
        * :code:`dockerfile`: The path to the Dockerfile of the AzureML environment.
        * :code:`build_context_path`: The path to the build context of the AzureML environment.
        * :code:`conda_file_path`: The path to the conda file.
        * :code:`name`: The name of the AzureML environment.
        * :code:`version`: The version of the AzureML environment.

    * :code:`aml_environment_config`: The configuration for the AzureML environment, which includes the following attributes:

        * :code:`name`: The name of the AzureML environment.
        * :code:`version`: The version of the AzureML environment.
        * :code:`label`: The label of the AzureML environment.

    * :code:`requirements_file`: The path to the requirements file. If provided, Olive will install the required packages from the requirements file in the AzureML environment.
    * :code:`tags`: The tags for the AzureML environment. This is optional.
    * :code:`resources`: The resources dictionary for the AzureML environment. This is optional.
    * :code:`instance_count`: The instance count for the AzureML environment. Default is 1.
    * :code:`datastore`: The datastore name where to export artifacts. Default is `workspaceblobstore`.
    * :code:`olive_managed_env`: A boolean flag to indicate if the environment is managed by Olive. This is optional and defaults to False.

.. note::

    * Both :code:`aml_docker_config` and :code:`aml_environment_config` cannot be ``None`` at the same time.
    * If :code:`aml_environment_config` is provided, Olive will use the existing AzureML environment with the specified name, version and label.
    * Otherwise, Olive will create a new AzureML environment using the :code:`aml_docker_config` configuration.
    * The :code:`azureml_client_config` will be propagated from engine :code:`azureml_client` if not provided.
    * The :code:`requirements_file` is only used when :code:`olive_managed_env = True` to install the required packages in the AzureML environment.
    * The ``device`` and ``execution_providers`` for AzureML system is mandatory. Otherwise, Olive will raise an error.

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

Native AzureML System
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
        "type": "AzureML",
        "accelerators": [
            {
                "device": "gpu",
                "execution_providers": [
                    "CUDAExecutionProvider"
                ]
            }
        ],
        "aml_compute": "gpu-cluster",
        "aml_docker_config": {
            "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
            "conda_file_path": "conda.yaml"
        },
        "aml_environment_config": {
            "name": "myenv",
            "version": "1"
        }
    }

AzureML Readymade Systems
"""""""""""""""""""""""""

There are some readymade systems available for AzureML. These systems are pre-configured with the necessary.

.. code-block:: json

    {
        "type": "AzureNDV2System",
        "accelerators": [
            {"device": "gpu", "execution_providers": ["CUDAExecutionProvider"]},
            {"device": "cpu", "execution_providers": ["CPUExecutionProvider"]},
        ],
        "aml_compute": "gpu-cluster",
        "aml_docker_config": {
            "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            "conda_file_path": "conda.yaml"
        }
    }

.. note::
    The accelerators specified in the readymade systems will be filtered against the devices supported by the readymade system. If the specified device is not supported by the readymade system, Olive will filter out the accelerator.
    In above example, the readymade system supports only GPU. Therefore, the final accelerators will be ``[{"device": "gpu", "execution_providers": ["CUDAExecutionProvider"]}]`` and the CPU will be filtered out.

Please refer to :ref:`olive_system_alias` for the list of supported AzureML readymade systems.


Managed AzureML System
^^^^^^^^^^^^^^^^^^^^^^

When :code:`olive_managed_env = True`, Olive will manage the AzureML environment by installing the required packages from the :code:`requirements_file` in the AzureML environment if provided.

From the time being, Olive only supports the following base Dockerfiles based on input execution providers:

* CPUExecutionProvider: (*Dockerfile.cpu*)
* CUDAExecutionProvider: (*Dockerfile.gpu*)
* TensorrtExecutionProvider: (*Dockerfile.gpu*)
* OpenVINOExecutionProvider: (*Dockerfile.openvino*)

A typical managed AzureML system can be configured by the following example:

.. code-block:: json

    {
        "systems": {
            "azureml_system": {
                "type": "AzureML",
                "accelerators": [
                    {
                        "device": "cpu",
                        "execution_providers": [
                            "CPUExecutionProvider",
                            "OpenVINOExecutionProvider"
                        ]
                    }
                ],
                "azureml_client_config": {
                    "subscription_id": "subscription_id",
                    "resource_group": "resource_group",
                    "workspace_name": "workspace_name"
                },
                "aml_compute": "cpu-cluster",
                "requirements_file": "mnist_requirements.txt",
                "olive_managed_env": true,
            }
        },
        "engine": {
            "target": "azureml_system",
        }
    }

Please refer to this `example <https://github.com/microsoft/Olive/blob/main/examples/bert/conda.yaml>`__
for :code:`"conda.yaml"`.

Please refer to :ref:`azureml_system_config` for more details on the config options.


Isolated ORT System
-------------------
The isolated ORT system represents the isolated ONNX Runtime environment in which the ``olive-ai`` is not installed. It can only be configured as a target system. The isolated ORT system is configured with the following attributes:

.. admonition:: Configuration

    * :code:`accelerators`: The list of accelerators that are supported by the system.
    * :code:`python_environment_path`: The path to the python virtual environment.
    * :code:`environment_variables`: The environment variables that are required to run the python environment. This is optional.
    * :code:`prepend_to_path`: The path that will be prepended to the PATH environment variable. This is optional.


.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "IsolatedORT",
                "python_environment_path": "/home/user/.virtualenvs/myenv/bin",
                "accelerators": [{"device": "cpu"}]
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.systems.isolated_ort import IsolatedORTSystem
            from olive.system.common import Device

            python_environment_system = IsolatedORTSystem(
                python_environment_path = "/home/user/.virtualenvs/myenv/bin",
                accelerators = [{"device": Device.CPU}]
            )

.. note::

    * Isolated ORT System does not support :code:`olive_managed_env` and can only be used to evaluate ONNX models.
    * The accelerators for Isolated ORT system is optional. If not provided, Olive will get the available execution providers installed in current virtual environment and infer its device.
    * For each accelerator, either ``device`` or ``execution_providers`` is optional but not both if the accelerators are specified. If ``device`` or ``execution_providers`` is not provided, Olive will infer the ``device`` or ``execution_providers`` if possible.

.. important::

    The Isolated ORT environment must have the relevant ONNX runtime package installed!

Please refer to :ref:`isolated_ort_system_config` for more details on the config options.
