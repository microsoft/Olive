<<<<<<< HEAD
.. _systems:
OliveSystems
=================================
The following systems are available in Olive.

Config
--------

.. autopydantic_settings:: olive.systems.system_config.SystemConfig

SystemType
^^^^^^^^^^^

.. autoclass:: olive.systems.common.SystemType
    :members:
    :undoc-members:

.. _local_system_config:
LocalTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.LocalTargetUserConfig

**Device**

.. autoclass:: olive.systems.common.Device
    :members:
    :undoc-members:

.. _docker_system_config:
DockerTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.DockerTargetUserConfig

**LocalDockerConfig**

.. autopydantic_settings:: olive.systems.docker.LocalDockerConfig

.. _azureml_system_config:
AzureMLTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.AzureMLTargetUserConfig

**AzureMLDockerConfig**

.. autopydantic_settings:: olive.systems.azureml.AzureMLDockerConfig

.. _python_environment_system_config:
PythonEnvironmentTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.PythonEnvironmentTargetUserConfig


Classes
---------

LocalSystem
^^^^^^^^^^^
.. autoclass:: olive.systems.local.LocalSystem

AzureMLSystem
^^^^^^^^^^^^^
.. autoclass:: olive.systems.azureml.AzureMLSystem


DockerSystem
^^^^^^^^^^^^^
.. autoclass:: olive.systems.docker.DockerSystem

PythonEnvironmentSystem
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: olive.systems.python_environment.PythonEnvironmentSystem
=======
.. _systems:

OliveSystems
=================================
The following systems are available in Olive.

Config
--------

.. autopydantic_settings:: olive.systems.system_config.SystemConfig

SystemType
^^^^^^^^^^^

.. autoclass:: olive.systems.common.SystemType
    :members:
    :undoc-members:

.. _local_system_config:

LocalTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.LocalTargetUserConfig


.. _docker_system_config:

DockerTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.DockerTargetUserConfig

**LocalDockerConfig**

.. autopydantic_settings:: olive.systems.docker.LocalDockerConfig

.. _azureml_system_config:

AzureMLTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.AzureMLTargetUserConfig

**AzureMLDockerConfig**

.. autopydantic_settings:: olive.systems.azureml.AzureMLDockerConfig

.. _python_environment_system_config:

PythonEnvironmentTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.PythonEnvironmentTargetUserConfig


Classes
---------

LocalSystem
^^^^^^^^^^^
.. autoclass:: olive.systems.local.LocalSystem

AzureMLSystem
^^^^^^^^^^^^^
.. autoclass:: olive.systems.azureml.AzureMLSystem


DockerSystem
^^^^^^^^^^^^^
.. autoclass:: olive.systems.docker.DockerSystem

PythonEnvironmentSystem
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: olive.systems.python_environment.PythonEnvironmentSystem
>>>>>>> 5ec0a52c973f1addd2a0491e2fdf38d5e2b56224
