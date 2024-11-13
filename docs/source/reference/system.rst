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

.. _device:

Device
^^^^^^^

.. autoclass:: olive.hardware.accelerator.Device
    :members:
    :undoc-members:

.. _accelerator_config:

AcceleratorConfig
^^^^^^^^^^^^^^^^^^

.. autoclass:: olive.systems.common.AcceleratorConfig
    :members:
    :undoc-members:
    :exclude-members: validate_device_and_execution_providers

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

.. _isolated_ort_system_config:

IsolatedORTTargetUserConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autopydantic_settings:: olive.systems.system_config.IsolatedORTTargetUserConfig


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

IsolatedORTSystem
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: olive.systems.isolated_ort.IsolatedORTSystem

.. _olive_system_alias:

System Alias
------------------------

.. automodule:: olive.systems.system_alias
    :members:
    :undoc-members:
    :inherited-members:
    :exclude-members: AzureMLSystemAlias, SurfaceSystemAlias
