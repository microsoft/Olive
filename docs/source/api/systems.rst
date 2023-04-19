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
