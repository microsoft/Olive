ResourcePath
=============

The following are the resource paths available in Olive.

ResourceType
------------
.. autoclass:: olive.resource_path.ResourceType
    :members:
    :undoc-members:

Each resource path is followed by a description of the path and a list of the its configuration options.

LocalFile
^^^^^^^^^
.. autoconfigclass:: olive.resource_path.LocalFile

LocalFolder
^^^^^^^^^^^
.. autoconfigclass:: olive.resource_path.LocalFolder

StringName
^^^^^^^^^^
.. autoconfigclass:: olive.resource_path.StringName

AzureMLModel
^^^^^^^^^^^^
.. autoconfigclass:: olive.resource_path.AzureMLModel

**AzureMLClientConfig**

.. autopydantic_settings:: olive.azureml.azureml_client.AzureMLClientConfig

AzureMLDatastore
^^^^^^^^^^^^^^^^
.. autoconfigclass:: olive.resource_path.AzureMLDatastore

AzureMLJobOutput
^^^^^^^^^^^^^^^^
.. autoconfigclass:: olive.resource_path.AzureMLJobOutput
