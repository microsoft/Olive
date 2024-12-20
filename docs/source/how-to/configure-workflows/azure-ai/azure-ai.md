# Azure AI Integration

This documents outlines the integrations between Olive and Azure Machine Learning. Discover how to use your Azure Machine Learning assets within Olive.

## Azure Machine Learning client
If you will use Azure ML resources and assets, you need to provide your Azure ML client configurations. For example:

* You have AzureML system for targets or hosts.
* You have Azure ML model as input model.

You can set your client in your Olive configuration using:

```json
"azureml_client": {
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>",
    "read_timeout" : 4000,
    "max_operation_retries" : 4,
    "operation_retry_interval" : 5
},
```

Where:

- `subscription_id: [str]` Azure account subscription id.
- `resource_group: [str]` Azure account resource group name.
- `workspace_name: [str]` Azure ML workspace name.
- `aml_config_path: [str]` The path to Azure config file, if Azure ML client config is in a separate file.
- `read_timeout: [int]` read timeout in seconds for HTTP requests, user can increase if they find the default value too small. The default value from azureml sdk is 3000 which is too large and cause the evaluations and pass runs to sometimes hang for a long time between retries of job stream and download steps.
- `max_operation_retries: [int]` The maximum number of retries for Azure ML operations like resource creation and download.
The default value is 3. User can increase if there are network issues and the operations fail.
- `operation_retry_interval: [int]` The initial interval in seconds between retries for Azure ML operations like resource creation and download. The interval doubles after each retry. The default value is 5. User can increase if there are network issues and the operations fail.
- `default_auth_params: Dict[str, Any]` Default auth parameters for AzureML client. Please refer to [azure DefaultAzureCredential](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python#parameters) for more details. For example, if you want to exclude managed identity credential, you can set the following:

### Using AzureML registered model
You can run Olive workflow with your AML workspace registered model. In the input model section, define the model config as:
```json
"model_path": {
    "type": "azureml_model",
    "name": "<model_name>",
    "version": "<model_version>"
 }
```
Olive will automatically download the model and run the workflow in the specified target or host with this model as input model.


### Using AzureML curated model
You can run Olive workflow with AML registered model. In the input model section, define the model config as:
```json
"model_path": {
    "type": "azureml_registry_model",
    "name": "model_name",
    "registry_name": "registry_name",
    "version": 1
}
```
Olive will automatically download the model and run the workflow in the specified target or host with this model as input model.

Note: you don't need the `azureml_client` section for AzureML curated model.

### Using model stored in AzureML datastore
You can specify your model path from an AzureML datastore as:
```json
"model_path": {
    "type": "azureml_datastore",
    "azureml_client": {
        "subscription_id": "my_subscription_id",
        "resource_group": "my_resource_group",
        "workspace_name": "my_workspace"
    },
    "datastore_name": "my_datastore",
    "relative_path": "model_dir/my_model.pt" // Relative path to the resource from the datastore root
}
```

### Using a model from an AzureML job output
You can specify your model path from an AzureML job output as:
```json
"model_path": {
    "type": "azureml_job_output",
    "azureml_client": {
        "subscription_id": "my_subscription_id",
        "resource_group": "my_resource_group",
        "workspace_name": "my_workspace"
    },
    "job_id": "my_job_id", // id of the job
    "output_name": "my_output_name", // name of the job output
    "relative_path": "model_dir/my_model.pt" // Relative path to the resource from the job output root
}
```

### Using data stored in AzureML datastore
You can use data files or folders that are stored in your Azure ML datastore as:
```json
"data_dir": {
    "type": "azureml_datastore",
    "azureml_client": {
        "subscription_id": "my_subscription_id",
        "resource_group": "my_resource_group",
        "workspace_name": "my_workspace"
    },
    "datastore_name": "my_datastore",
    "relative_path": "data_dir" // Relative path to the resource from the datastore root
}
```

### Using Azure ML compute as host or target
You can specify your Azure ML Compute as an Olive System and use it as a host to run Pass, or a target to evaluate the model.

```json
"systems": {
    "aml_system": {
        "type": "AzureML",
        "aml_compute": "cpu-cluster",
        "aml_docker_config": {
            "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
            "conda_file_path": "conda.yaml"
        }
    }
}
```

Olive supports all the compute types of Azure ML, including compute via Azure Arc.

Then you can specify where to use it in the Engine config:
```json
{
    "host": "aml_system",
    "target": "aml_system",
}
```
