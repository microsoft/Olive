# Azure AI Integration

This documents outlines the integrations between Olive and Azure Machine Learning. Discover how to use your Azure Machine Learning assets within Olive.

## Azure Machine Learning client
If you will use Azure ML resources and assets, you need to provide your Azure ML client configurations. For example:

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
