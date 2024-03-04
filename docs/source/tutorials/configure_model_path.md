# How To Set Model Path

This document describes how to configure model paths in Olive for both local and remote model resources.

## Local Model Path
If the model path is a local path or a string name (such as a model hub id), it can be directly specified as a string. Olive will automatically
infer if it is a string name, local file or local directory.

```json
{
    "model_path": "my_model.pt"
}
```

You can also specify the resource type explicitly.

### Local File
```json
{
    "model_path": {
        "type": "file",
        "config": {
            "path": "my_model.pt"
        }
    }
}
```

### Local Folder
```json
{
    "model_path": {
        "type": "folder",
        "config": {
            "path": "my_model_dir"
        }
    }
}
```

### String Name
```json
{
    "model_path": {
        "type": "string_name",
        "config": {
            "name": "my_model"
        }
    }
}
```

## Remote Model Path
Olive supports remote model resources. Currently, it supports AzureML model, AzureML datastore and AzureML job output.

### AzureML Model
Models registered in an Azure Machine Learning workspace.

```json
{
    "model_path": {
        "type": "azureml_model",
        "config": {
            "azureml_client": {
                "subscription_id": "my_subscription_id",
                "resource_group": "my_resource_group",
                "workspace_name": "my_workspace"
            },
            "name": "my_model",
            "version": 1
        }
    }
}
```

### AzureML Registry Model
Models curated in an Azure Machine Learning or models in your own registry. Azure ML curated model doesn't require an `azureml_client` config section, but you can still add this section for additional `mlclient` configuration.

```json
{
    "model_path": {
        "type": "azureml_registry_model",
        "config": {
            "name": "model_name",
            "registry_name": "registry_name",
            "version": 1
        }
    }
}
```

### AzureML Datastore
Model files or folders stored in an Azure Machine Learning datastore.

```json
{
    "model_path": {
        "type": "azureml_datastore",
        "config": {
            "azureml_client": {
                "subscription_id": "my_subscription_id",
                "resource_group": "my_resource_group",
                "workspace_name": "my_workspace"
            },
            "datastore_name": "my_datastore",
            "relative_path": "model_dir/my_model.pt" // Relative path to the resource from the datastore root
        }
    }
}
```

### AzureML Job Output
Model files or folders generated by an Azure Machine Learning job and saved in the job output.

```json
{
    "model_path": {
        "type": "azureml_job_output",
        "config": {
            "azureml_client": {
                "subscription_id": "my_subscription_id",
                "resource_group": "my_resource_group",
                "workspace_name": "my_workspace"
            },
            "job_id": "my_job_id", // id of the job
            "output_name": "my_output_name", // name of the job output
            "relative_path": "model_dir/my_model.pt" // Relative path to the resource from the job output root
        }
    }
}
```

**Note**: If the workflow config file has `azureml_client` at the top level, `azureml_client` in the model path config can be omitted. The
workflow will automatically use the top level `azureml_client` if it is not specified in the model path config.

```json
{
    "azureml_client": {
        "subscription_id": "my_subscription_id",
        "resource_group": "my_resource_group",
        "workspace_name": "my_workspace"
    },
    "input_model": {
        "type": "PytorchModel",
        "config": {
            "model_path": {
                "type": "azureml_model",
                "config": {
                    "name": "my_model",
                    "version": 1
                }
            }
        }
    }
}
```