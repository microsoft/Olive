# Azure ML integration

This documents outlines the integrations between Olive and Azure Machine Learning. Discover how to use your Azure Machine Learning assets within Olive.

## Olive Core
### Using AzureML registered model
You can run Olive workflow with your AML workspace registered model. In the input model section, define the model config as:
```
"model_path": {
    "type": "azureml_model",
    "config": {
        "name": "<model_name>",
        "version": "<model_version>"
    }
 }
```
Olive will automatically download the model and run the workflow in the specified target or host with this model as input model.

### Using model stored in AzureML datastore
You can specify your model path from an AzureML datastore as:
```
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
```

### Using a model from an AzureML job output
You can specify your model path from an AzureML job output as:
```
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
```

### Using data stored in AzureML datastore
You can use data files or folders that are stored in your Azure ML datastore as:
```
"data_dir": {
    "type": "azureml_datastore",
    "config": {
        "azureml_client": {
            "subscription_id": "my_subscription_id",
            "resource_group": "my_resource_group",
            "workspace_name": "my_workspace"
        },
        "datastore_name": "my_datastore",
        "relative_path": "data_dir" // Relative path to the resource from the datastore root
    }
}
```

### Using Azure ML compute as host or target
You can specify your Azure ML Compute as an Olive System and use it as a host to run Pass, or a target to evaluate the model.
```
"systems": {
    "aml_system": {
        "type": "AzureML",
    	 "config": {
            "aml_compute": "cpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                "conda_file_path": "conda.yaml"
            }
         }  
    }
}
```
Then you can specify where to use it in the Engine config:
```
{
    "host": "aml_system",
    "target": "aml_system",
}
```

## Connect your own device to Azure ML as target or host by Azure Arc
If you have your own device, you have the option to link it to your Azure ML Workspace as a Compute via Azure Arc. This allows you to use it as a Target for evaluating your model.

Please follow this instruction to setup your local device: [Self-hosted Kubernetes cluster](../tutorials/azure_arc.md)


## Azure ML Helper Scripts
Olive offers several scripts to assit you in managing your Azure ML assets.

For details on the available scripts, refer to: [Azure ML scripts](../tutorials/azureml_scripts.md)
