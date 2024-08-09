# Remote Workflow

## Running Olive Workflow Remotely on Azure Machine Learning workspace compute

If your Olive workflow takes a long time to run or you need to step away or your local environment is not powerful enough to run the workflow then Olive provides the flexibility to run your workflow on a compute or cluster in your Azure Machine Learning workspace. This allows you to turn off your terminal session or shut down your local machine without interrupting the process.

## Install Extra Dependencies

You can install the necessary dependencies by running:

```shell
pip install olive-ai[azureml]
```

## Configure an AzureML system

To run the workflow on AzureML system, the configuration is same as normal AzureML system configuration. Please find more details in [this instruction](../tutorials/configure_systems.rst).

## Run Olive Workflow

To run the Olive workflow, add a `workflow_host` configuation at the top level of the Olive config file:

```json
"workflow_host": "aml_system"
```

Then run the Olive workflow with the following command:

```shell
python -m olive run --config <config_file.json>
```

Olive will submit this workflow to the Azure ML system compute. It is safe to close your terminal session or turn off your local machine. The workflow will continue running on the Azure ML workspace compute.

## Workflow outputs

The artifacts of the workflow, including cache and outputs, will be automatically exported to Azure Machine Learning datastore. The default datastore is `workspaceblobstore`. If you want to export to your own datastore, please add `datastore: <your-datastore-name>` in AzureML system config:

```json
"aml_system": {
    "type": "AzureML",
    "config": {
        "datastore": "<datastore-name>"
    }
}
```

The cache and outputs will be exported to `<datastore>/workflow_artifacts/<workflow_id>/cache` and `<datastore>/workflow_artifacts/<workflow_id>/output`. You can check logs to find the detailed output path.
