# Azure ML scripts

Olive provides a couple of scripts to help you manage your Azure ML assets.

## Scripts list

### `manage_compute_instance`
This Python script provides a command-line interface for managing compute resources in an Azure Machine Learning workspace.

* `--create` or `-c`: A flag indicating that a new compute resource should be created. This is mutually exclusive with `--delete` - only one of them can be specified at a time.

* `--delete` or `-d`: A flag indicating that an existing compute resource should be deleted. This is mutually exclusive with `--create` - only one of them can be specified at a time.

* `--subscription_id`: The ID of your Azure subscription.

* `--resource_group`: The name of your Azure resource group.

* `--workspace_name`: The name of your Azure Machine Learning workspace.

* `--aml_config_path`: The path to your AzureML config file. If this is provided, subscription_id, resource_group and workspace_name are ignored.

* `--compute_name`: The name of the new compute resource. This is a required argument.

* `--vm_size`: The VM size of the new compute resource. This is required if you are creating a compute instance.

* `--location`: The location where the new compute resource should be created. This is required if you are creating a compute instance.

* `--min_nodes`: The minimum number of nodes for the new compute resource. If this is not provided, the default value is 0.

* `--max_nodes`: The maximum number of nodes for the new compute resource. If this is not provided, the default value is 2.

* `--idle_time_before_scale_down`: The number of idle seconds before the compute resource scales down. If this is not provided, the default value is 120 seconds.

`aml_config_path` is a json file for your azureml config:
```json
{
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>",
}
```

#### Usage

You can use ``olive manage-aml-compute`` command line tool to create an AzureML compute instance from the command line like this:

```bash
olive manage-aml-compute --create --subscription_id <subscription_id> --resource_group <resource_group> --workspace_name <workspace_name> --compute_name <compute_name> --vm_size <vm_size> --location <location> --min_nodes <min_nodes> --max_nodes <max_nodes> --idle_time_before_scale_down <idle_time_before_scale_down>
```

or

```bash
olive manage-aml-compute --create --aml_config_path </path/to/aml_config.json> --compute_name <compute_name> --vm_size <vm_size> --location <location> --min_nodes <min_nodes> --max_nodes <max_nodes> --idle_time_before_scale_down <idle_time_before_scale_down>
```

You can delete an AzureML compute instance by:
```bash
olive manage-aml-compute --delete --compute_name <compute_name>
```

More details can be found at [Command Line Tools](../../../reference/cli.rst).
