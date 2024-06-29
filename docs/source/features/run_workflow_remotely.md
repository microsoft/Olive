# Running Olive Workflow Remotely on an Azure Virtual Machine

If your workflow takes a long time to run or you need to step away, Olive provides the flexibility to run your workflow on a remote virtual machine. This allows you to turn off your terminal session or shut down your local machine without interrupting the process. You can later retrieve artifacts and logs at your convenience.

This documentation demonstrates how to configure the remote dispactcher, run the workflow, and retrieve the results of the remote workflow.

## Install Extra Dependencies

You can install the necessary dependencies by running:

```shell
pip install olive-ai[dispactcher]
```

or

```shell
pip install paramiko
```

## Configure a Remote Dispatcher

The Remote Dispatcher is a type of dispatcher specifically designed for running workflows remotely on an Azure Virtual Machine. To configure the remote dispatcher, you will need to provide the following attributes:

* `type: [str]`
  * Currently we only support `Remote` as dispatcher type.
* `config_path: [str]`
  * The path to the dispatcher configuration file.

Dispatcher configuration file is a json file that includes configurations for the dispatcher. As for Remote Dispatcher, you should include the following attributes:

* `workflow_path: [str]`
  * The path to run the Olive workflow on the remote dispatcher. The `cache_dir` and `output_dir` will be under this path. For example, the `olive_path` is `/home/ucm/olive`, and the `cache_dir` and `output_dir` in the config file are `cache` and `outputs`, then the cache will be saved in `/home/ucm/olive/cache` and the outputs will be saved in `/home/ucm/olive/outputs` on the remote dispatcher.
* `conda_path: [str]`
  * The path to conda to activate the environment in this session. Olive will run `source <conda_path>` to activate your conda. For example, the path could be `/opt/conda/etc/profile.d/conda.sh`.
* `conda_name: [str]`
  * The name of the conda environment to activate. Olive should be installed in this conda environment to run the workflow.
* `hostname: [str]`
  * The hostname of the remote dispatcher. Can be an IP address or a domain name.
* `username: [str]`
  * The username to login to the remote dispatcher.
* `os: [str]`
  * The operating system of the remote dispatcher. Can be "Linux" or "Windows".
* `key_filename: [str]`
  * The path to the private key file used to log in to the remote dispatcher. For example, the path could be `C:\Users\ucm\.ssh\key.pem` on Windows or `/home/ucm/.ssh/key.pem` on Linux.
* `password: [str]`
  * The password to login to the remote dispatcher. This is optional when `key_filename` is provided.

Here is an example of remote dispatcher configuration file:

```(json)
{
    "workflow_path": "/home/ucm/olive",
    "conda_path": "/home/ucm/conda",
    "conda_name": "olive",
    "hostname": "8.8.8.8",
    "os": "Linux",
    "username": "ucm",
    "key_filename": "/home/ucm/.ssh/key.pem"
}
```

## Run Olive Workflow

To run the Olive workflow, add a dispatcher configuation at the top level of the Olive config file:

```(json)
"dispatcher": {
    "type": "Remote",
    "config_path": "config.json"
}
```

Then run the Olive workflow with the following command:

```shell
python -m olive run --config <config_file.json>
```

Olive will submit this workflow to the remote host `8.8.8.8`, and save cache and logs to path `/home/ucm/olive`. While the logs are streaming, it is safe to close your terminal session or turn off your local machine. The workflow will continue running on the remote dispatcher.

If you keep your session open, when the workflow completes, Olive will download the output artifacts and logs to your local machine in the configured cache and output directories.

## Retrieve Workflow Artifacts and Logs to Local

If you turn off your local session or machine, you can retrieve the workflow logs and artifacts from the remote dispatcher by adding the `--retrieve` flag to the workflow run command:

```shell
python -m olive run --config <config_file.json> --retrieve
```

Olive will retrieve the logs and artifacts from the remote dispatcher using the `workflow_id` specified in the config file. If no `workflow_id` is specified, Olive will retrieve the default workflow `default_workflow`.

If the remote workflow is still running, a log message `Workflow is still running. Please wait for completion.` will be printed. Once the workflow is completed, Olive will download the output artifacts and logs to your local machine in the configured cache and output directories.

## Notes

* To track the workflow running logs on the remote dispatcher, Olive will automatically enable `log_to_file` for this workflow run. The log file will be saved to `<cache_dir>/<workflow_id>/<workflow_id>.log`.
* Currently, only Azure Virtual Machine is tested for this feature. Support for other cloud service virtual machines will be tested soon.
