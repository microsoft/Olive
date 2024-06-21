# Running Olive Workflow Remotely on an Azure Virtual Machine

If your workflow takes a long time to run or you need to step away, Olive provides the flexibility to run your workflow on a cloud virtual machine. This allows you to turn off your terminal session or shut down your local machine without interrupting the process. You can later retrieve artifacts and logs at your convenience.

This documentation demonstrates how to configure the Cloud System, run the workflow, and retrieve the results of the remote workflow.

## Install Extra Dependencies

You can install the necessary dependencies by running:

```shell
pip install olive-ai[cloud]
```

or

```shell
pip install paramiko
```

## Configure Cloud System

The Cloud System is a type of Olive system specifically designed for running workflows remotely on an Azure Virtual Machine. To configure the cloud system, you will need to provide the following attributes:

* `accelerators: [List[str]]`
  * The list of accelerators that are supported by the cloud system.
* `hf_token: [bool]`
  * Whether to use the Hugging Face token. This is optional and defaults to False.
* `olive_path: [str]`
  * The path to run the Olive workflow in the cloud system. The `cache_dir` and `output_dir` will be under this path. For example, the `olive_path` is `/home/ucm/olive`, and the `cache_dir` and `output_dir` in the config file are `cache` and `outputs`, then the cache will be saved in `/home/ucm/olive/cache` and the outputs will be saved in `/home/ucm/olive/outputs` in the cloud virtual machine.
* `conda_path: [str]`
  * The path to conda to activate the environment in this session. Olive will run `source <conda_path>` to activate your conda. For example, the path could be `/opt/conda/etc/profile.d/conda.sh`.
* `conda_name: [str]`
  * The name of the conda environment to activate. Olive should be installed in this conda environment to run the workflow.
* `hostname: [str]`
  * The hostname of the cloud virtual machine. Can be an IP address or a domain name.
* `username: [str]`
  * The username to login to the cloud virtual machine.
* `os: [str]`
  * The operating system of the cloud virtual machine. Can be "Linux" or "Windows".
* `key_filename: [str]`
  * The path to the private key file used to log in to the cloud virtual machine. For example, the path could be `C:\Users\ucm\.ssh\key.pem` on Windows or `/home/ucm/.ssh/key.pem` on Linux.
* `password: [str]`
  * The password to login to the cloud virtual machine. This is optional when `key_filename` is provided.

Here is an example cloud system configuration:

```
"cloud_system": {
    "type": "Cloud",
    "config": {
        "olive_path": "/home/ucm/olive-remote-wfl",
        "conda_path": "/opt/conda/etc/profile.d/conda.sh",
        "conda_name": "olive",
        "hostname": "8.8.8.8",
        "os": "Linux",
        "username": "ucm",
        "key_filename": "C:\Users\ucm\.ssh\key.pem",
        "accelerators": [
            {
                "device": "cpu",
                "execution_providers": [
                    "CPUExecutionProvider",
                    "OpenVINOExecutionProvider"
                ]
            }
        ]
    }
}
```

## Run Olive Workflow

To run the Olive workflow on the cloud system, configure the engine to use the cloud system as the host:

```
"engine": {
    ...
    "host": "cloud_system",
    "target": "local_system",
    "cache_dir": "cache",
    "output_dir": "models"
}
```

Then run the Olive workflow with the following command:

```shell
python -m olive run --config <config_file.json>
```

Olive will submit this workflow to the remote host `8.8.8.8`, and save cache and logs to path `/home/ucm/olive-remote-wfl`. While the logs are streaming, it is safe to close your terminal session or turn off your local machine. The workflow will continue running on the cloud virtual machine.

If you keep your session open, when the workflow completes, Olive will download the output artifacts and logs to your local machine in the configured cache and output directories.

## Retrieve Workflow Artifacts and Logs to Local

If you turn off your local session or machine, you can retrieve the workflow logs and artifacts from the cloud virtual machine by adding the `--retrieve` flag to the workflow run command:

```shell
python -m olive run --config <config_file.json> --retrieve
```

Olive will retrieve the logs and artifacts from the cloud virtual machine using the `workflow_id` specified in the config file. If no `workflow_id` is specified, Olive will retrieve the default workflow `default_workflow`.

If the remote workflow is still running, a log message `Workflow is still running. Please wait for completion.` will be printed. Once the workflow is completed, Olive will download the output artifacts and logs to your local machine in the configured cache and output directories.

## Notes

* If you specify the cloud system as `host` in the engine config, Olive will run the workflow as `local system` on the cloud virtual machine.
* If you specify the cloud system as `host` in the engine config while specifying either cloud system or local system as target in the engine config, Olive will use `local system` as target on the cloud virtual machine.
* To track the workflow running logs on the cloud virtual machine, Olive will automatically enable `log_to_file` for this workflow run. The log file will be saved to `<cache_dir>/<workflow_id>/<workflow_id>.log`.
* Currently, only Azure Virtual Machine is tested for this feature. Support for other cloud service virtual machines will be tested soon.
