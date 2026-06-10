# Shared Cache

## Overview

The Shared Cache allows Olive to store intermediate models in Azure Blob Storage. When a pass is executed, Olive checks if the same pass with the same input model has been processed before. If a matching output model is found in the shared cache, Olive reuses it for the next step. With this feature, you can easily share output models with others.

## Setup and Usage

### Install Dependencies

Install the required dependencies for enabling shared cache support:

```shell
pip install olive-ai[shared-cache]
```

### Configure Azure Blob Storage

Olive uses Azure Blob Storage for the shared cache. You need to set up an Azure Blob Storage container for this feature. Learn more about [Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction).

Once your container is ready, locate the account name and container name on your storage account page. The account URL typically looks like `https://<account_name>.blob.core.windows.net`.

Make sure you are logged in to Azure by running:

```shell
az login
```

### Configure the Shared Cache

To enable the shared cache, add the Azure Blob Storage account URL with container name to the cache_dir field in your configuration file:

```json
{
    //...
    "cache_dir": "https://<account_name>.blob.core.windows.net/<container_name>",
    //...
}
```

If you want to custom local cache directory also, you can provide multiple `cache_dir`:

```json
{
    //...
    "cache_dir": [
      "local_cache",
      "https://myaccountname.blob.core.windows.net/mycontainername"
    ],
    //...
}
```

When you set up the shared cache this way, `update_shared_cache` will default to `True`.

### Alternative Configuration

You can also configure the shared cache using the `cache_config`:

* `cache_config`:
  * `account_name [str]`: The account name from the Azure Storage account.
  * `container_name [str]`: The container name to store model cache.
  * `update_shared_cache [bool]`: If True, Olive will upload output models to the shared cache. Defaults to True.

Here is an example configuration:

```json
{
    //...
    "cache_config": {
        "account_name": "account-name",
        "container_name": "container-name",
        "update_shared_cache": true
    },
    //...
}
```

With this configuration, Olive will check if a cached model exists in your blob container. If you only want to download cached models from the blob without uploading the output models, set `update_shared_cache` to `false`.

## Important Notes

* The Shared Cache feature does not support models with Callable attributes, such as `model_loader`, `io_config`, or `dummy_inputs_func`, if any of these is a Callable, shared cache will be disabled.
* If the output model is found in the shared cache and evaluation is needed, Olive will download the output model from the shared cache.
* If no evaluation is required, Olive skips downloading intermediate models if cached models are found for the next pass.
* If the output model is found in the shared cache and no evaluation is needed, Olive will check for cached models for the next Pass instead of immediately downloading the current cached model. If a cached model for the next Pass is found, Olive will skip downloading the intermediate model and proceed directly to the next Pass. This process will continue until the final Pass is completed or no cached model is found.
* When this feature is enabled, the workflow running time may be affected, potentially making it faster or slower. Since Olive will download output models, factors that can influence this include the size of the output model, internet speed, the number of Passes, and how many intermediate models are cached in the shared cache.
