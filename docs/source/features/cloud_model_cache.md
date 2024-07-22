# Cloud Model Cache

## What is Cloud Model Cache

The Cloud Model Cache is a system where Olive stores intermediate models in Azure Blob Storage. Whenever a Pass is executed, Olive checks if the same input model has been processed before and if the corresponding output model is available in Azure Blob Storage. If the output model is found, Olive directly uses it for the next step. With this feature, you can easily share output models with others.

This feature currently only supports native Huggingface models as input.

## How to use Cloud Model Cache

### Install Extra Dependencies

You can install the necessary dependencies by running:

```shell
pip install olive-ai[azureml]
```

### Azure Blob Storage

Olive uses Azure Blob Storage for the Cloud Model Cache, so you'll need a prepared container for this feature. Learn more about Azure Blob Storage [here](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction).

Once your container is ready, locate the account URL and container name on your storage account page. The account URL typically looks like `https://<account_url>.blob.core.windows.net`.

Please make sure you have logged in Azure with command:

```shell
az login
```

### Cloud Cache Configuration

Add `cloud_cache_config` to the configuration:

* `cloud_cache_config`:
  * `enable_cloud_cache [bool]`: Whether to enable cloud cache. Default by `True`.
  * `account_url [str]`: The account url from the Azure Storage account.
  * `container_name [str]`: The container name to store model cache.
  * `upload_to_cloud [bool]`: Whether to upload output model to the cloud cache for this workflow run. Default by `True`.

Here is an example configuration:

```json
{
    //...
    "cloud_cache_config": {
        "account_url": "https://<account_url>.blob.core.windows.net",
        "container_name": "olivecachemodel"
    },
    //...
}
```

With this configuration, Olive will check if a cached model exists in your blob container. If you only want to download cached models from the blob without uploading the output models, set `upload_to_cloud` to `false` in the `cloud_cache_config`.

## Notes

* If the output model is found in the cloud cache and evaluation is needed, Olive will download the output model from the cloud cache.
* If the output model is found in the cloud cache and no evaluation is needed, Olive will check for cached models for the next Pass instead of immediately downloading the current cached model. If a cached model for the next Pass is found, Olive will skip downloading the intermediate model and proceed directly to the next Pass. This process will continue until the final Pass is completed or no cached model is found.
* When this feature is enabled, the workflow running time may be affected, potentially making it faster or slower. Since Olive will download output models, factors that can influence this include the size of the output model, your internet speed, the number of Passes, and how many intermediate models are cached in the cloud cache.
