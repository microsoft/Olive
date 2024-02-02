from olive.data.component.dataloader import default_dataloader
from olive.data.registry import Registry
from olive.platform_sdk.qualcomm.utils.data_loader import FileListCommonDataLoader


@Registry.register_dataloader()
def huggingface_data_to_raw_dataloader(dataset, batch_size=1, io_config=None, **kwargs):
    data_loader = default_dataloader(dataset, batch_size=batch_size, **kwargs)
    return FileListCommonDataLoader(data_loader, batch_size=batch_size, io_config=io_config, **kwargs)


@Registry.register_post_process()
def snpe_post_process(output_data, **kwargs):
    import torch

    logits = torch.tensor(output_data["results"]["logits"])
    _, preds = torch.max(logits, dim=1)

    return preds
