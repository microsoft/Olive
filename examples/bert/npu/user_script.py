from olive.data.registry import Registry


@Registry.register_post_process()
def snpe_post_process(output_data, **kwargs):
    import torch

    logits = torch.tensor(output_data["logits"])
    _, preds = torch.max(logits, dim=-1)

    return preds
