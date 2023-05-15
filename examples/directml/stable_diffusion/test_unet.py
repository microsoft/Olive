from collections import defaultdict

import torch
from diffusers import UNet2DConditionModel
from functools import reduce

from huggingface_hub import model_info
from diffusers.utils.hub_utils import _get_model_file

def unet_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 4, 64, 64), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, 768), dtype=torch_dtype),
        "return_dict": False,
    }


def load_lora_weights(model_path):
    from diffusers.loaders import LORA_WEIGHT_NAME
    from diffusers.utils import DIFFUSERS_CACHE
    from diffusers.utils.hub_utils import _get_model_file

    model_file = _get_model_file(
        model_path,
        weights_name=LORA_WEIGHT_NAME,
        cache_dir=DIFFUSERS_CACHE,
        force_download=False,
        resume_download=False,
        proxies=None,
        local_files_only=False,
        use_auth_token=None,
        revision=None,
        subfolder=None,
        user_agent={
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        },
    )

    state_dict = torch.load(model_file, map_location="cpu")
    return state_dict


def merge_lora_weights(base_model, lora_model_id, submodel_name="unet", scale=1.0):
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.loaders import LORA_WEIGHT_NAME
    from diffusers.utils import DIFFUSERS_CACHE
    from collections import defaultdict
    from functools import reduce

    # Load LoRA weights
    model_file = _get_model_file(
        lora_model_id,
        weights_name=LORA_WEIGHT_NAME,
        cache_dir=DIFFUSERS_CACHE,
        force_download=False,
        resume_download=False,
        proxies=None,
        local_files_only=False,
        use_auth_token=None,
        revision=None,
        subfolder=None,
        user_agent={
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        },
    )
    lora_state_dict = torch.load(model_file, map_location="cpu")

    # All keys in the LoRA state dictionary should have 'lora' somewhere in the string.
    keys = list(lora_state_dict.keys())
    assert all("lora" in k for k in keys)

    if all(key.startswith(submodel_name) for key in keys):
        # New format (https://github.com/huggingface/diffusers/pull/2918) supports LoRA weights in both the
        # unet and text encoder where keys are prefixed with 'unet' or 'text_encoder', respectively.
        submodel_state_dict = {k: v for k, v in lora_state_dict.items() if k.startswith(submodel_name)}
    else:
        # Old format. Keys will not have any prefix.
        submodel_state_dict = lora_state_dict

    # Group LoRA weights into attention processors
    attn_processors = {}
    lora_grouped_dict = defaultdict(dict)
    for key, value in submodel_state_dict.items():
        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
        lora_grouped_dict[attn_processor_key][sub_key] = value

    for key, value_dict in lora_grouped_dict.items():
        rank = value_dict["to_k_lora.down.weight"].shape[0]
        cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
        hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

        attn_processors[key] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
        )
        attn_processors[key].load_state_dict(value_dict)

    # Merge LoRA attention processor weights into existing Q/K/V/Out weights
    for name, proc in attn_processors.items():
        attention_name = name[: -len(".processor")]
        attention = reduce(getattr, attention_name.split(sep="."), base_model)
        attention.to_q.weight.data += scale * torch.mm(proc.to_q_lora.up.weight, proc.to_q_lora.down.weight)
        attention.to_k.weight.data += scale * torch.mm(proc.to_k_lora.up.weight, proc.to_k_lora.down.weight)
        attention.to_v.weight.data += scale * torch.mm(proc.to_v_lora.up.weight, proc.to_v_lora.down.weight)
        attention.to_out[0].weight.data += scale * torch.mm(proc.to_out_lora.up.weight, proc.to_out_lora.down.weight)


if __name__ == "__main__":
    lora_model_id = "sayakpaul/sd-model-finetuned-lora-t4"
    lora_model_id = "CompVis/stable-diffusion-v1-4"
    info = model_info(lora_model_id)
    
    base_model_id = info.cardData.get("base_model", lora_model_id)

    print(f"base = {base_model_id}")
    print(f"lora = {lora_model_id}")

    # model_file = _get_model_file(
    #     lora_model_id,
    #     weights_name=LORA_WEIGHT_NAME,
    #     cache_dir=DIFFUSERS_CACHE,
    #     force_download=False,
    #     resume_download=False,
    #     proxies=None,
    #     local_files_only=False,
    #     use_auth_token=None,
    #     revision=None,
    #     subfolder=None,
    #     user_agent={
    #         "file_type": "attn_procs_weights",
    #         "framework": "pytorch",
    #     }
    # )

    # torch.manual_seed(0)
    # inputs = unet_inputs(1, torch_dtype=torch.float32)

    

    # # Diffusers API
    # from diffusers import StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)
    # pipe.load_lora_weights(lora_model_id)
    # unet_v1 = pipe.unet

    # # Merged version
    # unet_v2 = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    # merge_lora_weights(unet_v2, lora_model_id, "unet")

    # # Compare output
    # with torch.no_grad():
    #     out_v1 = unet_v1.forward(inputs["sample"], inputs["timestep"], inputs["encoder_hidden_states"])
    #     out_v2 = unet_v2.forward(inputs["sample"], inputs["timestep"], inputs["encoder_hidden_states"])
    #     # print(out_v1)
    #     # print(out_v2)
    #     print(torch.isclose(out_v1[0], out_v2[0]))

    # convert_unet(unet, "unet.onnx")
