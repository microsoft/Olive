#  Optimizations via Parallelism
Sample use cases of Olive to optimize a LLMs using parallelization techniques including tensor & pipeline parallelism.
Implementation uses transformation network from [orchard](https://github.com/devang-ml/orchard) to generate ranked models
using either pipeline parallelism or tensor parallelism. These ranked models can then be used to run a regular inferencing session.

Supported models include the following -
- [Llama2 7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Llama2 13b](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [Llama2 70b](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [Llama2 7b chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [Llama2 13b chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [Llama2 70b chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [Mistral 7b](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Gemma 2b](https://huggingface.co/google/gemma-2b)
- [Gemma 7b](https://huggingface.co/google/gemma-7b)

## Prerequisites
### Clone the repository and install Olive
Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Install dependencies
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Prepare a HuggingFace model for Olive
This optimization technique only support HF model indirectly i.e. by converting the model to PyTorch weights only format.
Orchard includes a script to load HF model and generate compatible output. Use the following command to download &
generate a compatible output model.
```bash
./orchard/utils/prepare.sh <model_name>
```

For example, to convert the _gemma-2b_ model, use the following command -
```bash
./orchard/utils/prepare.sh google/gemma-2b
```

The shell script will generate the model as _checkpoints/google/gemma-2b/model.pth_.

## Run the config to optimize the model
You can optionally generate the config file by running the following command:
```bash
python run.py --model_name meta-llama/Llama-2-7b-hf --mode [pp|tp] --world_size <n> --only_config
```
Or you can skip the _only_config_ argument to run the Olive pipeline to generate the ranked models.

For example, to generate the ranked models for _gemma-2b_ model:
```bash
# run to optimize the model: FP16/INT4
python run.py --model_name google/gemma-2b --mode pp --world_size 3
```
The above command will generate three new models in the output folder.

**Note:**
- When generating ranked models for tensor parallel, the number of attention heads in the model must be divisible by the input _world\_size_.
- When generating ranked models for pipeline parallel, the number of layers in the model must be divisible by the input _world\_size_.

## Run an inferencing session
Inference session is run using the orchard module.
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun                           \
    --standalone                                            \
    --nproc_per_node=3                                      \
    examples/llama2_driver.py                               \
    --checkpoint_path checkpoints/google/gemma-2b/model.pth \
    --model_path "gemma-2b-pp/model_{:02d}.pt"              \
    --pp                                                    \
    --prompt "What's an apple?"
```

# License
Please see the [LICENSE](./LICENSE) file for more details. Also please follow the [user policy](./USE-POLICY-META-LLAMA-2.md) of the model provider. Besides, please refer to the [Responsible
Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/) for more details on how to use the model responsibly.
