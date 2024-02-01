An example of mistral model optimization using olive workflows.

- CPU: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model*

## Prerequisites
* transformers>=4.34.99
* optimum
* neural-compressor>=2.4.1
* ort-nightly

## Installation
```bash
conda create -n olive python=3.8 -y
conda activate olive
git clone https://github.com/microsoft/Olive.git
cd Olive
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -e .
cd examples/mistral
pip install -r requirements.txt
# manually install the nightly ORT
pip install ort-nightly==1.17.0.dev20231225002 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

In above steps, please run the following command in Administrator command prompt if you hit "Filename too long" when installing the packages.
```bash
git config --system core.longpaths true
```

## Usage
```bash
python -m olive.workflows.run --config mistral_optimize.json
```

### Local model
if the input model is saved locally, you can specify the configuration like the following:
```json
{
    "input_model": {
        "type": "PyTorchModel",
        "config": {
            "model_path": "C:/git/Olive/examples/mistral/mistral-7b2"
        }
    },
    //...
    "passes": {
        "convert": {
            "type": "OptimumConversion",
            "config": {
                "target_opset": 14,
                "extra_args": {
                    "legacy": false,
                    "no_post_process": false,
                    "task": "text-generation-with-past"
                }
            }
        }
    }
    //...
}
```

## Known issues
From the time being, the latency for sequence length larger than 1 will be worse than that of original model if the int4 quantized model is running in CPU. So, we'd suggest to run the int4 quantized model in GPU for better performance.

To make sure int4 quantized model running in GPU, please start with the example by changing the accelerator/EP to GPU/CUDA in the config file.

The following table show the latency comparison between original model and int4 quantized model with different accuracy level when running in my CPU (AMD EPYC 7763) with sequence length 32.
| Model | Average Latency in ms |
| --- | --- |
| Original | 944.14496 |
| int4 quantized with accuracy level 0 | 1663.0327 |
| int4 quantized with accuracy level 4 | 1743.15224 |
