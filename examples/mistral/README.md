An example of mistral model optimization using olive workflows.

- CPU: *PyTorch Model -> Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model*

## Prerequisites
* transformers>=4.34.99
* optimum>1.17.0
* neural-compressor>=2.4.1
* onnxruntime>=1.17.0 or onnxruntime-gpu>=1.17.0
* onnxruntime-genai or onnxruntime-genai-cuda

## Installation
```bash
conda create -n olive python=3.12 -y
conda activate olive
git clone https://github.com/microsoft/Olive.git
cd Olive
pip install -e .
cd examples/mistral
pip install -r requirements.txt
```

In above steps, please run the following command in Administrator command prompt if you hit "Filename too long" when installing the packages.
```bash
git config --system core.longpaths true
```

## Run Optimization
CPU:
```bash
python mistral.py --optimize --config mistral_int4.json
```

GPU:
```bash
python mistral.py --optimize --config mistral_fp16.json
```

**NOTE:** You can run the optimization for a locally saved model by setting the `--model_id` to the path of the model.

## Test Inference
To test inference on the model run the script with `--inference`
```bash
python mistral.py --config mistral_fp16.json --inference
```

**NOTE:**
- You can provide you own prompts using `--prompt` argument. For example:
```bash
python mistral.py --config mistral_fp16.json --inference --prompt "Language models are very useful" "What is the meaning of life?"
```
- `--max_length` can be used to specify the maximum length of the generated sequence.
- Use `CUDA_VISIBLE_DEVICES` to specify the GPU to run the inference on. For example:
```bash
CUDA_VISIBLE_DEVICES=6 python mistral.py --config mistral_fp16.json --inference
```

## Known issues
From the time being, the latency for sequence length larger than 1 will be worse than that of original model if the int4 quantized model is running in CPU. So, we'd suggest to run the int4 quantized model in GPU for better performance.

To make sure int4 quantized model running in GPU, please start with the example by changing the EP to CUDA in the config file.

The following table show the latency comparison between original model and int4 quantized model with different accuracy level when running in an AMD EPYC 7763 CPU with sequence length 32.
| Model | Average Latency in ms |
| --- | --- |
| Original | 944.14496 |
| int4 quantized with accuracy level 0 | 1663.0327 |
| int4 quantized with accuracy level 4 | 1743.15224 |
