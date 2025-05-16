# CLIP model Quantization

## Environment Setup

Install uv following the [instructions](https://docs.astral.sh/uv/#installation).

```bash
# install dependencies
cd /path/to/olive/examples/clip/qdq
uv sync

# activate the virtual environment
.venv/Scripts/activate
```

## Convert the model

Use the `olive run` command to convert and quantize the CLIP model

### Usage

```bash
olive run --config <config_file>
```

- `<config_file>`: Path to the JSON configuration file specifying the model, quantization settings, and conversion parameters.

### Examples

Convert the text and vision encoders of `laion/clip-vit-b-32-laion2b-s34b-b79k`

```bash
olive run --config laion_clip_text_b32_qdq.json
olive run --config laion_clip_vision_b32_qdq.json
```

```bash
# Convert with predefined config file
olive run --config {any_config.json}

# e.g.
olive run --config ./laion_clip_text_b32_qdq.json
olive run --config ./laion_clip_vision_b32_qdq.json
```

## Evaluate the model

Use the `olive_eval.py` script (located in `olive/examples/utils`) to evaluate the quantized model on a specified platform with a chosen evaluator.

### Usage

```bash
python ../../utils/olive_eval.py --config <config_file> --target <platform> --evaluator <evaluator_name>
```

- `<config_file>`: Path to the JSON configuration file used for conversion (e.g., `laion_clip_text_b32_qdq.json`).
- `<platform>`: Target platform for evaluation. Options: `cpu`, `qnn`, `vitis`, `ov`.
- `<evaluator_name>`: Name of the evaluator (e.g., `sanity_check`, `accuracy`).

### Examples

Evaluate the quantized text and vision encoders with QNNEP:

```bash
python ../utils/olive_eval.py --config laion_clip_text_b32_qdq.json --target qnn --evaluator sanity_check
python ../utils/olive_eval.py --config laion_clip_vision_b32_qdq.json --target qnn --evaluator sanity_check
```

## Evaluate Image Retrieval Accuracy with QNNEP

Use the `eval_retrieval_qnn.py` script to evaluate the image-text retrieval accuracy of the quantized CLIP model on a dataset (e.g., Flickr 1K).

### Usage

```bash
python eval_retrieval_qnn.py --model <huggingface_model> --text-encoder <text_model_path> --image-encoder <image_model_path> --dataset <dataset_name>
```

- `<huggingface_model>`: Hugging Face model identifier (e.g., `laion/clip-vit-base-patch32`).
- `<text_model_path>`: Path to the quantized text encoder ONNX model.
- `<image_model_path>`: Path to the quantized image encoder ONNX model.
- `<dataset_name>`: Hugging Face dataset for evaluation (e.g., `nlphuji/flickr_1k_test_image_text_retrieval`).

### Example

Evaluate retrieval accuracy of `laion/clip-vit-b-32-laion2b-s34b-b79k` on Flickr 1K:

```bash
python eval_retrieval_qnn.py --model laion/clip-vit-b-32-laion2b-s34b-b79k --text-encoder models/laion/clip_b32/text/model.onnx --image-encoder models/laion/clip_b32/image/model.onnx --dataset nlphuji/flickr_1k_test_image_text_retrieval
```

## Additional Resources

- [Olive Documentation](https://github.com/microsoft/Olive)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Hugging Face CLIP Models](https://huggingface.co/models?filter=clip)
- [Qualcomm Neural Network SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
