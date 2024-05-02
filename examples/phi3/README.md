# WIP: Phi-3 with multiple adapters
Run the following workflows first:

## Fine-tuning, Optimization, Extract Adapters
**CPU**:
```bash
olive run --config phi3_qlora_cpu.json
```

**CUDA**:
```bash
olive run --config phi3_qlora_cuda.json
```

## Fine-tuning, Export adapters
Then run the other fine-tuning workflow:
Download [dataset-classification.json](https://github.com/samuel100/phi-ft/blob/master/dataset/dataset-classification.json) to `models` folder.

```bash
olive run --config phi3_classification.json
```

## Inference with multiple adapters
Run the [notebook](generation.ipynb) to generate examples by commenting out which ever EP you don't want to use.

