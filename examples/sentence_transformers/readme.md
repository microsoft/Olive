# Sentence-transformers/all-MiniLM-L6-v2 Optimization

This folder contains examples of [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) optimization.
- Qualcomm NPU: [with QNN execution provider in ONNX Runtime](#sentence-transformers-optimization-with-qnn-execution-providers)

## Optimization Workflows
### Sentence Transformers optimization with QNN execution providers
This workflow performs the optimization pipeline:
- *Huggingface Model -> Onnx Model -> QNN Quantized Onnx Model*

| Model | Pearson Correlation | Spearman Correlation | latency (avg) |
|-|-|-| -|
| Original model |  0.8274 | 0.8203 | 0.1457 |
| Quantized model | 0.8272 | 0.8198 | 0.0289s |

## How to run
### Pip requirements
```
pip install olive-ai
pip install onnxruntime==1.20.2
pip install datasets
```

### Run config
```
olive run --config <config_file>.json
```
After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.

### Evaluate with sentence-transformers/stsb datasets
*The Semantic Textual Similarity Benchmark* is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data. We are evaluating the output model's performance using Pearson correlation and Spearman correlation to measure the alignment between predicted similarity scores and human-labeled scores.
```
pip install scipy 
python eval_stsb.py
```