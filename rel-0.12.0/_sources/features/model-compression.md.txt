# Model Compression

## SparseGPT
`SparseGPT` prunes GPT like models using a pruning method called [SparseGPT](https://arxiv.org/abs/2301.00774). This one-shot pruning method can perform unstructured
sparsity up to 60% on large models like OPT-175B and BLOOM-176B efficiently with negligible perplexity increase. It also supports semi-structured sparsity patterns such
as 2:4 and 4:8 patterns.

Please refer to the original paper linked above for more details on the algorithm and performance results for different models, sparsities and datasets.

This pass only supports HfModels. Please refer to [SparseGPT](sparsegpt) for more details on the types of transformers models supported.

**Note:** TensorRT can accelerate inference on 2:4 sparse models as described in [this blog](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/).

### Example Configuration
```json
{
    "type": "SparseGPT",
    "sparsity": 0.5
}
```
```json
{
    "type": "SparseGPT",
    "sparsity": [2,4]
}
```

## SliceGPT
`SliceGPT` is post-training sparsification scheme that makes transformer networks smaller by applying orthogonal transformations to each transformer layer that reduces the model size by slicing off the least-significant rows and columns of the weight matrices. This results in speedups and a reduced memory footprint.

Please refer to the original [paper](https://arxiv.org/abs/2401.15024) for more details on the algorithm and expected results for different models, sparsities and datasets.

This pass only supports HuggingFace transformer PyTorch models. Please refer to [SliceGPT](slicegpt) for more details on the types of transformers models supported.

### Example Configuration
```json
{
    "type": "SliceGPT",
    "sparsity": 0.4,
    "calibration_data_config": "wikitext2"
}
```
