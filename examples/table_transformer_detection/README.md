# Table Transformer Detection Optimization
This folder contains examples of [Table Transformer Detection](https://huggingface.co/microsoft/table-transformer-detection) optimization using different workflows.
- Qualcomm NPU: [with QNN execution provider in ONNX Runtime](#table-transformer-detection-optimization-with-qnn-execution-providers)

## Optimization Workflows
### Table Transformer Detection optimization with QNN execution providers
This example performs Table Transformer Detection optimization with QNN execution providers in one workflow. It performs the optimization pipeline:
- *Huggingface Model -> Onnx Model -> QNN Quantized Onnx Model*

Config file: [ttd_config.json](ttd_config.json)
## How to run
### Prerequisite
```
pip install timm
```
### Prepare dataset
- *TableBank* is a new image-based table detection and recognition dataset built with novel weak supervision from Word and Latex documents on the internet, contains 417K high-quality labeled tables. We extract a small datasets as calibrate and evaluate datasets.
- Download from https://huggingface.co/datasets/liminghao1630/TableBank/tree/main. Extract the split archive.
- Run
    ```
    python prepare_datasets.py
    ```
### Run Olive
```
olive run --config <config_file>.json
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
