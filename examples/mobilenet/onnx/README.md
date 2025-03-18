# TIMM Model Optimization (Quantization & QDQ)
This folder contains examples of **TIMM (PyTorch Image Models) optimization** using **Olive workflows**, focusing on **ONNX conversion, quantization, and QDQ transformation**.

## **Optimization Workflow**
This example optimizes `timm/mobilenetv3_small_100.lamb_in1k` for **CPU execution** by:
- *Converting PyTorch model to ONNX*
- *Applying ONNX quantization*
- *Applying QDQ (Quantize-DeQuantize) transformation*

- **Model**: [timm/mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **Dataset**: [ImageNet-1K](https://huggingface.co/datasets/imagenet-1k)

---

## **Running the Optimization**
### **Running with Config File**
The provided `config.json` configuration performs **ONNX conversion, quantization, and QDQ transformation**.

**Install Required Dependencies**
```sh
pip install -r requirements.txt
olive run --config config.json --setup
```
**Run Model Optimization**
```sh
olive run --config config.json
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.

