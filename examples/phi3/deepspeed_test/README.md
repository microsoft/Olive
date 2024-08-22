# Install Olive and Requirements
```bash
pip install git+https://github.com/microsoft/Olive
```

Install the requirements:
```bash
pip install -r requirements.txt
```

# Run the example
```bash
# If multiple GPUs are available, the pass currently just does naive model parallelism
# Is that worth testing? Does deepspeed support model parallel models?
CUDA_VISIBLE_DEVICES=0 olive run --config finetune.json
```

Things we have to test:
- Fine-tuning method:
    - qlora
    - lora
- Data type?:
    - float32
    - float16
    - bfloat16
- What is a good default configuration for deepspeed?
