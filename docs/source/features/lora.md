# LoRA Adapters

Olive provides a standard interface and a specialized interface to finetune a model and generate LoRA adapter. The standard interface, usually powered with an UX such as VS Code, provides maximum flexibility where the user can describe the fine-tuning parameters, model transformations parameters, datasets etc., in a .json or .yaml configuration which is provided to Olive as an input. For example,

```shell
$ olive run -- config llama2_qlora.config
```

Alternatively the user can use simplified command line to finetune HF model and generate LoRA adapters. For example,

```shell
$ olive finetune --method qlora --m meta-llama/Llama-2-7b-hf -d nampdn-ai/tiny-codes \
--train_split "train[:4096]" --eval_split "train[4096:4224]" --per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 --max_steps 150 --logging_steps 50 -o models/tiny-codes \
--text_template "### Language: {programming_language} \n### Question: {prompt} \n### Answer: {response}"
```

The output of the above commands will be LoRA adapter file as well as optimized ONNX Model.

In addition, Olive provides a separate command to produce LoRA adapter for ORT’s consumption from already finetuned PEFT model.

```shell
$ olive export-adapters [-h] [--adapter_path ADAPTER_PATH] \
                      [--output_path OUTPUT_PATH] [--dtype {float32,float16}] \
                      [--pack_weights] [--quantize_int4] \
                      [--int4_block_size {16,32,64,128,256}] \
                      [--int4_quantization_mode {symmetric,asymmetric}]
```

