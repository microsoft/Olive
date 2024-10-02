# Finetune

The `olive finetune` command will finetune a PyTorch/Hugging Face model and output a Hugging Face PEFT adapter. If you want to convert the PEFT adapter into a format for the ONNX Runtime, you can execute the `olive generate-adapter` command after finetuning.

## :material-clock-fast: Quickstart

The following example shows how to finetune [Llama-3.2-1B-Instruct from Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) either using your local computer (if you have a GPU device) or using remote compute via Azure AI integration with Olive.

=== "Local"

    !!! info
        You'll need a GPU device on your local machine to fine-tune a model. 

    ```bash
    olive finetune \ 
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \ 
        --trust_remote_code \ 
        --output_path finetuned-model \ 
        --data_name xxyyzzz/phrase_classification \ 
        --text_template "<|start_header_id|>user<|end_header_id|>\n{phrase}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{tone}" \ 
        --method qlora \ 
        --max_steps 30 \ 
        --log_level 1 \ 
    ```

=== "Azure AI"

    You can fine-tune on remote Azure ML compute by updating the placeholders (`{}`) in the following code snippet with your workspace, resource group and compute name details. Read the [How to create a compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?view=azureml-api-2&tabs=azure-studio) article for more details on setting up a GPU cluster in Azure ML.

    ```bash
    olive finetune \ 
        --model_name_or_path azureml://registries/azureml-meta/models/Llama-3.2-1B/versions/2 \  # (1)!
        --trust_remote_code \ 
        --output_path finetuned-model \ 
        --data_name xxyyzzz/phrase_classification \ 
        --text_template "<|start_header_id|>user<|end_header_id|>\n{phrase}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{tone}" \ 
        --method qlora \ 
        --max_steps 30 \ 
        --log_level 1 \ 
        --resource_group {RESOURCE_GROUP_NAME} \ 
        --workspace_name {WORKSPACE_NAME} \ 
        --aml_compute {COMPUTE_NAME}
    ```
    
    1. Note that the model path is pointing to an Azure ML registry.

    You can download the model artefact using the Azure ML CLI:

    ```bash
    az ml job download --name {JOB_ID} --resource-group {RESOURCE_GROUP_NAME} --workspace-name {WORKSPACE_NAME} -all
    ```

### :simple-onnx: Generate the adapters for ONNX Runtime

If you would like your fine-tuned model to run on the ONNX Runtime, you'll need to execute the `olive generate-adapter` command, using

```bash
olive generate-adapter \ 
    --model_name_or_path finetuned-model/model \ 
    --adapter_path finetuned-model/adapter \ 
    --use_ort_genai \ 
    --output_path adapter-onnx \ 
    --log_level 1
```