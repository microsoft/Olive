# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Usage examples for Olive Python API.

This file demonstrates how to use the Olive Python API to programmatically
execute workflows and obtain WorkflowOutput results containing ModelOutput instances.
"""

def example_auto_opt():
    """Example: Auto-optimize a HuggingFace model."""
    from olive import auto_opt
    
    # Auto-optimize a model for CPU with int8 quantization
    workflow_output = auto_opt(
        model_path="microsoft/phi-3-mini-4k-instruct",
        output_path="./optimized_phi3",
        device="cpu",
        provider="CPUExecutionProvider", 
        precision="int8",
        data_name="squad",  # For evaluation
        enable_search=True  # Enable optimization search
    )
    
    # Access the best optimized model
    if workflow_output.has_output_model():
        best_model = workflow_output.get_best_candidate()
        print(f"Best model saved at: {best_model.model_path}")
        print(f"Model metrics: {best_model.metrics_value}")
    
    return workflow_output


def example_finetune():
    """Example: Fine-tune a model with LoRA."""
    from olive import finetune
    
    # Fine-tune with LoRA on a dataset
    workflow_output = finetune(
        model_path="microsoft/phi-3-mini-4k-instruct",
        output_path="./finetuned_phi3",
        method="lora",
        lora_r=16,
        lora_alpha=32,
        data_name="squad",
        # Pass HuggingFace training arguments
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
    )
    
    # Access fine-tuned adapter
    if workflow_output.has_output_model():
        adapter_model = workflow_output.get_best_candidate()
        print(f"Fine-tuned adapter saved at: {adapter_model.model_path}")
    
    return workflow_output


def example_quantize():
    """Example: Quantize an ONNX model.""" 
    from olive import quantize
    
    # Quantize model with dynamic quantization
    workflow_output = quantize(
        model_path="./model.onnx",
        output_path="./quantized_model",
        algorithm="rtn",  # Round-to-nearest
        precision="int8",
        data_name="squad"  # For static quantization calibration
    )
    
    # Access quantized model
    if workflow_output.has_output_model():
        quantized_model = workflow_output.get_best_candidate()
        print(f"Quantized model: {quantized_model.model_path}")
    
    return workflow_output


def example_run_config():
    """Example: Run a workflow from configuration."""
    from olive import run
    
    # Define workflow configuration
    config = {
        "input_model": {
            "type": "HfModel",
            "model_path": "microsoft/phi-3-mini-4k-instruct"
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]
            }
        },
        "passes": {
            "onnx_conversion": {"type": "OnnxConversion"},
            "onnx_dynamic_quantization": {"type": "OnnxDynamicQuantization"}
        },
        "host": "local_system",
        "target": "local_system"
    }
    
    # Run the workflow
    workflow_output = run(
        config=config,
        output_path="./config_output"
    )
    
    # Access results
    if workflow_output.has_output_model():
        models = workflow_output.get_output_models()
        for model in models:
            print(f"Output model: {model.model_path} (pass: {model.from_pass()})")
    
    return workflow_output


def example_utility_functions():
    """Example: Use utility functions."""
    from olive import configure_qualcomm_sdk, convert_adapters, extract_adapters, shared_cache
    
    # Configure Qualcomm SDK (no return value)
    configure_qualcomm_sdk(py_version="3.8", sdk="qnn")
    
    # Convert LoRA adapters to ONNX format (no return value)
    convert_adapters(
        adapter_path="./fine_tuned_adapter",
        output_path="./converted_adapter.onnx",
        adapter_format="onnx_adapter"
    )
    
    # Extract LoRA adapters from PyTorch model (no return value)
    extract_adapters(
        model_path="./pytorch_model_with_lora",
        output_path="./extracted_adapters",
        format="onnx_adapter"
    )
    
    # Manage shared cache (no return value)
    shared_cache(
        account="mystorageaccount",
        container="mycache",
        delete=True,
        model_hash="abc123",
        yes=True
    )


if __name__ == "__main__":
    print("Olive Python API Examples")
    print("=" * 50)
    
    print("\nNOTE: These examples require dependencies to be installed and models to be available.")
    print("Run with proper setup to execute actual workflows.\n")
    
    # Show function signatures without executing
    import inspect
    from olive import auto_opt, finetune, quantize, run
    
    print("Function signatures:")
    print("-" * 20)
    
    for func in [auto_opt, finetune, quantize, run]:
        sig = inspect.signature(func)
        print(f"{func.__name__}{sig}")
    
    print("\nFor complete examples, see the function definitions in this file.")