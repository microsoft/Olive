# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Workflow Python API implementations.

This module provides Python API functions corresponding to Olive CLI commands.
Each function returns a WorkflowOutput object containing ModelOutput instances.
"""
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from olive.cli.auto_opt import AutoOptCommand
from olive.cli.capture_onnx import CaptureOnnxGraphCommand  
from olive.cli.finetune import FineTuneCommand
from olive.cli.generate_adapter import GenerateAdapterCommand
from olive.cli.quantize import QuantizeCommand
from olive.cli.run import WorkflowRunCommand
from olive.cli.session_params_tuning import SessionParamsTuningCommand
from olive.constants import Precision
from olive.engine.output import WorkflowOutput
from olive.workflows import run as olive_run


def run(
    config: Union[str, Dict[str, Any]],
    *,
    input_model: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    log_level: Optional[int] = None,
    setup: bool = False,
    packages: bool = False,
    tempdir: Optional[str] = None,
    package_config: Optional[str] = None,
) -> WorkflowOutput:
    """
    Run an Olive workflow from a configuration.
    
    Args:
        config: Path to config file or config dictionary
        input_model: Input model configuration to override config file
        output_path: Output directory path
        log_level: Logging level (1-5)
        setup: Setup environment needed to run the workflow
        packages: List packages required to run the workflow  
        tempdir: Root directory for tempfile directories and files
        package_config: Path to optional package config file
        
    Returns:
        WorkflowOutput: Contains optimized models and metrics
    """
    from olive.common.config_utils import load_config_file
    
    # Load config from file or use provided dict
    if isinstance(config, str):
        run_config = load_config_file(config)
    else:
        run_config = config.copy()
    
    # Override config with provided parameters
    if input_model is not None:
        run_config["input_model"] = input_model
        
    if output_path is not None:
        # Remove from engine config if exists
        run_config.get("engine", {}).pop("output_dir", None)
        run_config["output_dir"] = output_path
        
    if log_level is not None:
        # Remove from engine config if exists  
        run_config.get("engine", {}).pop("log_severity_level", None)
        run_config["log_severity_level"] = log_level
    
    return olive_run(
        run_config,
        setup=setup,
        packages=packages,
        tempdir=tempdir,
        package_config=package_config,
    )


def auto_opt(
    model_path: str,
    *,
    output_path: str = "auto-opt-output",
    device: str = "cpu",
    provider: str = "CPUExecutionProvider",
    precision: Union[str, Precision] = Precision.FP32,
    # Dataset options
    data_name: Optional[str] = None,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    input_cols: Optional[List[str]] = None,
    batch_size: int = 1,
    # Model options
    task: Optional[str] = None,
    adapter_path: Optional[str] = None,
    use_dynamo_exporter: bool = False,
    use_model_builder: bool = False,
    use_qdq_encoding: bool = False,
    use_ort_genai: bool = False,
    # Advanced options
    enable_search: Optional[bool] = None,
    log_level: int = 3,
    **kwargs
) -> WorkflowOutput:
    """
    Automatically optimize a model for performance.
    
    Args:
        model_path: Path to model (file path or HuggingFace model name)
        output_path: Output directory path
        device: Target device ("cpu", "gpu", "npu")  
        provider: Execution provider
        precision: Output precision (fp32, fp16, int8, int4, nf4)
        data_name: Dataset name for evaluation
        split: Dataset split
        subset: Dataset subset
        input_cols: Input column names
        batch_size: Batch size for evaluation
        task: Model task (for HuggingFace models)
        adapter_path: Path to adapter weights
        use_dynamo_exporter: Use dynamo export API
        use_model_builder: Use model builder pass
        use_qdq_encoding: Use QDQ encoding for quantization
        use_ort_genai: Use ORT GenAI
        enable_search: Enable search optimization
        log_level: Logging level (1-5)
        **kwargs: Additional arguments
        
    Returns:
        WorkflowOutput: Contains optimized models and metrics
    """
    # Create args namespace similar to CLI parsing
    from argparse import Namespace
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    args = Namespace(
        model_path=model_path,
        output_path=str(Path(output_path).resolve()),
        device=device,
        provider=provider,
        precision=precision if isinstance(precision, Precision) else Precision(precision),
        data_name=data_name,
        split=split,
        subset=subset,
        input_cols=input_cols,
        batch_size=batch_size,
        task=task,
        adapter_path=adapter_path,
        use_dynamo_exporter=use_dynamo_exporter,
        use_model_builder=use_model_builder,
        use_qdq_encoding=use_qdq_encoding,
        use_ort_genai=use_ort_genai,
        enable_search=enable_search,
        log_level=log_level,
        # Set defaults for other required args
        memory=None,
        num_splits=None,
        cost_model=None,
        dynamic_to_fixed_shape_dim_param=None,
        dynamic_to_fixed_shape_dim_value=None,
        mixed_precision_overrides_config=None,
        save_config_file=False,
        **kwargs
    )
    
    # Create command instance and run
    command = AutoOptCommand(None, args)
    return command._run_workflow()


def finetune(
    model_path: str,
    *,
    output_path: str = "finetuned-adapter",
    method: str = "lora",
    lora_r: int = 64,
    lora_alpha: int = 16,
    target_modules: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    # Dataset options  
    data_name: Optional[str] = None,
    data_files: Optional[str] = None,
    text_template: Optional[str] = None,
    eval_split: Optional[str] = None,
    # Training args will be passed as **kwargs
    log_level: int = 3,
    **training_kwargs
) -> WorkflowOutput:
    """
    Fine-tune a model using LoRA or QLoRA.
    
    Args:
        model_path: Path to HuggingFace model
        output_path: Output directory path
        method: Fine-tuning method ("lora", "qlora")
        lora_r: LoRA rank value
        lora_alpha: LoRA alpha value
        target_modules: Target modules for LoRA (comma-separated)
        torch_dtype: PyTorch dtype for training
        data_name: Dataset name
        data_files: Path to data files
        text_template: Text template for formatting
        eval_split: Evaluation dataset split
        log_level: Logging level (1-5)
        **training_kwargs: HuggingFace training arguments
        
    Returns:
        WorkflowOutput: Contains fine-tuned adapter
    """
    from argparse import Namespace
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    args = Namespace(
        model_path=model_path,
        output_path=str(Path(output_path).resolve()),
        method=method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        torch_dtype=torch_dtype,
        data_name=data_name,
        data_files=data_files,
        text_template=text_template,
        eval_split=eval_split,
        log_level=log_level,
        save_config_file=False,
    )
    
    # Convert training kwargs to unknown_args format for CLI compatibility
    unknown_args = []
    for key, value in training_kwargs.items():
        unknown_args.extend([f"--{key}", str(value)])
    
    # Create command instance and run
    command = FineTuneCommand(None, args, unknown_args)
    return command._run_workflow()


def quantize(
    model_path: str,
    *,
    output_path: str = "quantized-model",
    algorithm: str = "rtn",
    precision: Union[str, Precision] = "int8",
    act_precision: Union[str, Precision] = "int8", 
    implementation: Optional[str] = None,
    use_qdq_encoding: bool = False,
    # Dataset options for static quantization
    data_name: Optional[str] = None,
    log_level: int = 3,
    **kwargs
) -> WorkflowOutput:
    """
    Quantize a PyTorch or ONNX model.
    
    Args:
        model_path: Path to model file
        output_path: Output directory path
        algorithm: Quantization algorithm (e.g., "rtn", "gptq", "awq")
        precision: Quantization precision (int8, int4, etc.)
        act_precision: Activation precision for static quantization
        implementation: Specific implementation of quantization algorithms
        use_qdq_encoding: Use QDQ encoding in ONNX model
        data_name: Dataset name (for static quantization)
        log_level: Logging level (1-5)
        **kwargs: Additional quantization parameters
        
    Returns:
        WorkflowOutput: Contains quantized model
    """
    from argparse import Namespace
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    args = Namespace(
        model_path=model_path,
        output_path=str(Path(output_path).resolve()),
        algorithm=algorithm,
        precision=precision,
        act_precision=act_precision,
        implementation=implementation,
        use_qdq_encoding=use_qdq_encoding,
        data_name=data_name,
        log_level=log_level,
        save_config_file=False,
        **kwargs
    )
    
    command = QuantizeCommand(None, args)
    return command._run_workflow()


def capture_onnx(
    model_path: str,
    *,
    output_path: str = "captured-model",
    log_level: int = 3,
    **kwargs
) -> WorkflowOutput:
    """
    Capture ONNX graph from a PyTorch model.
    
    Args:
        model_path: Path to PyTorch model
        output_path: Output directory path
        log_level: Logging level (1-5)
        **kwargs: Additional capture parameters
        
    Returns:
        WorkflowOutput: Contains captured ONNX model
    """
    from argparse import Namespace
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    args = Namespace(
        model_path=model_path,
        output_path=str(Path(output_path).resolve()),
        log_level=log_level,
        save_config_file=False,
        **kwargs
    )
    
    command = CaptureOnnxGraphCommand(None, args)
    return command._run_workflow()


def generate_adapter(
    model_path: str,
    *,
    output_path: str = "generated-adapter",
    adapter_format: str = "onnx_adapter",
    log_level: int = 3,
    **kwargs  
) -> WorkflowOutput:
    """
    Generate adapter for an ONNX model.
    
    Args:
        model_path: Path to ONNX model
        output_path: Output directory path
        adapter_format: Format to save weights in
        log_level: Logging level (1-5)
        **kwargs: Additional generation parameters
        
    Returns:
        WorkflowOutput: Contains generated adapter
    """
    from argparse import Namespace
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    args = Namespace(
        model_path=model_path,
        output_path=str(Path(output_path).resolve()),
        adapter_format=adapter_format,
        log_level=log_level,
        save_config_file=False,
        **kwargs
    )
    
    command = GenerateAdapterCommand(None, args)
    return command._run_workflow()


def session_params_tuning(
    model_path: str,
    *,
    output_path: str = "tuned-params",
    device: str = "cpu",
    provider: str = "CPUExecutionProvider",
    cpu_cores: Optional[int] = None,
    io_bind: bool = False,
    enable_cuda_graph: bool = False,
    log_level: int = 3,
    **kwargs
) -> WorkflowOutput:
    """
    Tune ONNX Runtime session parameters for optimal performance.
    
    Args:
        model_path: Path to ONNX model
        output_path: Output directory path
        device: Target device
        provider: Execution provider
        cpu_cores: CPU cores for thread tuning
        io_bind: Enable IOBinding search
        enable_cuda_graph: Enable CUDA graph
        log_level: Logging level (1-5)
        **kwargs: Additional tuning parameters
        
    Returns:
        WorkflowOutput: Contains tuning results
    """
    from argparse import Namespace
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    args = Namespace(
        model_path=model_path,
        output_path=str(Path(output_path).resolve()),
        device=device,
        provider=provider,
        cpu_cores=cpu_cores,
        io_bind=io_bind,
        enable_cuda_graph=enable_cuda_graph,
        log_level=log_level,
        save_config_file=False,
        **kwargs
    )
    
    command = SessionParamsTuningCommand(None, args)
    return command._run_workflow()