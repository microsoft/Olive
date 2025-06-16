import os
import sys
import json
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

# Add parent directory to path to import olive
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Olive UI Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests
)

# Add GZip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Load passes configuration from static JSON
PASSES_CONFIG_PATH = Path(__file__).parent.parent / "passes.json"
if PASSES_CONFIG_PATH.exists():
    with open(PASSES_CONFIG_PATH, "r") as f:
        passes_json = json.load(f)
        # Handle both formats - with or without root "passes" key
        if "passes" in passes_json:
            PASSES_DATA = passes_json["passes"]
        else:
            PASSES_DATA = passes_json
        logger.info(f"Loaded {len(PASSES_DATA)} passes from passes.json")
else:
    logger.error("passes.json not found! Please run update_passes_json.py to generate it.")
    PASSES_DATA = {}

class PassConfig(BaseModel):
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)
    disable_search: bool = False

class WorkflowConfig(BaseModel):
    input_model: Dict[str, Any]
    passes: Dict[str, PassConfig]
    engine: Optional[Dict[str, Any]] = None
    systems: Optional[Dict[str, Any]] = None
    output_dir: str = "olive_outputs"

class CLICommand(BaseModel):
    command: str
    args: Dict[str, Any]

class JobStatus(BaseModel):
    job_id: str
    status: str
    output: Optional[str] = None
    error: Optional[str] = None
    workflow_output: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

# Store active jobs
active_jobs = {}

@app.get("/")
async def root():
    return {"message": "Olive UI Server is running"}

@app.get("/api/passes")
async def get_passes():
    """Get all available passes organized by category from static JSON"""
    try:
        if not PASSES_DATA:
            logger.error("No passes data loaded")
            return JSONResponse(
                status_code=500,
                content={"error": "No passes data available. Please ensure passes.json exists."}
            )
        
        # Use static passes data for fast loading
        passes_by_category = {}
        
        # Organize passes by category and subcategory from static data
        for pass_name, pass_info in PASSES_DATA.items():
            category = pass_info.get("category", "Other")
            subcategory = pass_info.get("subcategory", "Other")
            
            # Initialize category if not exists
            if category not in passes_by_category:
                passes_by_category[category] = {}
            
            # Initialize subcategory if not exists
            if subcategory not in passes_by_category[category]:
                passes_by_category[category][subcategory] = []
            
            # Create pass data
            pass_data = {
                "name": pass_info["name"],
                "module_path": pass_info.get("module_path", ""),
                "supported_accelerators": pass_info.get("supported_accelerators", []),
                "supported_precisions": pass_info.get("supported_precisions", []),
                "extra_dependencies": pass_info.get("extra_dependencies", []),
                "dataset_required": pass_info.get("dataset_required", "not_required"),
                "category": category,
                "subcategory": subcategory
            }
            
            passes_by_category[category][subcategory].append(pass_data)
        
        return JSONResponse(content=passes_by_category)
    except Exception as e:
        logger.error(f"Error in get_passes: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch passes: {str(e)}"}
        )

@app.get("/api/pass/{pass_name}/schema")
async def get_pass_schema(pass_name: str):
    """Get the configuration schema for a specific pass from static JSON"""
    # First try to get from static data
    if PASSES_DATA and pass_name in PASSES_DATA:
        pass_info = PASSES_DATA[pass_name]
        schema = pass_info.get("schema", {})
        
        # If schema is empty, try to dynamically load it
        if not schema:
            schema = await get_dynamic_schema(pass_name, pass_info.get("module_path", ""))
            
        return {
            "pass_name": pass_name,
            "schema": schema,
            "pass_info": {
                "module_path": pass_info.get("module_path", ""),
                "supported_accelerators": pass_info.get("supported_accelerators", []),
                "supported_precisions": pass_info.get("supported_precisions", []),
                "extra_dependencies": pass_info.get("extra_dependencies", []),
                "category": pass_info.get("category", "Other"),
                "subcategory": pass_info.get("subcategory", "Other"),
                "dataset_required": pass_info.get("dataset_required", "not_required")
            }
        }
    
    raise HTTPException(status_code=404, detail=f"Pass '{pass_name}' not found")

async def get_dynamic_schema(pass_name: str, module_path: str) -> Dict[str, Any]:
    """Dynamically get schema for a pass"""
    schema = {}
    
    if not module_path:
        return schema
        
    try:
        module_parts = module_path.split(".")
        
        # Import the module
        module = __import__(module_path, fromlist=[module_parts[-1]])
        pass_class = getattr(module, module_parts[-1])
        
        # Get the default config using _default_config method
        if hasattr(pass_class, "_default_config"):
            # Create a dummy accelerator spec (we'll use CPU as default)
            from olive.hardware.accelerator import AcceleratorSpec
            accelerator_spec = AcceleratorSpec("cpu")
            
            default_config = pass_class._default_config(accelerator_spec)
            
            # Convert PassConfigParam objects to schema format
            for param_name, param_config in default_config.items():
                # Extract type information
                param_type = param_config.type_
                type_str = "string"  # default
                choices = None
                
                # Map Python types to JSON schema types
                type_name = str(param_type)
                
                if param_type in (int, type(int)) or "int" in type_name.lower():
                    type_str = "integer"
                elif param_type in (float, type(float)) or "float" in type_name.lower():
                    type_str = "float"
                elif param_type in (bool, type(bool)) or "bool" in type_name.lower():
                    type_str = "boolean"
                elif param_type in (list, type(list)) or (hasattr(param_type, "__origin__") and param_type.__origin__ == list) or "list" in type_name.lower():
                    type_str = "array"
                elif param_type in (dict, type(dict)) or (hasattr(param_type, "__origin__") and param_type.__origin__ == dict) or "dict" in type_name.lower() or "Dict" in type_name:
                    type_str = "object"
                elif param_type in (str, type(str)) or "str" in type_name.lower():
                    type_str = "string"
                elif hasattr(param_type, "__base__") and hasattr(param_type.__base__, "__name__") and "Enum" in param_type.__base__.__name__:
                    # Handle enums
                    type_str = "string"
                    choices = [str(item.value) for item in param_type]
                elif "DataConfig" in type_name:
                    type_str = "object"
                elif "Path" in type_name:
                    type_str = "string"
                
                # Handle Union types
                if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
                    # For Union types, use the first non-None type
                    for arg in param_type.__args__:
                        if arg != type(None):
                            if arg in (int, type(int)):
                                type_str = "integer"
                            elif arg in (float, type(float)):
                                type_str = "float"
                            elif arg in (bool, type(bool)):
                                type_str = "boolean"
                            elif arg in (list, type(list)):
                                type_str = "array"
                            elif arg in (dict, type(dict)):
                                type_str = "object"
                            break
                
                # Handle default values
                default_value = param_config.default_value
                
                # Handle ConditionalDefault
                if hasattr(default_value, "__class__") and "ConditionalDefault" in str(default_value.__class__):
                    # For conditional defaults, use None or try to extract a sensible default
                    default_value = None
                
                # Convert enum default values to strings
                if hasattr(default_value, "value"):
                    default_value = str(default_value.value)
                
                # Handle search defaults for choices
                if hasattr(param_config, "search_defaults") and param_config.search_defaults:
                    search_defaults = param_config.search_defaults
                    if hasattr(search_defaults, "choices"):
                        choices = search_defaults.choices
                        # Convert enum values to strings
                        if choices and hasattr(choices[0], "value"):
                            choices = [str(c.value) for c in choices]
                    elif hasattr(search_defaults, "__class__") and "Boolean" in str(search_defaults.__class__):
                        # Boolean search parameter
                        choices = [True, False]
                
                # Handle enum types to extract choices
                if choices is None and hasattr(param_type, "__members__"):
                    # It's an enum, extract all possible values
                    choices = [str(member.value) for member in param_type]
                
                schema[param_name] = {
                    "type": type_str,
                    "default": default_value,
                    "required": param_config.required,
                    "description": param_config.description or ""
                }
                
                if choices:
                    schema[param_name]["choices"] = choices
                    
        else:
            logger.warning(f"Pass {pass_name} does not have _default_config method")
            
    except Exception as e:
        logger.error(f"Could not introspect schema for pass {pass_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
    
    # If we couldn't get the schema dynamically, try alternate methods
    if not schema and module_path:
        try:
            # Try to get basic info from the pass class even without _default_config
            module_parts = module_path.split(".")
            module = __import__(module_path, fromlist=[module_parts[-1]])
            pass_class = getattr(module, module_parts[-1])
            
            # Check if it has a config_class attribute
            if hasattr(pass_class, "config_class"):
                config_class = pass_class.config_class
                schema = {}
                
                # Try to extract fields from the config class
                if hasattr(config_class, "__fields__"):
                    for field_name, field_info in config_class.__fields__.items():
                        schema[field_name] = {
                            "type": "string",  # default type
                            "default": getattr(field_info, "default", None),
                            "required": getattr(field_info, "required", False),
                            "description": ""
                        }
                        
        except Exception as e:
            logger.error(f"Alternative schema extraction failed for {pass_name}: {str(e)}")
    
    # If we still couldn't get the schema, use hardcoded schemas
    if not schema:
        # Hardcoded schemas for common passes
        hardcoded_schemas = {
            "LoRA": {
                "r": {"type": "integer", "default": 64, "description": "LoRA rank (dimension of the low-rank matrices)", "choices": [16, 32, 64, 128]},
                "alpha": {"type": "float", "default": 16, "description": "LoRA alpha scaling factor"},
                "lora_dropout": {"type": "float", "default": 0.05, "description": "LoRA dropout probability"},
                "target_modules": {"type": "array", "default": ["q_proj", "v_proj"], "description": "Target modules to apply LoRA"},
                "torch_dtype": {"type": "string", "default": "bfloat16", "choices": ["float32", "float16", "bfloat16"], "description": "Data type for training"},
                "device_map": {"type": "string", "default": "auto", "choices": ["auto", "cpu", "cuda"], "description": "Device map to use"},
                "train_data_config": {"type": "string", "required": True, "description": "Data config name for training"},
                "eval_data_config": {"type": "string", "description": "Data config name for evaluation"},
                "training_args": {"type": "object", "default": {"per_device_train_batch_size": 1, "max_steps": 150}, "description": "Training arguments"}
            },
            "QLoRA": {
                "r": {"type": "integer", "default": 64, "description": "LoRA rank", "choices": [16, 32, 64, 128]},
                "alpha": {"type": "float", "default": 16, "description": "LoRA alpha scaling factor"},
                "lora_dropout": {"type": "float", "default": 0.05, "description": "LoRA dropout probability"},
                "target_modules": {"type": "array", "default": ["q_proj", "v_proj"], "description": "Target modules"},
                "quant_type": {"type": "string", "default": "nf4", "choices": ["fp4", "nf4"], "description": "Quantization type"},
                "double_quant": {"type": "boolean", "default": True, "description": "Use double quantization"},
                "torch_dtype": {"type": "string", "default": "bfloat16", "choices": ["float32", "float16", "bfloat16"], "description": "Data type"}
            },
            "OnnxQuantization": {
                "per_channel": {"type": "boolean", "default": True, "description": "Use per-channel quantization"},
                "reduce_range": {"type": "boolean", "default": False, "description": "Reduce quantization range"},
                "quant_format": {"type": "string", "default": "QDQ", "choices": ["QDQ", "QOperator"], "description": "Quantization format"},
                "quant_mode": {"type": "string", "default": "static", "choices": ["static", "dynamic"], "description": "Quantization mode"},
                "activation_type": {"type": "string", "default": "int8", "choices": ["int8", "uint8"], "description": "Activation data type"},
                "weight_type": {"type": "string", "default": "int8", "choices": ["int8", "uint8"], "description": "Weight data type"}
            },
            "OnnxStaticQuantization": {
                "per_channel": {"type": "boolean", "default": True, "description": "Use per-channel quantization"},
                "reduce_range": {"type": "boolean", "default": False, "description": "Reduce quantization range"},
                "quant_format": {"type": "string", "default": "QDQ", "choices": ["QDQ", "QOperator"], "description": "Quantization format"},
                "calibrate_method": {"type": "string", "default": "MinMax", "choices": ["MinMax", "Entropy", "Percentile"], "description": "Calibration method"}
            },
            "OnnxDynamicQuantization": {
                "per_channel": {"type": "boolean", "default": True, "description": "Use per-channel quantization"},
                "reduce_range": {"type": "boolean", "default": False, "description": "Reduce quantization range"},
                "weight_type": {"type": "string", "default": "int8", "choices": ["int8", "uint8"], "description": "Weight data type"}
            },
            "OrtTransformersOptimization": {
                "model_type": {"type": "string", "required": True, "description": "Transformer model type (e.g., bert, gpt2)"},
                "num_heads": {"type": "integer", "default": 12, "description": "Number of attention heads"},
                "hidden_size": {"type": "integer", "default": 768, "description": "Hidden layer size"},
                "float16": {"type": "boolean", "default": False, "description": "Convert to FP16"},
                "use_gpu": {"type": "boolean", "default": False, "description": "Optimize for GPU execution"}
            }
        }
        
        if pass_name in hardcoded_schemas:
            schema = hardcoded_schemas[pass_name]
            logger.info(f"Using hardcoded schema for {pass_name}")
        else:
            # Generic schema as last resort
            schema = {
                "config": {
                    "type": "object",
                    "default": {},
                    "description": f"Configuration for {pass_name} pass. Please refer to the pass documentation for available parameters."
                }
            }
    
    return schema



@app.get("/api/cli-commands")
async def get_cli_commands():
    """Get all available CLI commands and their arguments"""
    commands = [
        {
            "name": "run",
            "description": "Run an olive workflow from a configuration file",
            "args": {
                "run-config": {"type": "file", "required": True, "description": "Path to configuration file"},
                "setup": {"type": "boolean", "default": False, "description": "Setup environment needed to run the workflow"},
                "packages": {"type": "boolean", "default": False, "description": "List packages required to run the workflow"},
                "tempdir": {"type": "string", "required": False, "description": "Root directory for tempfile directories and files"},
                "package-config": {"type": "file", "required": False, "description": "Path to optional package config file"},
                "log_level": {"type": "choice", "choices": ["0", "1", "2", "3", "4"], "default": "1", "description": "Logging level (0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL)"},
                "model_name_or_path": {"type": "string", "required": False, "description": "Model name or path (optional)"},
                "output_path": {"type": "string", "required": False, "description": "Output directory"}
            }
        },
        {
            "name": "run-pass",
            "description": "Run a single pass on an input model",
            "args": {
                "pass-name": {"type": "string", "required": True, "description": "Name of the pass to run"},
                "pass-config": {"type": "string", "required": False, "description": "JSON string with pass-specific configuration parameters"},
                "list-passes": {"type": "boolean", "default": False, "description": "List all available passes and exit"},
                "model_name_or_path": {"type": "string", "required": True, "description": "Model name or path"},
                "output_path": {"type": "string", "required": False, "description": "Output directory"},
                "log_level": {"type": "choice", "choices": ["0", "1", "2", "3", "4"], "default": "1", "description": "Logging level (0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL)"}
            }
        },
        {
            "name": "auto-opt",
            "description": "Automatically optimize model performance",
            "args": {
                "model_name_or_path": {"type": "string", "required": True, "description": "Model name or path"},
                "device": {"type": "choice", "choices": ["gpu", "cpu", "npu"], "default": "cpu", "description": "Target device"},
                "provider": {"type": "choice", "choices": ["CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"], "default": "CPUExecutionProvider", "description": "Execution provider"},
                "output_path": {"type": "string", "required": False, "description": "Output directory"},
                "log_level": {"type": "choice", "choices": ["0", "1", "2", "3", "4"], "default": "1", "description": "Logging level (0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL)"}
            }
        },
        {
            "name": "quantize",
            "description": "Quantize models",
            "args": {
                "model_name_or_path": {"type": "string", "required": True, "description": "Model name or path"},
                "algorithm": {"type": "choice", "choices": ["awq", "gptq", "rtn"], "required": True, "description": "Quantization algorithm"},
                "precision": {"type": "choice", "choices": ["int4", "int8"], "default": "int4", "description": "Quantization precision"},
                "output_path": {"type": "string", "required": False, "description": "Output directory"},
                "log_level": {"type": "choice", "choices": ["0", "1", "2", "3", "4"], "default": "1", "description": "Logging level (0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL)"}
            }
        }
    ]
    return commands

@app.post("/api/workflow/validate")
async def validate_workflow(config: WorkflowConfig):
    """Validate a workflow configuration"""
    try:
        # Basic validation
        errors = []
        warnings = []
        
        # Check if input model is specified
        if not config.input_model:
            errors.append("Input model must be specified")
        
        # Check if at least one pass is specified
        if not config.passes:
            errors.append("At least one pass must be specified")
        
        # Validate each pass
        for pass_name, pass_config in config.passes.items():
            if pass_config.type not in PASSES_DATA:
                errors.append(f"Unknown pass type: {pass_config.type}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/workflow/run")
async def run_workflow(config: WorkflowConfig):
    """Run a workflow configuration using Python API"""
    try:
        # Generate a unique job ID
        import uuid
        job_id = str(uuid.uuid4())
        logger.info(f"Starting workflow with job ID: {job_id}")
        logger.info(f"Workflow config: {config.dict()}")
        
        # Save configuration to a temporary file
        import tempfile
        temp_dir = tempfile.gettempdir()
        config_path = Path(temp_dir) / f"olive_config_{job_id}.json"
        
        # Convert model config if needed
        config_dict = config.dict()
        
        # Validate configuration
        if not config_dict.get("passes"):
            raise ValueError("No passes configured in the workflow")
        
        if not config_dict.get("input_model"):
            raise ValueError("No input model configured")
        
        # Log the configuration for debugging
        logger.info(f"Saving config to: {config_path}")
        logger.info(f"Config passes: {list(config_dict.get('passes', {}).keys())}")
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Store job information
        active_jobs[job_id] = {
            "status": "running", 
            "output": None, 
            "error": None,
            "created_at": datetime.now().isoformat(),
            "config": {**config_dict, "cli_command": "olive.workflows.run() - Python API (not CLI)"}
        }
        
        # Run the workflow in the background
        async def run_job():
            try:
                # Always use the current Python environment
                python_exe = sys.executable
                
                # Create Python code to run the workflow using the Python API
                run_code = f'''
import json
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path to import olive
sys.path.insert(0, r"{Path(__file__).parent.parent.parent}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from olive.workflows import run as olive_run
    
    # Load configuration
    with open(r"{config_path}") as f:
        olive_config = json.load(f)
    
    # Set output directory
    olive_config["output_dir"] = r"{config.output_dir}"
    
    # Add system configuration for local execution
    if "systems" not in olive_config:
        olive_config["systems"] = {{
            "local_system": {{
                "type": "LocalSystem",
                "config": {{}}
            }}
        }}
        
        # Set the default execution system
        olive_config["engine"] = {{
            "target": "local_system",
            "clean_run_cache": False
        }}
    
    # Log configuration
    logger.info("Starting Olive workflow with configuration:")
    logger.info(f"Input model: {{olive_config.get('input_model')}}")
    logger.info(f"Passes: {{list(olive_config.get('passes', {{}}).keys())}}")
    logger.info(f"Output directory: {{olive_config.get('output_dir')}}")
    
    # Run the workflow using Python API
    workflow_output = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    
    # Print results
    print("\\n=== WORKFLOW COMPLETED SUCCESSFULLY ===")
    print(f"Output model information:")
    if workflow_output:
        # Extract data from WorkflowOutput object based on the actual structure
        output_data = {{}}
        
        # Get available devices
        if hasattr(workflow_output, 'get_available_devices'):
            output_data['available_devices'] = workflow_output.get_available_devices()
        
        # Get the best candidate model
        best_model = None
        if hasattr(workflow_output, 'get_best_candidate'):
            best_model = workflow_output.get_best_candidate()
            if best_model:
                output_data['best_model'] = {{
                    'model_id': best_model.model_id,
                    'model_path': best_model.model_path,
                    'model_type': best_model.model_type,
                    'from_pass': best_model.from_pass() if hasattr(best_model, 'from_pass') else None,
                    'device': best_model.from_device() if hasattr(best_model, 'from_device') else None,
                    'execution_provider': best_model.from_execution_provider() if hasattr(best_model, 'from_execution_provider') else None,
                    'parent_model_id': best_model.get_parent_model_id() if hasattr(best_model, 'get_parent_model_id') else None
                }}
                
                # Get metrics
                if hasattr(best_model, 'metrics') and best_model.metrics:
                    output_data['best_model']['metrics'] = best_model.metrics
                if hasattr(best_model, 'metrics_value') and best_model.metrics_value:
                    output_data['best_model']['metrics_value'] = best_model.metrics_value
        
        # Get all output models
        if hasattr(workflow_output, 'get_output_models'):
            all_models = workflow_output.get_output_models()
            if all_models:
                output_data['output_models'] = []
                for model in all_models:
                    model_info = {{
                        'model_id': model.model_id,
                        'model_path': model.model_path,
                        'model_type': model.model_type,
                        'from_pass': model.from_pass() if hasattr(model, 'from_pass') else None,
                        'device': model.from_device() if hasattr(model, 'from_device') else None,
                        'execution_provider': model.from_execution_provider() if hasattr(model, 'from_execution_provider') else None
                    }}
                    
                    # Get metrics
                    if hasattr(model, 'metrics_value') and model.metrics_value:
                        model_info['metrics'] = model.metrics_value
                    
                    output_data['output_models'].append(model_info)
        
        # Get device-specific outputs
        available_devices = output_data.get('available_devices', [])
        for device_name in available_devices:
            device_output = workflow_output[device_name]
            if device_output and hasattr(device_output, 'get_output_models'):
                device_models = device_output.get_output_models()
                if device_models:
                    output_data[device_name + '_models_count'] = len(device_models)
        
        # Get the run history if available
        if best_model and hasattr(workflow_output, 'trace_back_run_history'):
            try:
                run_history = workflow_output.trace_back_run_history(best_model.model_id)
                if run_history:
                    output_data['run_history'] = run_history
            except:
                pass
        
        # Print the extracted data
        print(json.dumps(output_data, indent=2))
    else:
        print("Workflow completed but no output model information available")
    
except ImportError as e:
    print(f"ERROR: Failed to import Olive modules: {{str(e)}}", file=sys.stderr)
    print("Please ensure Olive is installed in the selected environment", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    import traceback
    print(f"ERROR: {{str(e)}}", file=sys.stderr)
    print("\\nFull traceback:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
'''
                
                # Set up environment
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
                
                # Run in subprocess with the current Python environment
                logger.info(f"Running workflow with Python: {python_exe}")
                process = await asyncio.create_subprocess_exec(
                    python_exe,
                    "-c",
                    run_code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                # Stream output
                stdout_data = []
                stderr_data = []
                
                # Read stdout and stderr concurrently
                async def read_stream(stream, data_list, stream_name):
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        decoded_line = line.decode('utf-8', errors='replace')
                        data_list.append(decoded_line)
                        # Update job output in real-time
                        active_jobs[job_id]["output"] = ''.join(stdout_data)
                        logger.info(f"{stream_name}: {decoded_line.strip()}")
                
                # Start reading both streams
                await asyncio.gather(
                    read_stream(process.stdout, stdout_data, "STDOUT"),
                    read_stream(process.stderr, stderr_data, "STDERR")
                )
                
                # Wait for process to complete
                await process.wait()
                
                if process.returncode == 0:
                    output_text = ''.join(stdout_data)
                    
                    # Try to parse workflow output from the output
                    workflow_output = None
                    try:
                        # Look for JSON output after the success message
                        if "WORKFLOW COMPLETED SUCCESSFULLY" in output_text:
                            json_start = output_text.find("{", output_text.find("WORKFLOW COMPLETED SUCCESSFULLY"))
                            if json_start != -1:
                                # Find the matching closing brace
                                brace_count = 0
                                json_end = json_start
                                for i in range(json_start, len(output_text)):
                                    if output_text[i] == '{':
                                        brace_count += 1
                                    elif output_text[i] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i + 1
                                            break
                                
                                if json_end > json_start:
                                    json_str = output_text[json_start:json_end]
                                    workflow_output = json.loads(json_str)
                    except Exception as e:
                        logger.error(f"Failed to parse workflow output: {e}")
                    
                    active_jobs[job_id].update({
                        "status": "completed",
                        "output": output_text,
                        "error": None,
                        "workflow_output": workflow_output,
                        "completed_at": datetime.now().isoformat()
                    })
                else:
                    active_jobs[job_id].update({
                        "status": "failed",
                        "output": ''.join(stdout_data),
                        "error": ''.join(stderr_data),
                        "completed_at": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Job execution error: {str(e)}")
                active_jobs[job_id].update({
                    "status": "failed",
                    "output": None,
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                })
            finally:
                # Clean up config file
                if config_path.exists():
                    config_path.unlink()
        
        # Start the job
        asyncio.create_task(run_job())
        
        return {"job_id": job_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return its path for use in CLI commands"""
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {"file_path": str(file_path.absolute()), "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/api/cli/run")
async def run_cli_command(command: CLICommand):
    """Run a CLI command"""
    try:
        # Generate a unique job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Build the command
        cmd = [sys.executable, "-m", "olive", command.command]
        
        # Add arguments
        for arg_name, arg_value in command.args.items():
            if arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        cmd.append(f"--{arg_name}")
                else:
                    cmd.extend([f"--{arg_name}", str(arg_value)])
        
        # Store the actual CLI command that will be executed
        cli_command_str = " ".join(cmd)
        
        # Initialize job
        active_jobs[job_id] = {
            "status": "running", 
            "output": None, 
            "error": None,
            "created_at": datetime.now().isoformat(),
            "config": {"command": command.command, "args": command.args, "cli_command": cli_command_str}
        }
        
        # Run the command in the background
        async def run_job():
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Initialize output buffers
                stdout_buffer = []
                stderr_buffer = []
                
                # Stream stdout
                async def read_stream(stream, buffer, is_stderr=False):
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        decoded_line = line.decode('utf-8', errors='replace')
                        buffer.append(decoded_line)
                        
                        # Update job output in real-time
                        if is_stderr:
                            active_jobs[job_id]["error"] = ''.join(stderr_buffer)
                        else:
                            active_jobs[job_id]["output"] = ''.join(stdout_buffer)
                
                # Read both streams concurrently
                await asyncio.gather(
                    read_stream(process.stdout, stdout_buffer, False),
                    read_stream(process.stderr, stderr_buffer, True)
                )
                
                # Wait for process to complete
                await process.wait()
                
                # Final update based on return code
                if process.returncode == 0:
                    output_text = ''.join(stdout_buffer)
                    
                    # Try to parse output model path from CLI output
                    workflow_output = None
                    import re
                    
                    # Look for various patterns that indicate output model location
                    patterns = [
                        r'Saved output model to\s+(.+?)(?:\n|$)',
                        r'Output model saved to:\s+(.+?)(?:\n|$)',
                        r'Model saved at:\s+(.+?)(?:\n|$)',
                        r'Output folder:\s+(.+?)(?:\n|$)'
                    ]
                    
                    model_path = None
                    for pattern in patterns:
                        match = re.search(pattern, output_text, re.IGNORECASE)
                        if match:
                            model_path = match.group(1).strip()
                            break
                    
                    # If we didn't find a specific pattern, check the last few lines for a path
                    if not model_path:
                        lines = output_text.strip().split('\n')
                        for line in reversed(lines[-5:]):  # Check last 5 lines
                            line = line.strip()
                            if line and (Path(line).exists() or '\\' in line or '/' in line):
                                if 'model' in line.lower() or 'output' in line.lower():
                                    model_path = line
                                    break
                    
                    if model_path:
                        logger.info(f"Found output model path: {model_path}")
                        
                        # Try to load model_config.json
                        try:
                            config_path = Path(model_path) / "model_config.json"
                            if config_path.exists():
                                with open(config_path, 'r') as f:
                                    model_config = json.load(f)
                                
                                # Extract model information
                                workflow_output = {
                                    "best_model": {
                                        "model_path": str(model_path),
                                        "model_type": model_config.get("type", "Unknown"),
                                        "device": model_config.get("model_attributes", {}).get("device", "cpu"),
                                        "execution_provider": model_config.get("model_attributes", {}).get("execution_provider", "CPUExecutionProvider"),
                                        "from_pass": model_config.get("pass_flows", [])[-1] if model_config.get("pass_flows") else "Unknown",
                                        "metrics_value": model_config.get("model_attributes", {}).get("metrics", {})
                                    },
                                    "output_models": [{
                                        "model_path": str(model_path),
                                        "model_type": model_config.get("type", "Unknown"),
                                        "device": model_config.get("model_attributes", {}).get("device", "cpu"),
                                        "execution_provider": model_config.get("model_attributes", {}).get("execution_provider", "CPUExecutionProvider"),
                                        "from_pass": model_config.get("pass_flows", [])[-1] if model_config.get("pass_flows") else "Unknown",
                                        "metrics": model_config.get("model_attributes", {}).get("metrics", {})
                                    }]
                                }
                                logger.info("Successfully loaded model configuration")
                        except Exception as e:
                            logger.error(f"Failed to load model config: {e}")
                    
                    active_jobs[job_id].update({
                        "status": "completed",
                        "output": output_text,
                        "error": None if not stderr_buffer else ''.join(stderr_buffer),
                        "workflow_output": workflow_output,
                        "completed_at": datetime.now().isoformat()
                    })
                else:
                    active_jobs[job_id].update({
                        "status": "failed",
                        "output": ''.join(stdout_buffer),
                        "error": ''.join(stderr_buffer),
                        "completed_at": datetime.now().isoformat()
                    })
            except Exception as e:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["error"] = str(e)
                active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Start the job
        asyncio.create_task(run_job())
        
        return {"job_id": job_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        output=job.get("output"),
        error=job.get("error"),
        workflow_output=job.get("workflow_output"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at"),
        config=job.get("config")
    )

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await websocket.accept()
    
    try:
        while True:
            if job_id in active_jobs:
                job = active_jobs[job_id]
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job["status"],
                    "output": job["output"],
                    "error": job["error"]
                })
                
                if job["status"] in ["completed", "failed"]:
                    break
            
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description='Olive UI Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        # Increase limits to handle large responses
        limit_concurrency=1000,
        limit_max_requests=10000,
        timeout_keep_alive=30,
        # Enable access logging for debugging
        access_log=True
    )