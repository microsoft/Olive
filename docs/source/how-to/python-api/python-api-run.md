# Run Workflow Python API

The `run()` function provides programmatic access to execute Olive workflows from configuration files or dictionaries, giving you full control over complex optimization pipelines.

## {octicon}`zap` Quickstart

```python
from olive import run

# Run workflow from configuration file
workflow_output = run(
    config="./workflow_config.json",
    output_path="./workflow_results"
)

# Access workflow results
if workflow_output.has_output_model():
    best_model = workflow_output.get_best_candidate()
    print(f"Best model: {best_model.model_path}")
```

## Function Signature

```python
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
) -> WorkflowOutput
```

## Parameters

### Required Parameters
- **`config`** *(str | dict)*: Path to configuration file or configuration dictionary

### Optional Parameters
- **`input_model`** *(dict)*: Input model configuration to override config file
- **`output_path`** *(str)*: Output directory path (overrides config)
- **`log_level`** *(int)*: Logging level 1-5 (overrides config)
- **`setup`** *(bool)*: Setup environment needed to run the workflow (default: False)
- **`packages`** *(bool)*: List packages required to run the workflow (default: False)  
- **`tempdir`** *(str)*: Root directory for temporary files
- **`package_config`** *(str)*: Path to optional package configuration file

## Return Value

Returns a `WorkflowOutput` object containing all models produced by the workflow and their metrics.

## Usage Examples

### Running from Configuration File

```python
from olive import run

# Execute workflow from JSON file
result = run(
    config="./my_workflow.json",
    output_path="./results",
    log_level=2
)

# Process all workflow outputs
for i, model in enumerate(result.model_outputs):
    print(f"Model {i}: {model.model_path}")
    print(f"Metrics: {model.metrics_value}")
```

### Running from Configuration Dictionary

```python
from olive import run

# Define workflow configuration as dictionary
config = {
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/phi-3-mini-4k-instruct",
        "trust_remote_code": True
    },
    "passes": {
        "convert": {
            "type": "OnnxConversion"
        },
        "optimize": {
            "type": "OrtTransformersOptimization"
        },
        "quantize": {
            "type": "OnnxQuantization",
            "config": {
                "quant_mode": "rtn",
                "weight_type": "QUInt8"
            }
        }
    },
    "evaluators": {
        "common_evaluator": {
            "metrics": [
                {"name": "accuracy", "type": "accuracy"},
                {"name": "latency", "type": "latency"}
            ]
        }
    },
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "tpe"
    }
}

result = run(
    config=config,
    output_path="./workflow_results"
)
```

### Overriding Input Model

```python
from olive import run

# Override input model in existing config
input_model_override = {
    "type": "HfModel", 
    "model_path": "microsoft/phi-3-mini-4k-instruct",
    "trust_remote_code": True,
    "torch_dtype": "bfloat16"
}

result = run(
    config="./base_workflow.json",
    input_model=input_model_override,
    output_path="./custom_results"
)
```

### Setup and Package Management

```python
from olive import run

# Setup environment and check packages
result = run(
    config="./workflow.json",
    setup=True,  # Install required dependencies
    packages=True,  # List required packages
    package_config="./packages.json"
)
```

## Configuration Examples

### Multi-Pass Optimization Workflow

Create a JSON configuration file `workflow.json`:

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/phi-3-mini-4k-instruct"
    },
    "passes": {
        "conversion": {
            "type": "OnnxConversion",
            "config": {
                "target_opset": 17
            }  
        },
        "optimization": {
            "type": "OrtTransformersOptimization",
            "config": {
                "optimization_level": "all"
            }
        },
        "quantization": {
            "type": "OnnxQuantization", 
            "config": {
                "quant_mode": "rtn",
                "weight_type": "QUInt8"
            }
        }
    },
    "evaluators": {
        "accuracy_evaluator": {
            "metrics": [
                {
                    "name": "accuracy",
                    "type": "accuracy",
                    "data_config": {
                        "data_name": "squad",
                        "batch_size": 1
                    }
                }
            ]
        }
    },
    "engine": {
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "exhaustive"
        },
        "output_dir": "./workflow_output"
    }
}
```

```python
from olive import run

result = run(config="./workflow.json")
```

### Search Strategy Configuration

```python
from olive import run

# Configuration with advanced search
config = {
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/phi-3-mini-4k-instruct"
    },
    "passes": {
        "quantize": {
            "type": "OnnxQuantization",
            "config": {
                "quant_mode": ["rtn", "awq"],  # Search multiple modes
                "weight_type": ["QUInt8", "QInt8"]  # Search multiple types
            }
        }
    },
    "evaluators": {
        "performance": {
            "metrics": [
                {"name": "accuracy", "type": "accuracy"},
                {"name": "latency", "type": "latency"}
            ]
        }
    },
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "tpe",  # Tree-structured Parzen Estimator
        "num_samples": 20,
        "seed": 42
    }
}

result = run(config=config)
```

## Working with Results

```python
result = run(config="./workflow.json")

# Check if workflow produced models
if result.has_output_model():
    # Get the best model based on evaluation metrics
    best_model = result.get_best_candidate()
    print(f"Best model: {best_model.model_path}")
    print(f"Best metrics: {best_model.metrics_value}")
    
    # Iterate through all workflow outputs
    print(f"\nAll {len(result.model_outputs)} models:")
    for i, model in enumerate(result.model_outputs):
        print(f"Model {i}:")
        print(f"  Path: {model.model_path}")
        print(f"  Config: {model.config}")
        print(f"  Metrics: {model.metrics_value}")
        
        # Access specific metrics
        if 'accuracy' in model.metrics_value:
            accuracy = model.metrics_value['accuracy']
            print(f"  Accuracy: {accuracy}")
        
        if 'latency' in model.metrics_value:
            latency = model.metrics_value['latency']
            print(f"  Latency: {latency} ms")
```

## Advanced Configuration Features

### Data Configuration

```json
{
    "data_configs": {
        "squad_data": {
            "data_name": "squad",
            "batch_size": 1,
            "max_samples": 100
        },
        "custom_data": {
            "data_files": "./my_data.jsonl",
            "input_cols": ["input_text"],
            "max_samples": 500
        }
    },
    "passes": {
        "quantize": {
            "type": "OnnxQuantization",
            "config": {
                "data_config": "squad_data"
            }
        }
    }
}
```

### System Configuration

```json
{
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "config": {
                "accelerators": ["cpu"]
            }
        },
        "gpu_system": {
            "type": "LocalSystem", 
            "config": {
                "accelerators": ["gpu"]
            }
        }
    },
    "passes": {
        "optimization": {
            "type": "OrtTransformersOptimization",
            "host": "local_system",
            "evaluator": "gpu_system"
        }
    }
}
```

### Engine Configuration

```json
{
    "engine": {
        "output_dir": "./results",
        "log_severity_level": 1,
        "packaging_config": {
            "type": "Zipfile",
            "name": "optimized_model"
        },
        "cache_dir": "./cache",
        "clean_cache": false
    }
}
```

## Integration with Other APIs

You can combine `run()` with other Olive APIs:

```python
from olive import run, auto_opt

# First use auto_opt for quick optimization
quick_result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./quick_opt",
    precision="int8"
)

# Then use run() for advanced workflow on the optimized model
if quick_result.has_output_model():
    optimized_model = quick_result.get_best_candidate()
    
    advanced_config = {
        "input_model": {
            "type": "ONNXModel",
            "model_path": optimized_model.model_path
        },
        "passes": {
            "further_optimize": {
                "type": "OrtTransformersOptimization"
            }
        }
    }
    
    final_result = run(config=advanced_config)
```

## Equivalent CLI Command

```bash
# CLI equivalent
olive run \
    --config ./workflow.json \
    --output_path ./results \
    --log_level 2
```

```python
# Python API equivalent
from olive import run

result = run(
    config="./workflow.json",
    output_path="./results",
    log_level=2
)
```