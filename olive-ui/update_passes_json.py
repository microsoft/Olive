#!/usr/bin/env python3
"""
Script to update passes.json with all passes from olive_config.json
and add UI-specific metadata like category and subcategory
"""
import json
from pathlib import Path

# Load olive_config.json
olive_config_path = Path(__file__).parent.parent / "olive" / "olive_config.json"
with open(olive_config_path, "r") as f:
    olive_config = json.load(f)

# Load existing passes.json if it exists
passes_json_path = Path(__file__).parent / "passes.json"
if passes_json_path.exists():
    with open(passes_json_path, "r") as f:
        passes_data = json.load(f)
else:
    passes_data = {"passes": {}}

# Function to determine category and subcategory
def get_category_info(pass_name, module_path):
    """Determine category and subcategory based on pass name and module path"""
    module_path_lower = module_path.lower()
    pass_name_lower = pass_name.lower()
    
    if "onnx" in module_path_lower:
        category = "ONNX"
        if "conversion" in pass_name_lower or "convert" in pass_name_lower:
            subcategory = "Conversion"
        elif "quant" in pass_name_lower:
            subcategory = "Quantization"
        elif "optim" in pass_name_lower or "transformer" in pass_name_lower:
            subcategory = "Optimization"
        elif "graph" in pass_name_lower or "split" in pass_name_lower or "compose" in pass_name_lower:
            subcategory = "Graph Operations"
        elif "builder" in pass_name_lower or "static_llm" in pass_name_lower:
            subcategory = "Model Building"
        elif "prepost" in pass_name_lower or "append" in pass_name_lower or "io" in pass_name_lower:
            subcategory = "Pre/Post Processing"
        else:
            subcategory = "Optimization"
    elif "pytorch" in module_path_lower:
        category = "PyTorch"
        if any(x in pass_name_lower for x in ["lora", "qlora", "dora", "loha", "lokr", "loftq"]):
            subcategory = "Fine-tuning"
        elif "quant" in pass_name_lower or "gptq" in pass_name_lower or "awq" in pass_name_lower:
            subcategory = "Quantization"
        elif any(x in pass_name_lower for x in ["sparse", "slice", "quarot", "spin"]):
            subcategory = "Model Optimization"
        else:
            subcategory = "Tensor Operations"
    elif "openvino" in module_path_lower:
        category = "OpenVINO"
        if "conversion" in pass_name_lower:
            subcategory = "Conversion"
        elif "quant" in pass_name_lower:
            subcategory = "Quantization"
        else:
            subcategory = "Other"
    else:
        category = "Other"
        subcategory = "Other"
    
    return category, subcategory

# Update passes data with all passes from olive_config
for pass_name, pass_info in olive_config["passes"].items():
    # Skip non-Pass entries
    if pass_name in ["PowerOfTwoMethod", "VitisQDQQuantizer", "VitisQOpQuantizer"]:
        continue
    
    # Get category info
    category, subcategory = get_category_info(pass_name, pass_info.get("module_path", ""))
    
    # Create or update pass entry
    if pass_name not in passes_data["passes"]:
        passes_data["passes"][pass_name] = {}
    
    # Update with all info from olive_config
    passes_data["passes"][pass_name].update(pass_info)
    
    # Add UI-specific fields
    passes_data["passes"][pass_name]["name"] = pass_name
    passes_data["passes"][pass_name]["category"] = category
    passes_data["passes"][pass_name]["subcategory"] = subcategory
    
    # Ensure all fields exist
    if "supported_accelerators" not in passes_data["passes"][pass_name]:
        passes_data["passes"][pass_name]["supported_accelerators"] = ["*"]
    if "supported_precisions" not in passes_data["passes"][pass_name]:
        passes_data["passes"][pass_name]["supported_precisions"] = ["*"]
    if "extra_dependencies" not in passes_data["passes"][pass_name]:
        passes_data["passes"][pass_name]["extra_dependencies"] = []
    
    # Add dataset requirement info
    if "dataset" in pass_info:
        passes_data["passes"][pass_name]["dataset_required"] = pass_info["dataset"]
    else:
        passes_data["passes"][pass_name]["dataset_required"] = "not_required"
    
    # Add empty schema if not present (will be populated dynamically)
    if "schema" not in passes_data["passes"][pass_name]:
        passes_data["passes"][pass_name]["schema"] = {}

# Sort passes by name for better readability
sorted_passes = {k: v for k, v in sorted(passes_data["passes"].items())}
passes_data["passes"] = sorted_passes

# Save updated passes.json
with open(passes_json_path, "w") as f:
    json.dump(passes_data, f, indent=2)

print(f"Updated passes.json with {len(passes_data['passes'])} passes")
print(f"Categories found: {set(p['category'] for p in passes_data['passes'].values())}")