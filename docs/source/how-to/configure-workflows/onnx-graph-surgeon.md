# ONNX Surgeon Classes Documentation

This document provides an overview and detailed explanation of the available `GraphSurgeries` Pass for manipulating ONNX models.

## Important Note

In the examples, the `surgeries` Pass represents a list of surgeries to be applied sequentially on the ONNX model. You can combine multiple surgeries in the list to perform consecutive modifications in a single operation.

### Example
```json
"surgeries": {
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RenameInputs",
            "old_names": ["input1", "input2"],
            "new_names": ["renamed_input1", "renamed_input2"]
        },
        {
            "surgeon": "RenameOutputs",
            "old_names": ["output1", "output2"],
            "new_names": ["renamed_output1", "renamed_output2"]
        },
        {
            "surgeon": "InferShapes"
        }
    ]
}
```

## Surgeries

### `RenameInputs`

#### Description
Renames specific inputs in the ONNX model.

#### Configurations
- `old_names`: List of original input names to rename.
- `new_names`: List of new input names.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RenameInputs",
            "old_names": ["input1", "input2"],
            "new_names": ["renamed_input1", "renamed_input2"]
        }
    ]
}
```

### `RenameOutputs`

#### Description
Renames specific outputs in the ONNX model.

#### Configurations
- `old_names`: List of original output names to rename.
- `new_names`: List of new output names.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RenameOutputs",
            "old_names": ["output1", "output2"],
            "new_names": ["renamed_output1", "renamed_output2"]
        }
    ]
}
```

### `InferShapes`

#### Description
Performs shape inference on the ONNX model to populate missing shape information.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "InferShapes"
        }
    ]
}
```

### `RemoveShapes`

#### Description
Removes all shape information (`value_info`) from the ONNX model.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveShapes"
        }
    ]
}
```

### `ReorderInputs`

#### Description
Reorders the inputs of the ONNX model according to a specified permutation.

#### Configurations
- `permutation`: A list specifying the new order of inputs, as a permutation of the original indices.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ReorderInputs",
            "permutation": [1, 0]
        }
    ]
}
```

### `ZeroOutInput`

#### Description
Replaces a specific input of a node with a constant tensor of zeros.

#### Configurations
- `node_name`: Name of the node whose input is to be replaced.
- `input_idx`: Index of the input to be replaced.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ZeroOutInput",
            "node_name": "AddNode",
            "input_idx": 1
        }
    ]
}
```

### `RemoveInputs`

#### Description
Removes specific inputs from the ONNX model.

#### Configurations
- `names`: List of input names to remove.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveInputs",
            "names": ["input1"]
        }
    ]
}
```

### `ExposeOutputs`

#### Description
Exposes the outputs of specific nodes in the ONNX model as graph outputs.

#### Configurations
- `names`: List of node names whose outputs should be exposed as graph outputs.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ExposeOutputs",
            "names": ["ReluNode"]
        }
    ]
}
```

### `ExposeQuantizedOutput`

#### Description
Replaces a specified output with its quantized version and exposes its `scale` and `zero_point` as graph outputs.

#### Configurations
- `output_name`: Name of the output to replace with its quantized version.

#### Example
```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ExposeQuantizedOutput",
            "output_name": "output1"
        }
    ]
}
```
