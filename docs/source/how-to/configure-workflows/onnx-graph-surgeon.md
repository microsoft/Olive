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

Initial ONNX model graph:

```
graph {
  input: "input1"
  input: "input2"
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
}
```

After applying:

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

Transformed ONNX model graph:

```
graph {
  input: "renamed_input1"
  input: "renamed_input2"
  node {
    op_type: "Add"
    input: ["renamed_input1", "renamed_input2"]
    output: ["add_output"]
  }
}
```

### `RenameOutputs`

#### Description

Renames specific outputs in the ONNX model.

#### Configurations

- `old_names`: List of original output names to rename.
- `new_names`: List of new output names.

#### Example

Initial ONNX model graph:

```
graph {
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["output1"]
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RenameOutputs",
            "old_names": ["output1"],
            "new_names": ["renamed_output1"]
        }
    ]
}
```

Transformed ONNX model graph:

```
graph {
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["renamed_output1"]
  }
}
```

### `InferShapes`

#### Description

Performs shape inference on the ONNX model to populate missing shape information.

#### Example

Initial ONNX model graph:

```
graph {
  input: "input1" (FLOAT) shape: [1]
  input: "input2" (FLOAT) shape: [1]
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
  output: "add_output" (FLOAT)
}
```

After applying:

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

Transformed ONNX model graph (with inferred shapes):
```
graph {
  input: "input1" (FLOAT) shape: [1]
  input: "input2" (FLOAT) shape: [1]
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
  output: "add_output" (FLOAT) shape: [1]
}
```


### `RemoveShapes`

#### Description

Removes all shape information (`value_info`) from the ONNX model.

#### Example

Initial ONNX model graph:

```
graph test-model {
  # Inputs
  input: "input1" (FLOAT) shape: [1]
  input: "input2" (FLOAT) shape: [1]

  # Value Info
  value_info: "intermediate" (FLOAT) shape: [1]

  # Nodes
  node {
    op_type: "Add"
    name: "Add"
    input: ["input1", "input2"]
    output: ["intermediate"]
  }

  node {
    op_type: "Relu"
    name: "Relu"
    input: ["intermediate"]
    output: ["output1"]
  }

  # Outputs
  output: "output1" (FLOAT) shape: [1]
}
```

After applying:

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

Transformed ONNX model graph:

```
graph test-model {
  # Inputs
  input: "input1" (FLOAT) shape: [1]
  input: "input2" (FLOAT) shape: [1]

  # Nodes
  node {
    op_type: "Add"
    name: "Add"
    input: ["input1", "input2"]
    output: ["intermediate"]
  }

  node {
    op_type: "Relu"
    name: "Relu"
    input: ["intermediate"]
    output: ["output1"]
  }

  # Outputs
  output: "output1" (FLOAT) shape: [1]
}
```


### `RemoveInitializerFromInputs`

#### Description

Removes initializers from the input list (`graph.input`) of an ONNX model.

#### Example

Initial ONNX model graph:

```
graph {
  input: "input1"
  input: "input2"
  initializer {
    name: "input2"
    data_type: FLOAT
    dims: [1, 3]
    raw_data: "\000\000\200?\000\000\000@\000\000@@"
  }
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveInitializerFromInputs"
        }
    ]
}
```

Transformed ONNX model graph:

```
graph {
  input: "input1"
  initializer {
    name: "input2"
    data_type: FLOAT
    dims: [1, 3]
    raw_data: "\000\000\200?\000\000\000@\000\000@@"
  }
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
}
```

### `ReorderInputs`

#### Description

Reorders the inputs of the ONNX model according to a specified permutation.

#### Configurations

- `permutation`: A list specifying the new order of inputs, as a permutation of the original indices.

#### Example

Initial ONNX model graph:

```
graph {
  input: "input1"
  input: "input2"
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
}
```

After applying:

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

Transformed ONNX model graph:

```
graph {
  input: "input2"
  input: "input1"
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
}
```

### `ReplaceErfWithTanh`

#### Description

Replaces `Erf` nodes in the ONNX model with an equivalent computation using `Tanh`. The replacement involves scaling the input and applying the `Tanh` function to produce a result that approximates the `Erf` behavior.

#### Example

Initial ONNX model graph:

```
graph {
  input: "input"
  output: "erf_output"
  node {
    op_type: "Erf"
    input: ["input"]
    output: ["erf_output"]
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ReplaceErfWithTanh"
        }
    ]
}
```

Transformed ONNX model graph:

```
graph {
  input: "input"
  initializer: "scale_0" (FLOAT, value: 1.203)
  node {
    op_type: "Mul"
    input: ["input", "scale_0"]
    output: ["mul_0"]
    name: "Sub_Mul_0"
  }
  node {
    op_type: "Tanh"
    input: ["mul_0"]
    output: ["erf_output"]
    name: "Sub_Tanh_0"
  }
  output: "erf_output"
}
```

### `ZeroOutInput`

#### Description

Replaces a specific input of a node with a constant tensor of zeros.

#### Configurations

- `node_name`: Name of the node whose input is to be replaced.
- `input_idx`: Index of the input to be replaced.

#### Example

Initial ONNX model graph:

```
graph {
  input: "input1"
  input: "input2"
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
}
```

After applying:

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

Transformed ONNX model graph:

```
graph {
  input: "input1"
  node {
    op_type: "Constant"
    output: ["Add_zero_output_0"]
    value: [0.0]
  }
  node {
    op_type: "Add"
    input: ["input1", "Add_zero_output_0"]
    output: ["add_output"]
  }
}
```


### `RemoveInputs`

#### Description

Removes specific inputs from the ONNX model.

#### Configurations

- `names`: List of input names to remove.

#### Example

Initial ONNX model graph:

```
graph {
  input: "input1"
  input: "input2"
  node {
    op_type: "Add"
    input: ["input1", "input2"]
    output: ["add_output"]
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveInputs",
            "names": ["input2"]
        }
    ]
}
```

Transformed ONNX model graph:

```
graph {
  input: "input1"
  node {
    op_type: "Add"
    input: ["input1"]
    output: ["add_output"]
  }
}
```

### `ExposeOutputs`

#### Description

Exposes the outputs of specific nodes in the ONNX model as graph outputs.

#### Configurations

- `names`: List of node names whose outputs should be exposed as graph outputs.

#### Example

Initial ONNX model graph:

```
graph {
  node {
    op_type: "Relu"
    input: ["input1"]
    output: ["relu_output"]
  }
}
```

After applying:

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

Transformed ONNX model graph:

```
graph {
  node {
    op_type: "Relu"
    input: ["input1"]
    output: ["relu_output"]
  }
  output: "relu_output"
}
```


### `ExposeQuantizedOutput`

#### Description

Replaces a specified output with its quantized version and exposes its `scale` and `zero_point` as graph outputs.

#### Configurations

- `output_name`: Name of the output to replace with its quantized version.

#### Example

Initial ONNX model graph:

```
graph {
  node {
    op_type: "DequantizeLinear"
    input: ["quantized_tensor", "scale", "zero_point"]
    output: ["output1"]
  }
}
```

After applying:

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


Transformed ONNX model graph:

```
graph {
  output: "quantized_tensor"
  output: "scale"
  output: "zero_point"
}
```


### RMSNormToL2Norm

#### Description

Replace RMSNorm subgraph with L2Norm subgraph.

#### Example

Initial model graph:

```
RMSNorm pattern:
    +-----------------------------------------------+
    |                                               |
    |                                               v
[Root] --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul
          (y=2)     (axis=-1)   (B=E-6)
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RMSNormToL2Norm"
        }
    ]
}
```


Transformed model graph:

```
[Root] --> LpNormalization --> Mul
           (p=2, axis=-1)
```

### ReplaceAttentionMaskValue

#### Description

Replace the value of extended attention mask with a new value. This surgery is useful if the default mask value does not quantize well due to numerical instability.

#### Example

Initial model graph:

```
graph {
  node {
    input: "input1"
    output: "output1"
    name: "ConstantOfShape"
    op_type: "ConstantOfShape"
    attribute {
      name: "value"
      t {
        dims: 1
        data_type: 1
        float_data: -3.4028234663852886e+38
        name: ""
      }
      type: TENSOR
    }
  }
  node {
    output: "Constant_output"
    name: "Constant"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        data_type: 1
        float_data: -3.4028234663852886e+38
        name: ""
      }
      type: TENSOR
    }
  }
  initializer {
    data_type: 1
    float_data: -3.4028234663852886e+38
    name: "init"
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ReplaceAttentionMaskValue"
        }
    ]
}
```


Transformed model graph:

```
graph {
  node {
    input: "input1"
    output: "output1"
    name: "ConstantOfShape"
    op_type: "ConstantOfShape"
    attribute {
      name: "value"
      t {
        dims: 1
        data_type: 1
        float_data: -10000.0
        name: ""
      }
      type: TENSOR
    }
  }
  node {
    output: "Constant_output"
    name: "Constant"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        data_type: 1
        float_data: -10000.0
        name: ""
      }
      type: TENSOR
    }
  }
  initializer {
    data_type: 1
    float_data: -10000.0
    name: "init"
  }
}
```
