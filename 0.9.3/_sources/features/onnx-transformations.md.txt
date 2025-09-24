# ONNX Transformations

[ONNX](https://onnx.ai/) is an open graph format to represent machine learning models. [ONNX Runtime](https://onnxruntime.ai/docs/) is a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries.

Olive provides multiple transformations and optimizations based on various ONNX to improve model performance.

## Peeophole Optimizer

`OnnxPeepholeOptimizer` optimizes an ONNX model. The optimization process involves analyzing the structure of the ONNX model and identifying opportunities.

The `OnnxPeepholeOptimizer` leverages `onnxscript` (https://onnxscript.ai/tutorial/optimizer/optimize.html) and `onnxoptimizer`(https://github.com/onnx/optimizer) underneath.

| Optimization                      | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| **Constant Folding**              | Applies constant folding optimization to the model.                        |
| **Constant Propagation**          | Applies constant propagation optimization to the model. Applied as part of constant folding. |
| **Sequence Simplification**       | Simplifies Sequence-based ops (e.g., SequenceConstruct, ConcatFromSequence). Part of constant folding. |
| **Remove Unused Nodes**           | Removes unused nodes from the model.                                       |
| **Remove Unused Functions**       | Removes unused function protos from the model.                             |
| **Inline Functions with Unused Outputs** | Inlines function nodes with unused outputs.                                |
| **Inline Simple Functions**       | Inlines simple functions based on a node count threshold.                  |
| **Eliminate Nop Cast**            | Eliminates no-operation (nop) Casts.                                                |
| **Eliminate Nop Dropout**         | Eliminates no-operation Dropouts.                                                   |
| **Eliminate Nop Flatten**         | Eliminates no-operation Flattens.                                                   |
| **Extract Constant to Initializer** | Extracts constants to initializers.                                                 |
| **Eliminate If with Const Cond**  | Eliminates If nodes with constant conditions.                                       |
| **Eliminate Nop Monotone ArgMax** | Eliminates nop monotone ArgMax.                                                     |
| **Eliminate Nop Pad**             | Eliminates no-operation Pads.                                                       |
| **Eliminate Nop Concat**          | Eliminates no-operation Concats.                                                    |
| **Eliminate Nop Split**           | Eliminates no-operation Splits.                                                     |
| **Eliminate Nop Expand**          | Eliminates no-operation Expands.                                                    |
| **Eliminate Shape Gather**        | Eliminates Shape Gather operations.                                                 |
| **Eliminate Slice after Shape**   | Eliminates Slice nodes that occur after Shape nodes.                                |
| **Eliminate Nop Transpose**       | Eliminates no-operation Transposes.                                                |
| **Fuse Add Bias into Conv**       | Fuses Add operations as biases into Conv layers.                                    |
| **Fuse BN into Conv**             | Fuses BatchNormalization into Conv layers.                                          |
| **Fuse Consecutive Concats**      | Fuses consecutive Concat operations.                                                |
| **Fuse Consecutive LogSoftmax**   | Fuses consecutive LogSoftmax operations.                                            |
| **Fuse Consecutive Reduce+Unsqueeze** | Fuses consecutive Reduce and Unsqueeze operations.                                 |
| **Fuse Consecutive Squeezes**     | Fuses consecutive Squeeze operations.                                               |
| **Fuse Consecutive Transposes**   | Fuses consecutive Transpose operations.                                             |
| **Fuse MatMul+Add Bias into GEMM** | Fuses MatMul and Add operations into GEMM layers.                                   |
| **Fuse Pad into Conv**            | Fuses Pad operations into Conv layers.                                              |
| **Fuse Pad into Pool**            | Fuses Pad operations into Pool layers.                                              |
| **Fuse Transpose into GEMM**      | Fuses Transpose operations into GEMM layers.                                        |
| **Fuse Concat into Reshape**      | Fuses Concat operations into Reshape layers.                                        |
| **Eliminate Nop Reshape**         | Eliminates no-operation Reshapes.                                                   |
| **Eliminate Nop with Unit**       | Eliminates no-operation nodes with unit values.                                     |
| **Eliminate Common Subexpression** | Eliminates common sub-expressions.                                                 |
| **Fuse QKV**                      | Fuses query, key, and value layers in transformer models.                          |
| **Fuse Consecutive Unsqueezes**   | Fuses consecutive Unsqueeze operations.                                             |
| **Eliminate Deadend Nodes**       | Eliminates dead-end nodes.                                                          |
| **Eliminate Identity Nodes**      | Eliminates Identity nodes.                                                          |
| **Eliminate Shape Ops**           | Eliminates Shape operations where possible.                                         |
| **Fuse Consecutive Slices**       | Fuses consecutive Slice operations.                                                 |
| **Eliminate Unused Initializer**  | Eliminates unused initializers.                                                     |
| **Eliminate Duplicate Initializer** | Eliminates duplicate initializers.                                  |
| **Broadcast to MatMul**           | Converts broadcast patterns into MatMul operations for better efficiency.  |
| **Cast Constant of Shape**        | Simplifies constant casting for shape operations.                          |
| **GEMM to MatMul+Add**            | Converts GEMM operations into MatMul and Add for improved compatibility.   |
| **No-Op Removal**                 | Removes redundant or no-op operations in the computation graph.            |

Please refer to [OnnxPeepholeOptimizer](../reference/pass.rst#onnx_peephole_optimizer) for more details about the pass and its config parameters.

### Example Configuration

```json
{
    "type": "OnnxPeepholeOptimizer"
}
```

## ORT Transformers Optimization

While ONNX Runtime automatically applies most optimizations while loading transformer models, some of the latest optimizations that have not
yet been integrated into ONNX Runtime.
`OrtTransformersOptimization` provides an offline capability to optimize [transformers](https://huggingface.co/docs/transformers/index) models
in scenarios where ONNX Runtime does not apply the optimization at load time.
These optimizations are provided by onnxruntime through
[onnxruntime.transformers](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers). Please
refer to the [corresponding documentation](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md)
for more details on the optimizations done by this tool.

Please refer to [OrtTransformersOptimization](../reference/pass.rst#ort_transformers_optimization) for more details about the pass and its config parameters.

### Example Configuration

```json
{
    "type": "OrtTransformersOptimization",
    "model_type": "bert"
}
```

## ONNX Surgeries

Olive provides ability to apply many graph `surgeries` on the ONNX model. In the examples, the `surgeries` Pass represents a list of surgeries to be applied sequentially on the ONNX model. You can combine multiple surgeries in the list to perform consecutive modifications in a single operation.

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

### `DecomposeQuickGelu`

#### Description

Replaces QuickGelu operators with equivalent standard ONNX operators. This surgery converts a single QuickGelu node into a subgraph composed of Mul, Sigmoid, and Mul operations.

The QuickGelu function is mathematically defined as: `QuickGelu(x) = x * sigmoid(alpha * x)`, where alpha defaults to 1.702.

Reference: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/contrib_ops/contrib_defs.cc

#### Example

Initial model graph:

```proto
graph {
  input: "input"
  node {
    op_type: "QuickGelu"
    input: ["input"]
    output: ["quickgelu_output"]
    name: "QuickGeluNode"
  }
  output: "quickgelu_output"
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "DecomposeQuickGelu"
        }
    ]
}
```

Transformed model graph:

```proto
graph {
  input: "input"
  initializer: "QuickGeluNode_alpha" (FLOAT, value: 1.702)
  node {
    op_type: "Mul"
    input: ["input", "QuickGeluNode_alpha"]
    output: ["QuickGeluNode_mul1_output"]
    name: "QuickGeluNode_mul1"
  }
  node {
    op_type: "Sigmoid"
    input: ["QuickGeluNode_mul1_output"]
    output: ["QuickGeluNode_sigmoid_output"]
    name: "QuickGeluNode_sigmoid"
  }
  node {
    op_type: "Mul"
    input: ["input", "QuickGeluNode_sigmoid_output"]
    output: ["QuickGeluNode_mul2_output"]
    name: "QuickGeluNode_mul2"
  }
  output: "QuickGeluNode_mul2_output"
}
```

Pattern transformation:

```
Original pattern:
[Input] --> QuickGelu --> [Output]

Replaced pattern:
[Input] --> Mul --> Sigmoid --> Mul --> [Output]
             |                    ^
             |                    |
             +--------------------+
             (alpha=1.702)
```

### `DecomposeRotaryEmbedding`

#### Description

Decomposes RotaryEmbedding operator to standard ONNX operators (RoPE). This surgery converts a single RotaryEmbedding node into a subgraph composed of multiple standard ONNX operators.

The RotaryEmbedding function applies rotary position encoding transformations to the real and imaginary parts of input vectors.

Reference: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/contrib_ops/contrib_defs.cc

#### Example

Initial model graph:

```proto
graph {
  input: "input"
  input: "position_ids"
  input: "cos_cache"
  input: "sin_cache"
  node {
    op_type: "RotaryEmbedding"
    input: ["input", "position_ids", "cos_cache", "sin_cache"]
    output: ["rotaryembedding_output"]
    name: "RotaryEmbeddingNode"
  }
  output: "rotaryembedding_output"
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "DecomposeRotaryEmbedding"
        }
    ]
}
```

Transformed model graph:

```proto
graph {
  input: "input"
  input: "position_ids"
  input: "cos_cache"
  input: "sin_cache"
  node {
    op_type: "Reshape"
    input: ["position_ids", "RotaryEmbeddingNode_pos_flat_shape"]
    output: ["RotaryEmbeddingNode_pos_flat_output"]
    name: "RotaryEmbeddingNode_pos_flat"
  }
  node {
    op_type: "Gather"
    input: ["cos_cache", "RotaryEmbeddingNode_pos_flat_output"]
    output: ["RotaryEmbeddingNode_cos_g_output"]
    name: "RotaryEmbeddingNode_cos_g"
    axis: 0
  }
  node {
    op_type: "Gather"
    input: ["sin_cache", "RotaryEmbeddingNode_pos_flat_output"]
    output: ["RotaryEmbeddingNode_sin_g_output"]
    name: "RotaryEmbeddingNode_sin_g"
    axis: 0
  }
  node {
    op_type: "Slice"
    input: ["input", "RotaryEmbeddingNode_slice_start", "RotaryEmbeddingNode_slice_half", "RotaryEmbeddingNode_slice_axes"]
    output: ["RotaryEmbeddingNode_real_output"]
    name: "RotaryEmbeddingNode_real"
  }
  node {
    op_type: "Slice"
    input: ["input", "RotaryEmbeddingNode_slice_half", "RotaryEmbeddingNode_slice_end", "RotaryEmbeddingNode_slice_axes"]
    output: ["RotaryEmbeddingNode_imag_output"]
    name: "RotaryEmbeddingNode_imag"
  }
  node {
    op_type: "Mul"
    input: ["RotaryEmbeddingNode_real_output", "RotaryEmbeddingNode_cos_r_output"]
    output: ["RotaryEmbeddingNode_real_cos_output"]
    name: "RotaryEmbeddingNode_real_cos"
  }
  node {
    op_type: "Mul"
    input: ["RotaryEmbeddingNode_imag_output", "RotaryEmbeddingNode_sin_r_output"]
    output: ["RotaryEmbeddingNode_imag_sin_output"]
    name: "RotaryEmbeddingNode_imag_sin"
  }
  node {
    op_type: "Sub"
    input: ["RotaryEmbeddingNode_real_cos_output", "RotaryEmbeddingNode_imag_sin_output"]
    output: ["RotaryEmbeddingNode_out_r_output"]
    name: "RotaryEmbeddingNode_out_r"
  }
  node {
    op_type: "Mul"
    input: ["RotaryEmbeddingNode_real_output", "RotaryEmbeddingNode_sin_r_output"]
    output: ["RotaryEmbeddingNode_real_sin_output"]
    name: "RotaryEmbeddingNode_real_sin"
  }
  node {
    op_type: "Mul"
    input: ["RotaryEmbeddingNode_imag_output", "RotaryEmbeddingNode_cos_r_output"]
    output: ["RotaryEmbeddingNode_imag_cos_output"]
    name: "RotaryEmbeddingNode_imag_cos"
  }
  node {
    op_type: "Add"
    input: ["RotaryEmbeddingNode_real_sin_output", "RotaryEmbeddingNode_imag_cos_output"]
    output: ["RotaryEmbeddingNode_out_i_output"]
    name: "RotaryEmbeddingNode_out_i"
  }
  node {
    op_type: "Concat"
    input: ["RotaryEmbeddingNode_out_r_output", "RotaryEmbeddingNode_out_i_output"]
    output: ["rotaryembedding_output"]
    name: "RotaryEmbeddingNode_concat"
    axis: -1
  }
  output: "rotaryembedding_output"
}
```

Pattern transformation:

```
Original pattern:
[Input] --> RotaryEmbedding --> [Output]
              ^      ^
              |      |
    [position_ids] [cos_cache, sin_cache]

Replaced pattern:
                                [position_ids]
                                     |
                                  Reshape
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
                [cos_cache] --> Gather       [sin_cache] --> Gather
                                  |                             |
                                  v                             v
                               Reshape                       Reshape
                                  |                             |
[Input] --> Split ----------------+-----------------------------+
              |       |                             |
              v       v                             v
            real    imag                         cos, sin
              |       |                             |
              +-------+-----------------------------+
              |       |       |          |          |
              v       v       v          v          v
            Mul     Mul     Mul        Mul      (RoPE)
              |       |       |          |
              v       v       v          v
            Sub <-----+     Add <--------+
              |               |
              v               v
            real'           imag'
              |               |
              +-------+-------+
                      |
                   Concat
                      |
                      v
                  [Output]
```


### `RMSNormToL2Norm`

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

### `SimplifiedLayerNormToL2Norm`

#### Description
Replace Skip/SimplifiedLayerNormalization nodes with L2Norm subgraph.

#### Example
Initial model graph:

```
SimplifiedLayerNorm pattern:
[Root] --> SimplifiedLayerNormalization
             (axis=-1, epsilon=1e-6)

SkipSimpleLayerNorm pattern:
[Root1] ------------+
                    v
[Root2] --> SkipSimpleLayerNormalization
               (epsilon=1e-6)
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "SimplifiedLayerNormToL2Norm"
        }
    ]
}
```


Transformed model graph:

```
[Root] --> LpNormalization -> Mul
           (p=2, axis=-1)

[Root1] -------> Add
                  |
                  v
[Root2] --> LpNormalization --> Mul
            (p=2, axis=-1)
```

### `MatMulAddToGemm`

#### Description

Replace MatMul + Add with Gemm.

Second MatMul input must be a 2D tensor and the other input of the Add node must be a 1D tensor. If the first MatMul input is more than 2D and the shapes are static, it is reshaped to 2D before the Gemm node and reshaped back to the original shape after the Gemm node.

#### Example

Initial model graph:

```
Static/Dynamic shaped input:
             [2D weight]   [1D bias]
                 |            |
                 v            v
[2D Input] --> MatMul -----> Add

Static shaped input:
             [2D weight]    [1D bias]
                 |             |
                 v             v
[N-D Input] --> MatMul -----> Add
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "MatMulAddToGemm"
        }
    ]
}
```


Transformed model graph:

```
Static/Dynamic shaped inputs:
             [2D weight]
                 |
                 v
[2D Input] --> Gemm (alpha=1.0, beta=1.0)
                 ^
                 |
              [1D bias]

Static shaped inputs:
                          [2D weight]
                              |
                              v
[N-D Input] --> Reshape --> Gemm (alpha=1.0, beta=1.0) --> Reshape
                              ^
                              |
                          [1D bias]
```

### `ReplaceAttentionMaskValue`

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

### `RemoveQDQ`

#### Description

Remove QuantizeLinear and DequantizeLinear node pairs from the graph. Finds Q->DQ patterns and removes them, directly connecting their inputs/outputs. Optionally keeps Clip nodes after graph inputs for value range constraints.

#### Configurations

- `keep_clip_after_inputs`: Whether to keep Clip nodes after graph inputs (default: false).

#### Example

Initial model graph:

```
graph {
  input: "input1"
  node {
    op_type: "QuantizeLinear"
    input: ["input1", "scale", "zero_point"]
    output: ["quantized"]
  }
  node {
    op_type: "DequantizeLinear"
    input: ["quantized", "scale", "zero_point"]
    output: ["dequantized"]
  }
  node {
    op_type: "Conv"
    input: ["dequantized", "weight"]
    output: ["output"]
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveQDQ",
            "keep_clip_after_inputs": false
        }
    ]
}
```

Transformed model graph:

```
graph {
  input: "input1"
  node {
    op_type: "Conv"
    input: ["input1", "weight"]
    output: ["output"]
  }
}
```

### `QDQToClip`

#### Description

Replace QuantizeLinear-DequantizeLinear pairs with Clip operations. Converts Q->DQ patterns to Clip nodes with computed min/max values based on quantization scale and zero point, maintaining the same value constraints.

#### Example

Initial model graph:

```
graph {
  node {
    op_type: "QuantizeLinear"
    input: ["input", "scale", "zero_point"]
    output: ["quantized"]
  }
  node {
    op_type: "DequantizeLinear"
    input: ["quantized", "scale", "zero_point"]
    output: ["output"]
  }
  initializer: "scale" (value: 0.1)
  initializer: "zero_point" (value: 128)
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "QDQToClip"
        }
    ]
}
```

Transformed model graph:

```
graph {
  node {
    op_type: "Clip"
    input: ["input", "clip_min", "clip_max"]
    output: ["output"]
  }
  initializer: "clip_min" (value: -12.8)
  initializer: "clip_max" (value: 12.7)
}
```

### `MatMulToTransposeConvTranspose`

#### Description

Replace 2D Gemm/MatMul with Transpose and 1x1 Conv. When C==1, convert it to 1x1 Conv using TRANSPOSE + CONV + TRANSPOSE sequence for better DLA compatibility.

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "MatMulToTransposeConvTranspose"
        }
    ]
}
```

### `RemoveIntermediarySqueezeAndUnsqueeze`

#### Description

Remove all Unsqueeze and Squeeze operations that aren't directly connected to model inputs. This optimization removes unnecessary dimension expansion operations in the middle of the graph.

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveIntermediarySqueezeAndUnsqueeze"
        }
    ]
}
```

### `RemoveDeqLin`

#### Description

Remove DequantizeLinear nodes that operate on constant initializers. Dequantizes constant initializers at compile time and replaces DequantizeLinear nodes with the pre-computed float values, reducing runtime operations.

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveDeqLin"
        }
    ]
}
```

### `Non4DModelInputs`

#### Description

Add Unsqueeze node to model inputs if input is 2D or 3D. Ensures all model inputs are 4D by adding Unsqueeze operations:
- 2D inputs: adds dimensions at positions [0, -1]
- 3D inputs: adds dimension at position [1]

Updates existing Unsqueeze nodes if present to maintain 4D output.

#### Example

Initial model graph:

```
graph {
  input: "input1" shape: [32, 64]  # 2D input
  input: "input2" shape: [8, 16, 32]  # 3D input
  node {
    op_type: "Conv"
    input: ["input1", "weight1"]
    output: ["conv1_out"]
  }
  node {
    op_type: "Add"
    input: ["input2", "bias"]
    output: ["add_out"]
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DModelInputs"
        }
    ]
}
```

Transformed model graph:

```
graph {
  input: "input1" shape: [32, 64]  # 2D input
  input: "input2" shape: [8, 16, 32]  # 3D input
  node {
    op_type: "Unsqueeze"
    input: ["input1", "input1_unsqueeze_axes"]
    output: ["input1_unsqueeze_input"]
  }
  node {
    op_type: "Unsqueeze"
    input: ["input2", "input2_unsqueeze_axes"]
    output: ["input2_unsqueeze_input"]
  }
  node {
    op_type: "Conv"
    input: ["input1_unsqueeze_input", "weight1"]
    output: ["conv1_out"]
  }
  node {
    op_type: "Add"
    input: ["input2_unsqueeze_input", "bias"]
    output: ["add_out"]
  }
  initializer: "input1_unsqueeze_axes" (value: [0, -1])
  initializer: "input2_unsqueeze_axes" (value: [1])
}
```

### `Non4DModelOutputs`

#### Description

Add Squeeze to non 4D model outputs. Ensures model outputs match expected dimensions by adding Squeeze operations:
- For 2D outputs: squeeze dimensions [0, 3] or [0, 1]
- For 3D outputs: squeeze dimension [2]

Handles special case of Squeeze->Clip->Output pattern.

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DModelOutputs"
        }
    ]
}
```

### `StandaloneReduceSum`

#### Description

Modifies standalone ReduceSum operations (not already transformed) to:
- Set keepdims=1 to preserve dimensions
- Change reduction axis from [1] to [2] for DLA compatibility
- Skip if axes is already [-1] (reduce last dimension)

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "StandaloneReduceSum"
        }
    ]
}
```

### `Gather`

#### Description

Transforms Gather operations for DLA compatibility:
- Converts scalar indices to 1D vector format
- Updates axis attribute from 1 to 2 when needed
- Ensures indices are always in array format

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Gather"
        }
    ]
}
```

### `GatherElements`

#### Description

Transforms GatherElements operations for DLA compatibility:
- Converts scalar indices to 1D vector format
- Reshapes 3D indices to 4D by adding dimension at front
- Ensures indices match expected tensor format

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "GatherElements"
        }
    ]
}
```

### `Non4DInitializers`

#### Description

Expand non-4D initializers to 4D format for DLA compatibility:
- 1D [K] → [1×1×1×K] for Div/Sub/Mul operations
- 2D [C×K] → [K×C×1×1] (transpose and reshape) for most operations
- 2D [K,C] → [1,1,K,C] for Gemm operations
- 3D → 4D by adding dimension at front

Skips MatMul inputs and only expands initializers used by specific operations.

#### Example

Initial model graph:

```
graph {
  node {
    op_type: "Mul"
    input: ["input", "scale"]
    output: ["scaled"]
  }
  node {
    op_type: "Conv"
    input: ["scaled", "weight"]
    output: ["output"]
  }
  initializer: "scale" shape: [64]  # 1D
  initializer: "weight" shape: [128, 64]  # 2D
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DInitializers"
        }
    ]
}
```

Transformed model graph:

```
graph {
  node {
    op_type: "Mul"
    input: ["input", "scale"]
    output: ["scaled"]
  }
  node {
    op_type: "Conv"
    input: ["scaled", "weight"]
    output: ["output"]
  }
  initializer: "scale" shape: [1, 1, 1, 64]  # 1D -> 4D
  initializer: "weight" shape: [64, 128, 1, 1]  # 2D -> 4D (transposed)
}
```

### `RemoveAllTensorValueShapes`

#### Description

Remove all tensor shape information from value_info. Clears shape fields from all value_info entries in the graph, useful for models where shape inference is not needed or causes issues.

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "RemoveAllTensorValueShapes"
        }
    ]
}
```

### `Non4DReshape`

#### Description

Convert 3D Reshape operations to 4D for DLA compatibility. Updates Reshape nodes with 3D target shapes to 4D by inserting 1 at the appropriate position (e.g., [-1, 512, 768] -> [1, -1, 512, 768]).

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DReshape"
        }
    ]
}
```

### `Non4DExpand`

#### Description

Convert 3D Expand operations to 4D for DLA compatibility. Updates Expand nodes with 3D shapes to 4D by inserting 1 at dimension 0 (e.g., [2, 3, 4] -> [1, 2, 3, 4]).

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DExpand"
        }
    ]
}
```

### `Non4DTranspose`

#### Description

Update Transpose permutation attributes for non-4D tensors. Adjusts perm attribute of Transpose nodes to handle 4D tensors:
- 2D: [T0, T1] -> [0, 1, T0 + 2, T1 + 2]
- 3D: [T0, T1, T2] -> [0, T0 + 1, T1 + 1, T2 + 1]

Ensures transpose operations work correctly when tensors are expanded to 4D.

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DTranspose"
        }
    ]
}
```

### `Non4DSlice`

#### Description

Transform Slice axes of non4D tensors. Updates Slice operations to work with 4D tensors:
- Changes axes from specific dimensions to [-1] (last dimension)
- Only processes nodes not already transformed
- Ensures slicing operations remain valid after tensor expansion

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DSlice"
        }
    ]
}
```

### `Non4DLpNorm`

#### Description

Transform LpNormalization axes of non4D tensors. Updates LpNormalization operations for 4D tensor compatibility:
- Changes axis attribute to -1 (last dimension)
- Ensures normalization occurs along the correct dimension after tensor expansion to 4D

#### Example

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Non4DLpNorm"
        }
    ]
}
```

### `Flatten`

#### Description

Replace Flatten operations with Reshape operations using shape [1, 1, 1, -1]. This maintains compatibility with DLA which may not support Flatten directly, while preserving the flattening behavior.

#### Example

Initial model graph:

```
graph {
  node {
    op_type: "Flatten"
    input: ["input"]
    output: ["flattened"]
    axis: 1
  }
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "Flatten"
        }
    ]
}
```

Transformed model graph:

```
graph {
  node {
    op_type: "Reshape"
    input: ["input", "reshape_axes"]
    output: ["flattened"]
    name: "flatten_reshape"
  }
  initializer: "reshape_axes" (value: [1, 1, 1, -1])
}
```

### `AddIntermediateTensorsToOutputs`

#### Description

Debug function to add intermediate tensors to outputs. Exposes intermediate tensor values as model outputs for debugging:
- Can specify specific tensors to add via intermediate_tensor_to_add list
- If not specified, adds all intermediate tensors from node outputs
- Useful for inspecting values at different stages of the graph

#### Configurations

- `intermediate_tensor_to_add`: List of intermediate tensor names to expose as outputs.

#### Example

Initial model graph:

```
graph {
  input: "input"
  node {
    op_type: "Conv"
    input: ["input", "weight"]
    output: ["conv1_output"]
  }
  node {
    op_type: "Relu"
    input: ["conv1_output"]
    output: ["relu1_output"]
  }
  node {
    op_type: "Conv"
    input: ["relu1_output", "weight2"]
    output: ["final_output"]
  }
  output: "final_output"
}
```

After applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "AddIntermediateTensorsToOutputs",
            "intermediate_tensor_to_add": ["conv1_output", "relu1_output"]
        }
    ]
}
```

Transformed model graph:

```
graph {
  input: "input"
  node {
    op_type: "Conv"
    input: ["input", "weight"]
    output: ["conv1_output"]
  }
  node {
    op_type: "Relu"
    input: ["conv1_output"]
    output: ["relu1_output"]
  }
  node {
    op_type: "Conv"
    input: ["relu1_output", "weight2"]
    output: ["final_output"]
  }
  output: "final_output"
  output: "conv1_output"    # Added for debugging
  output: "relu1_output"     # Added for debugging
}
```

### `ReshapeReduceSum`

#### Description

Transform Reshape-ReduceSum pattern to parallel Slice-ReduceSum-Concat for DLA. Splits a Reshape-ReduceSum operation into parallel paths to improve DLA performance:
- Replaces single Reshape-ReduceSum with two parallel Slice operations
- Each slice processes part of the data with ReduceSum
- Results are concatenated to produce the same output
- Enables better parallelization on DLA hardware

#### Example

Initial pattern:
```
    x, shape
        |
    Reshape axes
        |   /
    ReduceSum
        |
    reducesum_output
```

after applying:


```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ReshapeReduceSum"
        }
    ]
}
```

Transformed pattern:
```
        x
    /        \
    Slice      Slice
    |           |
    ReduceSum   ReduceSum
        \       /
        Concat
        |
    reducesum_output
```


### `ReshapeClipReduceSum`

#### Description

Transform Reshape-Clip-ReduceSum pattern to parallel paths for DLA optimization. Similar to ReshapeReduceSum but includes Clip operation:
- Splits Reshape-Clip-ReduceSum into two parallel processing paths
- Each path: Slice -> Clip -> ReduceSum
- Maintains numerical equivalence while improving DLA parallelization
- Useful for quantized models where Clip enforces value ranges

#### Example

Initial pattern:
```
        (x)
        |
    Reshape
        |
    Clip
        |
    ReduceSum
        |
    (reducesum_output)
```

after applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ReshapeClipReduceSum"
        }
    ]
}
```

Transformed pattern:
```
        (x)
    /        \
    Slice      Slice
    |           |
    Clip        Clip
    |           |
    ReduceSum   ReduceSum
        \\       /
        Concat
        |
    (reducesum_output)
```


### `ReduceMax`

#### Description

Add Reshape after ReduceMax operations for DLA compatibility. Modifies ReduceMax operations to ensure output shape compatibility:
- Adds a Reshape node after ReduceMax with shape [1,1,1,3600]
- Updates axes to [3] and keepdims to 1
- Ensures ReduceMax output has the expected 4D shape for DLA
- Hardcoded output shape may need adjustment for different models

#### Example

Initial pattern:
```
    data   axes   keepdims
        |   /     /
    ReduceMax
        |
    reducemax_output
```

after applying:

```json
{
    "type": "GraphSurgeries",
    "surgeries": [
        {
            "surgeon": "ReduceMax"
        }
    ]
}
```

Transformed pattern:
```
    data   axes   keepdims
        |   /     /
    ReduceMax
        |    reshape_shape
        |   /
    Reshape
        |
    reducemax_output
```


## ORT Performance Tuning

ONNX Runtime provides high performance across a range of hardware options through its Execution Providers interface for different execution
environments.
For each model running with each execution provider, there are settings that can be tuned (e.g. thread number, execution mode, etc) to
improve performance.
`OrtSessionParamsTuning` covers basic knobs that can be leveraged to find the best performance for your model and hardware.

### Example Configuration

```json
{
    "type": "OrtSessionParamsTuning",
    "data_config": "session_params_tuning_data_config",
    "batch_size": 1,
    "providers_list" : [
        [
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2147483648, // 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": true,
            },
        ],
        "CPUExecutionProvider",
    ],
    "enable_profiling": false
}
```

Check out [this file](https://github.com/microsoft/olive-recipes/blob/main/intel-bert-base-uncased-mrpc/aitk/user_script.py)
for an example implementation of `"user_script.py"` and `"calib_data_config/dataloader_config/type"`.

[1]: <https://onnxruntime.ai/docs/performance/quantization.html> "ONNX Runtime Quantization"
[2]: <https://onnxruntime.ai/docs/performance/quantization.html#dynamic-quantization> "Dynamic Quantization"
[3]: <https://onnxruntime.ai/docs/performance/quantization.html#static-quantization> "Static Quantization"
