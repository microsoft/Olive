# ONNX

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

Please refer to [OnnxPeepholeOptimizer](../../../reference/pass.rst#onnx_peephole_optimizer) for more details about the pass and its config parameters.

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

Please refer to [OrtTransformersOptimization](../../../reference/pass.rst#ort_transformers_optimization) for more details about the pass and its config parameters.

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

## Append Pre/Post Processing Ops

`AppendPrePostProcessingOps` inserts pre and post processing ops into the ONNX graph.

### Example Configuration

```json
{
    "type": "AppendPrePostProcessingOps",
    "tool_command": "superresolution",
    "tool_command_args": {
        "output_format": "png"
    }
}
```

```json
{
    "type": "AppendPrePostProcessingOps",
    "tool_command": "whisper",
    "tool_command_args": {
        "use_audio_decoder": true
    }
}
```

`AppendPrePostProcessingOps` also supports pre/post processing ops by leveraging the [onnxruntime-extension steps](https://github.com/microsoft/onnxruntime-extensions/tree/main/onnxruntime_extensions/tools/pre_post_processing/steps) and `PrePostProcessor`.
You can refer to [here](https://github.com/microsoft/onnxruntime-extensions/blob/main/onnxruntime_extensions/tools/Example%20usage%20of%20the%20PrePostProcessor.md) to see how to leverage `PrePostProcessor` to customize pre and post processing ops.

* Olive introduces two placeholders to represent the model input/output shape dimension value: `__model_input__` and `__model_output__`.
* To support the IoMapEntry, the step need choose use the full form. For example:

```json
    "YCbCrToPixels": {
        "params": {
            "layout": "BGR",
        },
        "io_map": [
            ["Y1_uint8", 0, 0],
            ["Cb1_uint8", 0, 1],
            ["Cr1_uint8", 0, 2],
        ],
    }
```

* The `tool_command_args` will be used to describe the input parameters to create the `PrePostProcessor` instance. It is list of `PrePostProcessorInput`.
  The `name` is the tensor name. The `data_type` and `shape` will be used to create the tensor type. The `shape` can be a list of integers or a list of string.

Users that write their own pre/post processing steps need to have the knowledge about whether the step includes the operators that is built-in support or supported in onnxruntime-extensions.
For example, for some ops like `ConvertImageToBGR` which requires other extensions may be incompatible with ort-web, user need to exclude this kind of ops to generate proper models.

Here are some examples to describe the pre/post processing which is exactly same with [superresolution](https://github.com/microsoft/onnxruntime-extensions/blob/main/onnxruntime_extensions/tools/add_pre_post_processing_to_model.py#L89)

```json
{
    "pre": [
        {"ConvertImageToBGR": {}},
        {
            "Resize": {
                "resize_to": [
                    {"type": "__model_input__", "input_index": 0, "dim_index": -2},
                    {"type": "__model_input__", "input_index": 0, "dim_index": -1},
                ]
            }
        },
        {
            "CenterCrop": {
                "height": {"type": "__model_input__", "input_index": 0, "dim_index": -2},
                "width": {"type": "__model_input__", "input_index": 0, "dim_index": -1},
            }
        },
        {"PixelsToYCbCr": {"layout": "BGR"}},
        {"ImageBytesToFloat": {}},
        {"Unsqueeze": {"axes": [0, 1]}},
    ],
    "post": [
        {"Squeeze": {"axes": [0, 1]}},
        {"FloatToImageBytes": {"name": "Y1_uint8"}},
        {
            "Resize": {
                "params": {
                    "resize_to": [
                        {"type": "__model_output__", "output_index": 0, "dim_index": -2},
                        {"type": "__model_output__", "output_index": 0, "dim_index": -1},
                    ],
                    "layout": "HW",
                },
                "io_map": [["PixelsToYCbCr", 1, 0]],
            }
        },
        {"FloatToImageBytes": {"multiplier": 1.0, "name": "Cb1_uint8"}},
        {
            "Resize": {
                "params": {
                    "resize_to": [
                        {"type": "__model_output__", "output_index": 0, "dim_index": -2},
                        {"type": "__model_output__", "output_index": 0, "dim_index": -1},
                    ],
                    "layout": "HW",
                },
                "io_map": [["PixelsToYCbCr", 2, 0]],
            }
        },
        {"FloatToImageBytes": {"multiplier": 1.0, "name": "Cr1_uint8"}},
        {
            "YCbCrToPixels": {
                "params": {
                    "layout": "BGR",
                },
                "io_map": [
                    ["Y1_uint8", 0, 0],
                    ["Cb1_uint8", 0, 1],
                    ["Cr1_uint8", 0, 2],
                ],
            }
        },
        {"ConvertBGRToImage": {"image_format": "png"}},
    ],
    "tool_command_args": [
        {
            "name": "image",
            "data_type": "uint8",
            "shape": ["num_bytes"],
        }
    ],
    "target_opset": 16,
}
```

## Insert Beam Search Op

`InsertBeamSearch` chains two model components (for example, encoder and decoder) together by inserting beam search op in between them.

### Example Configuration

```json
{
    "type": "InsertBeamSearch",
    "no_repeat_ngram_size": 4
}
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

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert/user_script.py)
for an example implementation of `"user_script.py"` and `"calib_data_config/dataloader_config/type"`.

[1]: <https://onnxruntime.ai/docs/performance/quantization.html> "ONNX Runtime Quantization"
[2]: <https://onnxruntime.ai/docs/performance/quantization.html#dynamic-quantization> "Dynamic Quantization"
[3]: <https://onnxruntime.ai/docs/performance/quantization.html#static-quantization> "Static Quantization"
