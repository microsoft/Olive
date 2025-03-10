# How to add new optimization Pass

Olive provides simple interface to introduce new model optimization techniques. Each optimization technique is
represented as a Pass in Olive.

To introduce a new Pass follow these 3 steps.

## 1. Define a new class

Define a new class using Pass as the base class. For example

```python
from olive.passes import Pass

class NewOptimizationTrick(Pass):
    # Add any required data members to the class
```

## 2. Define configuration

Next, define the options used to configure this new technique by defining static method `_default_config`. This method
takes an `AcceleratorSpec` as input and returns `Dict[str, PassConfigParam]`.

`AcceleratorSpec` is a dataclass that holds the information about the accelerator. The dataclass has the following fields:

- `accelerator_type`: type of the accelerator. For example, `CPU`, `GPU` etc.

- `execution_provider`: execution provider for the accelerator. For example, `CPUExecutionProvider`, `CUDAExecutionProvider` etc.
    Please note if user specify some execution providers that don't belong to the installed onnxruntime, these execution providers
    will be ignored. For example, if user specify CUDA, TensorRT, DML execution provider, but the onnxruntime-gpu is installed then
    the DML execution provider will be ignored since it is only available in onnxruntime-directml package.

`PassConfigParam` is a dataclass that holds the information about the configuration option. The dataclass has the following fields:

- `type_` : type of the parameter

- `required` : whether the parameter is required
- `category`: The param category. It could be the following values:

    * `object` : whether the parameter is an object/function. If so, this parameter accepts the object or a string with the
    name of the object in the user script. The type must include `str`.
    * `path` : whether the parameter is a path. If so, this file/folder will be uploaded to the host system.
    * `data`: whether the parameter is a data path, which will be used to do data path normalization based on the data root.

- `description` : description of the parameter

- `default_value`: default value for the parameter. This value is used as the default when not searching or when there are no searchable values.
    Must be the same type as the parameter or a ConditionalDefault SearchParameter.

- `search_defaults`: default search values for the parameter. This value is used as the default when searching.
    Must be a Categorical or Conditional SearchParameter.

### Example
```python
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            # required parameter
            "param1": PassConfigParam(type_=int, required=True, description="param1 description"),
            # optional parameter with default value
            "param2": PassConfigParam(type_=int, default_value=1, description="param2 description"),
            # optional parameter with default value and searchable values
            "param3": PassConfigParam(
                type_=int, default_value=1, search_defaults=Categorical([1, 2, 3]), description="param3 description"
            ),
            # optional parameter with `category` set to `object`
            # the value of this parameter can be a string or a function that takes a string and returns the object,
            # say a class ObjectClass
            "param4": PassConfigParam(
                type_=Union[str, Callable[[str], Pass]], category=ParamCategory.OBJECT, description="param4 description"
            ),
            # optional parameter with default_value that depends on another parameter value
            "param5": PassConfigParam(
                type_=int,
                default_value=ConditionalDefault(parents="param2", support={(1,): 2, (2,): 3}, default=4),
                description="param5 description",
            ),
            # optional parameter with search_defaults that depends on other parameter values
            "param6": PassConfigParam(
                type_=int,
                default_value=1,
                search_defaults=Conditional(
                    parents=("param2", "param3"),
                    # invalid if (param2, param3) not in [(1, 1), (1, 2)]
                    support={
                        (1, 1): Categorical([1, 2, 3]),
                        (1, 2): Categorical([4, 5, 6]),
                    },
                ),
                description="param6 description",
            ),
        }

```

## 3. Implement the run function

The final step is to implement the `_run_for_config` method to optimize the input model. Olive Engine will invoke the
method while auto tuning the model. This method will also receive a search point (one set of configuration option from
the search space created based on the options defined in `_default_config(cls, accelerator_spec: AcceleratorSpec)`) along
with output path. The method should return a valid OliveModelHandler which can be used as an input for the next Pass.

```python
    def _run_for_config(self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str) -> ONNXModelHandler:
```

## 4. Update olive_config.json

The `olive_config.json` lists the features of the new Pass to help Olive determine when to use the pass. Add an entry
for the new pass with relevant info.

### Example
```
        "NewOptimizationTrick": {
            "module_path": "olive.passes.onnx.new_opt_pass.NewOptimizationTrick",
            "supported_providers": [ "CPUExecutionProvider" ],
            "supported_accelerators": [ "cpu" ],
            "supported_precisions": [ "int8" ],
            "supported_algorithms": [ "new_algorithm" ],
            "supported_quantization_encodings": [  ],
            "dataset": false,
            "extra_dependencies": [ "dependent_on_this_external_pkg" ],
            "run_on_target": true
        },
```
