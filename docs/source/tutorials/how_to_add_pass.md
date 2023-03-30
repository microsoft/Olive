(How-to-add-pass)=
# How to add new Pass

Olive provides simple interface to introduce new model optimization techniques. Each optimization technique is
represented as a Pass in Olive.

To introduce a new Pass follow these 3 steps.

## 1. Define a new class

Define a new class using Pass as the base class. For example

```python
from olive.passes import Pass

class NewOptimizationTrick(Pass):

```

## 2. Define configuration

Next, define the options used to configure this new technique by defining static method `_default_config`. The method should
return `Dict[str, PassConfigParam]`.

```python
    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(
                type_=str,
                default="static",
                searchable_values=Categorical(["dynamic", "static"]),
                description="""
                    Onnx Quantization mode. 'dynamic' for dynamic quantization,
                    'static' for static quantization.
                """,
            )
        }

```

### 3. Implement the run function

The final step is to implement the `_run_for_config` method to optimize the input model. Olive Engine will invoke the
method while auto tuning the model. This method will also receive a search point (one set of configuration option from
the search space created based on the options defined in `_default_config()`) along with output path. The method
should return a valid OliveModel which can be used as an input for the next Pass.

```python
    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
```
