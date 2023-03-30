# Olive Design Documentation

## Overview
This document describes the Olive components, and some implementation details. The components covered are:
- [Pass](#pass)
    - [Base Class](#base-class)
    - [default_config](#default_config)
    - [run](#run)
    - [Implemented Passes](#implemented-passes)
- [Engine](#engine)
    - [run](#run-1)
    - [cache](#cache)
- [Search](#search)
    - [Search Parameter](#searchparameter)
    - [Search Space](#searchspace)
    - [Search Algorithm](#searchalgorithm)
    - [Search Results](#searchresults)
- [Search Strategy](#search-strategy)
    - [Execution order](#execution-order)
    - [User Interface](#user-interface)
    - [Implementation](#implementation)
- [System](#system)
    - [OliveSystem Class](#olivesystem-class)
- [olive.workflows.run](#oliveolive)
    - [User Interface](#user-interface-1)

## Pass
Passes are the building blocks of an Olive workflow. Olive uses multiple Passes to process an input model.

### Base Class
The base class for Pass:
```python
class Pass(ABC):
    def __init__(self, config: Union[Dict[str, Any], BaseModel], disable_search: Optional[bool] = False):
        ...

    @classmethod
    def get_config_class(cls, disable_search: Optional[bool] = False) -> Type[BaseModel]:
        ...

    @staticmethod
    @abstractmethod
    def default_config() -> Dict[str, PassConfigParam]:
```
where `BaseModel` is a [pydantic model](https://docs.pydantic.dev/usage/models/)

It is initialized using:
- Config dictionary `{“param_name”: param_value}` and boolean `disable_search`.

  If `disable_search=False`, use default search parameters, if any, for parameters that are not specified
        in the config. Else use the default value.
- Pydantic model which behaves like a dataclass with type validation. Each pass class has a class method `get_config_class` which returns the pass specific pydantic model that users can instantiate.

Optional parameters can be fixed values or search values which are prescribed using `SearchParameter`.

Searchable parameters have default search values which can be used by assigning the config value as `SEARCHABLE_VALUES`. Optional parameters use the default fixed value, also assignable using `DEFAULT`, if not assigned.

During initialization, the pass compares the user provided config and pass config class to create a dictionary for fixed parameters (`_fixed_params`) and search parameters (`_search_space`).

### default_config
Each pass must implement the default_config static method. It returns a dictionary
```python
{“param_name”: PassConfigParam}
```

`PassConfigParam` is a dataclass which describes whether the parameter is required and holds its default fixed value and search values.

**Note:**
- To support hashing, the parameter values must be jasonify-able. So, they can only be string, int, float, tuple, bool, list, dict, or None.
- We will add support for other python objects and callables in the future for local use.

### run
To run a pass, the search parameters need to be assigned a value each from their search options (support). We call this a `search_point`.

The pass then combines those with the fixed parameters to form a complete configuration using `config_at_search_point`. This run configuration is also used by the engine as a unique id to cache the pass run.

### Implemented Passes
- OnnxConversionPass
- OrtTransformersOptimization
- OnnxDynamicQuantization
- OnnxStaticQuantization
- OnnxQuantization

## Engine
The engine is responsible for
- Managing and executing the passes
- Manage input, output and intermediate models
- Evaluating intermediate or final output models as needed

The user configures the engine using a configuration dictionary that also selects the search strategy to use. Passes are then created and registered along with their evaluators if any.

The engine maintains a cache directory to cache pass runs, models and evaluations.

### run
This method runs all the registered passes on the input model and produces one or more candidate models.

The engine delegates the search to the search strategy. It only executes passes and evaluates output models as prescribed by the search strategy. Evaluation results are fed back to the search strategy.

### cache
The engine maintains a cache directory with three sub-directories:
- `model`: stores model files and their corresponding json configs (framework, model path, and other information). Each model created during execution is given a unique model id. The input model is identified by the hash of its model file contents.
- `run`: stores the id of the output model (present in the model cache) for each run of a pass. The cache is indexed by `{pass type}_{hash of pass config}_{input model id}`.
- `evaluation`: stores the evaluation results for models. The cache is indexed by the model id.

## Search
Olive workflows support search parameters which are optimized using search algorithms.

At the most basic level is `SearchParameter` which describes the options for search parameters. `SearchSpace` combines search parameters for one or more passes and `SearchAlgorithm` provides different sampling algorithms to search for the best parameter configuration (search point) from the search space.

### SearchParameter
A search parameter defines a discrete categorical distribution.

There are two types of search parameters:
- `Categorical`: Discrete categorical distribution with uniform weights. `Boolean` is a special case of `Categorical` with support `[True, False]`.
- `Conditional`: Conditional discrete categorical distribution. The parameter has one or more parents from the same pass configuration. Each combination of parent values has an associated Categorical search parameter.

**Note:**
- There cannot be any cyclic parent child dependencies.
- Search algorithms order the search parameters topologically so that the parents are sampled before the children.

### SearchSpace
Search space combines search parameters from one or more passes and provides methods to iterate over the search space (`iterate`) or generate random samples (`random_sample`).

The search parameters are passed to SearchSpace and saved in the format
```python
{“pass_id/space_name”: {“param_name”: SearchParamter}}
```

This hierarchical structure that indexes the search parameter dictionaries by `pass_id` allows us to take advantage of the chronological order of pass execution if wanted.

The corresponding conceptual search space is the space of all possible parameter configurations. Each point in this space is called a `search point`:
```python
{“pass_id/space_name”: {“param_name”: param_value}}
```

### SearchAlgorithm
Search algorithm operates over a search space and provides samples/trials (search points) from the search space to execute and evaluate.

Each search algorithm provides the methods:
- `suggest`: returns a search point to execute and evaluate. The algorithm can sample a search point based on the evaluation results for previously suggested points.
- `report`: report evaluation results for a search point. The search point can also be pruned if it contains invalid pass configs or failed during execution/evaluation.

The following search algorithms have been implemented:
- `ExhaustiveSearchAlgorithm`: Exhaustively iterates over the search space.
- `RandomSearchAlgorithm`: Randomly samples points from the search space without replacement.
- `OptunaSearchAlgorithm`: Abstract base class for algorithms built using `optuna` samplers. This class cannot be used directy
    - `TPESearchAlgorithm`: Uses optuna `TPESampler`.

### SearchResults
`SearchResults` stores evaluation results for samples made from a search space and provides tools to analyze and select the best search point/s.

Results are reported using the `record` method.

Currently `best_search_point` selects the best search point by maximizing/minimizing metrics using tie breaking. We intend to provide different model selection strategies for both single and multi-objective optimization.

## Search Strategy
Search strategy provides an optimization pipeline that finds the best search point from the search space of one or more passes.

It consists of two sub-components – `execution_order` and `search_algorithm`. Search algorithm has been covered in the previous section.

### Execution Order
The execution order defines the order in which the passes are optimized.

Currently, we support two execution orders:
- `joint`: The search spaces of all passes are combined and searched together to find the best search point. Each search point that is evaluated has parameters for the search parameters of all passes.
- `pass-by-pass`: The search space of each pass is searched and optimized independently in order. For instance, for two passes `PassA, PassB` - `PassA` is optimized first, and its best output model is used as the input model for optimizing `PassB`. The final best output model is the best output model of `PassB`. **Note:** This execution order assumes that the intermediate searches produce single best models but we will support multiple best models in the future.

The above execution orders are implemented individually but both are specific instances of an execution order consisting of a chain of groups of passes.
- `joint`: `[PassA, PassB]`
- `pass-by-pass`: `PassA -> PassB`

It is possible to extend this to other instances such as
```
PassA -> [PassB, PassC] -> PassD
```

We intend to generalize the execution order implementation to support such chain.

There are also plans to support nested execution order of forms like `(PassA, [PassB, PassC])` where passes `B` and `C` are optimized for each search point in pass `A`.

### System
System encapsulates the system Olive is targeting as well the system on which Olive is running. A pass can select a system as a 'host' and the evaluator can 'target' a system. The 'host' and 'target' are user provided configuration options. Olive provides AzureMLSystem and DockerSystem in addition to LocalSystem.

### OliveSystem Class
The base class for System:
```python
class OliveSystem(ABC):
    def __init__(self, device: Device):
        ...
    @abstractmethod
    def run_pass(
        self,
        the_pass: Pass,
        model: OliveModel,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """

    @abstractmethod
    def evaluate_model(self, model: OliveModel, metrics: List[Metric]) -> Dict[str, Any]:
        """
        Evaluate the model
        """
```

The `run_pass` method is responsible to run the input Pass for the given model on this system.

The `evaluate_model` method is responsible to evaluate the model on this system.

### User Interface
The user or engine interacts with the search strategy using the following methods:
- `next_step`: returns the next search point to execute and evaluate, and the if of input model to use.
- `record_feedback_signal`: record the evaluation results for a search point along with ids of models generated during execution.

### Implementation
The current implementation of search strategy is not mature and will be updated in a follow-up PR to generalize the chained execution order.

Currently, it maintains a list of “search spaces groups” which are just groups of search parameter dictionaries from passes that are meant to be optimized together. Joint has one search spaces group while pass-by-pass has a group for each pass.

## olive.workflows.run
Users can build their own workflows using the python api but we also provide a command-line interface to execute workflows specified using json configuration files.

### User Interface
```bash
python –m olive.workflows.run --config <json config path>
```

Please refer to `examples/bert_ptq_cpu/bert_config.json` for an example.

This tool is also available for use in the python api
```python
from olive.workflows import run

run(config)
```
where `config` is a path to the json config file or a config dictionary.
