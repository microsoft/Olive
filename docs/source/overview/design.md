(Design)=
# Design
In this section, we discuss the core design concepts of Olive. Olive is composed of modular components
that are composed to construct a model optimization workflow.

The workflow which is run by the **Engine** is composed of **Passes** that are executed in a specific order.
Each Pass is responsible for performing a specific optimization on the model. Each Pass might have a set of parameters that
can be tuned to achieve the best metrics, say accuracy and latency, that are evaluated by the respective **Evaluator**.
The Engine employs a **Search Strategy** that uses a **Search Algorithm** to auto-tune each Pass one by one or set of Passes
together.

Each Pass can be run on any host **System** and its output model can be evaulated on the desired target **System**.

Olive can be used to find the "best" candidate model, pareto frontier models (for multi-objective search), or the first model
that meets some metric goals.

The following diagram illustrates the relationship between the different components:

```{figure} ../images/olive-design.png
:align: center
:alt: olive-design
:width: 1000px
```

## Pass
Passes are the building blocks of an Olive workflow. A Pass performs a specific model optimization technique such as ONNX
conversion or ONNX quantization.

Each pass is configured using a set of required and optional parameters. A Pass config parameter might have a
default value and default searchable values. When initializing a pass, the user can chose to set the values of parameters to
their default value (no search), default searchable values (search for the best value from the possible options) or a
combination of the two (fix some parameters to a certain value, default or user provided, and/or search for other parameters).

## System
A system is the environment (OS, hardware spec, device platform, supported EP) that a Pass is run in or a Model is
evaluated on. It can thus be the **host** of a Pass or the **target** of an evaluation.

The Pass **host** provides an environment with the required OS platform and packages to execute the optimization
technique.

The Evaluator **target** will ideally have the same specs as the desired deployment environment.

Olive supports three main types of Systems:
- **LocalSystem:** The local machine and environment. The Pass or model evaluation is run in the active python environment on
  the host machine.
- **AzureMLSystem:** An AzureML workspace compute and environment.
- **DockerSystem:** A docker container running on the host machine.

## Evaluator
An Evaluator is used to evaluate a model on a specific **target** and return values for some **metrics**.

To initialize an Evaluator, the user must provide a list of **Metrics** and the **target** system.

Olive provides two in-build **Metrics**, accuracy and latency, along with the options of the user to provide their own
**custom** metric.

## Engine
The **Engine** executes an Olive workflow and is responsible for
- Managing and executing the passes
- Managing input, output and intermediate models
- Evaluating intermediate or final output models as needed

The Engine uses a **Search Strategy** to auto-tune the Passes one by one or the set of Passes together. The Search Strategy in
turn employs a search algorithm to sample the search spaces of the Passes.

The user configures the engine using a configuration dictionary that also selects the search strategy to use. Passes are then
created and **registered** along with their host system and evaluators if any.

The engine also maintains a cache directory to cache pass runs, models and evaluations.

## Search Strategy
Search strategy provides an optimization pipeline that finds the best search point from the search space of one or more passes.

It consists of two sub-components – `execution_order` and `search_algorithm`.

### Execution Order
The execution order defines the order in which the passes are optimized.

Currently, we support two execution orders:
- `joint`: The search spaces of all passes are combined and searched together to find the best search point. Each search point
that is evaluated has parameters for the search parameters of all passes.
- `pass-by-pass`: The search space of each pass is searched and optimized independently in order.

### Search Algorithm
Search algorithm operates over a search space and provides samples/trials (search points) from the search space to execute and evaluate.

The following search algorithms have been implemented:
- `exhaustive`: Exhaustively iterates over the search space.
- `random`: Randomly samples points from the search space without replacement.
- `tpe`: ample using TPE (Tree-structured Parzen Estimator) algorithm to sample from the search space.
