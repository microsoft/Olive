How-to
=======
Find more details on specific Olive capabilities, such as quantization, running workflows on remote compute, model packaging, conversions, and more!

Set-up
------

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

.. grid:: 1 1 1 1
   :class-container: cards

   .. grid-item-card::

      **How to install Olive**

      Learn how to install `olive-ai`.

      :octicon:`arrow-right;1em;sd-text-info` `Install Olive <installation.html>`_

Working with the CLI
--------------------

.. toctree::
   :maxdepth: 1
   :hidden:

   cli/cli-auto-opt
   cli/cli-finetune
   cli/cli-quantize
   cli/cli-run

The Olive CLI provides a set of primitives such as `quantize`, `finetune`, `onnx-graph-capture`, `auto-opt` that enable you to *easily* optimize select models and experiment with different cutting-edge optimization strategies without the need to define workflows.

.. tip:: For users new to Olive, we recommend that you start with the CLI.

.. grid:: 2 2 2 3
   :class-container: cards

   .. grid-item-card::

      **Auto Optimizer**

      Learn how to use the `olive auto-opt` command to take a PyTorch/Hugging Face model and turn it into an optimized ONNX model.

      :octicon:`arrow-right;1em;sd-text-info` `olive auto-opt <cli/cli-auto-opt.html>`_

   .. grid-item-card::

      **Finetune**

      Learn how to use the `olive finetune` command to create (Q)LoRA adapters.

      :octicon:`arrow-right;1em;sd-text-info` `olive finetune <cli/cli-finetune.html>`_

   .. grid-item-card::

      **Quantize**

      Learn how to use the `olive quantize` command to quantize your model with different precisions and techniques such as AWQ.

      :octicon:`arrow-right;1em;sd-text-info` `olive quantize <cli/cli-quantize.html>`_

   .. grid-item-card::

      **Execute Olive Workflows**

      Learn how to use the `olive run` command to execute an Olive workflow.

      :octicon:`arrow-right;1em;sd-text-info` `olive run <cli/cli-run.html>`_

Working with the Python API
---------------------------

.. toctree::
   :maxdepth: 1
   :hidden:

   python-api/python-api-overview
   python-api/python-api-auto-opt
   python-api/python-api-finetune
   python-api/python-api-quantize
   python-api/python-api-run

The Olive Python API provides programmatic access to all CLI functionality, allowing you to execute optimization workflows directly in Python and receive structured `WorkflowOutput` objects with detailed results.

.. tip:: The Python API offers the same functionality as CLI commands but with Python-native parameters and structured return values.

.. grid:: 2 2 2 3
   :class-container: cards

   .. grid-item-card::

      **Python API Overview**

      Learn about the Python API structure, return types, and common usage patterns.

      :octicon:`arrow-right;1em;sd-text-info` `Python API Overview <python-api/python-api-overview.html>`_

   .. grid-item-card::

      **Auto Optimization API**

      Use the `auto_opt()` function to optimize models programmatically with structured results.

      :octicon:`arrow-right;1em;sd-text-info` `auto_opt() API <python-api/python-api-auto-opt.html>`_

   .. grid-item-card::

      **Fine-tuning API**

      Use the `finetune()` function to create LoRA/QLoRA adapters with Python control flow.

      :octicon:`arrow-right;1em;sd-text-info` `finetune() API <python-api/python-api-finetune.html>`_

   .. grid-item-card::

      **Quantization API**

      Use the `quantize()` function to apply various quantization techniques programmatically.

      :octicon:`arrow-right;1em;sd-text-info` `quantize() API <python-api/python-api-quantize.html>`_

   .. grid-item-card::

      **Workflow Execution API**

      Use the `run()` function to execute complex workflows from configuration files or dictionaries.

      :octicon:`arrow-right;1em;sd-text-info` `run() API <python-api/python-api-run.html>`_

How to prepare configuration
----------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   configure-workflows/build-workflow

For more complex scenarios, you can create a new configuration from scratch.

.. grid:: 2 2 2 3
   :class-container: cards

   .. grid-item-card::

      **How to write a new .json configuration**

      Learn how to write a new .json configure from scratch.

      :octicon:`arrow-right;1em;sd-text-info` `Configure workflow <configure-workflows/build-workflow.html>`_

How to customize configuration
------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   configure-workflows/how-to-configure-model
   configure-workflows/pass-configuration
   configure-workflows/how-to-configure-data
   configure-workflows/metrics-configuration
   configure-workflows/model-packaging
   configure-workflows/systems
   configure-workflows/engine-configuration

For more complex scenarios, you can customize configuration where you can run any of the 40+ supported optimization techniques in a sequence.

.. grid:: 2 2 2 3
   :class-container: cards

   .. grid-item-card::

      **How to configure model**

      Learn how to configure input model.

      :octicon:`arrow-right;1em;sd-text-info` `Configure models <configure-workflows/how-to-configure-model.html>`_

   .. grid-item-card::

      **How to configure passes**

      Learn how to configure passes.

      :octicon:`arrow-right;1em;sd-text-info` `Configure pass <configure-workflows/pass-configuration.html>`_

   .. grid-item-card::

      **How to configure data**

      Learn how to configure data such as pre and post processing instructions.

      :octicon:`arrow-right;1em;sd-text-info` `Configure data <configure-workflows/how-to-configure-data.html>`_

   .. grid-item-card::

      **How to configure metrics**

      Learn how to configure metrics such as accuracy, latency, throughput, and your own custom metrics.

      :octicon:`arrow-right;1em;sd-text-info` `Configure metrics <configure-workflows/metrics-configuration.html>`_

   .. grid-item-card::

      **How to package models**

      Learn how to package models for deployment.

      :octicon:`arrow-right;1em;sd-text-info` `Model packaging <configure-workflows/model-packaging.html>`_

   .. grid-item-card::

      **How to configure systems**

      Learn how to configure systems such as local compute and remote compute to be a *host* (machine that executes optimization) and/or a *target* (machine that model will inference on).

      :octicon:`arrow-right;1em;sd-text-info` `Configure systems <configure-workflows/systems.html>`_

   .. grid-item-card::

      **How to configure engine**

      Learn how to configure engine to control where output gets saved, logging verbosity, caching, and packaging.

      :octicon:`arrow-right;1em;sd-text-info` `Configure engine <configure-workflows/engine-configuration.html>`_
