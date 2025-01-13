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

The Olive CLI provides a set of primitives such as `quantize`, `finetune`, `onnx-graph-capture`, `auto-opt` that enable you to *easily* optimize models and experiment with different cutting-edge optimization strategies without the need to define workflows.

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

Configure Workflows (Advanced)
------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   configure-workflows/pass/pass-configuration
   configure-workflows/model-opt-and-transform/index
   configure-workflows/how-to-configure-data
   configure-workflows/metrics-configuration
   configure-workflows/model-packaging
   configure-workflows/systems
   configure-workflows/auto-opt
   configure-workflows/model-splitting
   configure-workflows/onnx-graph-surgeon

For more complex scenarios, you can create fully customize workflows where you can run any of the 40+ supported optimization techniques in a sequence.

.. grid:: 2 2 2 3
   :class-container: cards

   .. grid-item-card::

      **How to configure passes**

      Learn how to configure passes.

      :octicon:`arrow-right;1em;sd-text-info` `Configure pass <configure-workflows/pass/pass-configuration.html>`_

   .. grid-item-card::

      **How to configure models**

      Learn how to configure and optimize models.

      :octicon:`arrow-right;1em;sd-text-info` `Configure models <configure-workflows/model-opt-and-transform/index.html>`_

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

      **How to use Auto Optimizer**

      Learn how to use Auto Optimizer - a tool that automatically creates the best model for you - in a workflow.

      :octicon:`arrow-right;1em;sd-text-info` `Auto Optimizer <configure-workflows/auto-opt.html>`_

   .. grid-item-card::

      **Model splitting**

      Learn how to split a model into multiple components.

      :octicon:`arrow-right;1em;sd-text-info` `Model splitting <configure-workflows/model-splitting.html>`_

   .. grid-item-card::

      **ONNX Graph Surgeon**

      Learn how to use ONNX Graph Surgeon to manipulate ONNX graphs.

      :octicon:`arrow-right;1em;sd-text-info` `ONNX Graph Surgeon <configure-workflows/onnx-graph-surgeon.html>`_

Integrations
------------

.. toctree::
   :maxdepth: 1
   :hidden:

   configure-workflows/azure-ai/index
   configure-workflows/huggingface-integration

.. grid:: 2 2 2 3
   :class-container: cards

   .. grid-item-card::

      **How to integrate with Azure AI**

      Learn how to use integrations with Azure AI, such as model catalog, remote compute, and data/job artifacts.

      :octicon:`arrow-right;1em;sd-text-info` `Integrate with Azure AI <configure-workflows/azure-ai/index.html>`_

   .. grid-item-card::

      **How to integrate with Hugging Face**

      Learn how to use integrations with Hugging Face, such as models, data, and metrics.

      :octicon:`arrow-right;1em;sd-text-info` `Integrate with Hugging Face <configure-workflows/huggingface-integration.html>`_
