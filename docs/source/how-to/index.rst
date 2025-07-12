How-to
=======
Find more details on specific Olive capabilities, such as quantization, running workflows on remote compute, model packaging, conversions, and more!

Installation and Setup
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

Olive Command Lines
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

      How to use the `olive auto-opt` command to take a PyTorch/Hugging Face model and turn it into an optimized ONNX model

      :octicon:`arrow-right;1em;sd-text-info` `olive auto-opt <cli/cli-auto-opt.html>`_

   .. grid-item-card::

      **Finetune**

      how to use the `olive finetune` command to create (Q)LoRA adapters

      :octicon:`arrow-right;1em;sd-text-info` `olive finetune <cli/cli-finetune.html>`_

   .. grid-item-card::

      **Quantize**

      How to use the `olive quantize` command to quantize your model with different precisions and techniques such as AWQ

      :octicon:`arrow-right;1em;sd-text-info` `olive quantize <cli/cli-quantize.html>`_

   .. grid-item-card::

      **Execute Olive Workflows**

      How to use the `olive run` command to execute an Olive workflow

      :octicon:`arrow-right;1em;sd-text-info` `olive run <cli/cli-run.html>`_


Olive Python API
----------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   python_api

.. grid:: 1 1 1 1
   :class-container: cards

   .. grid-item-card::
      **Python API**

      How to use the Python API to run Olive workflows programmatically

      :octicon:`arrow-right;1em;sd-text-info` `Python API <python_api.html>`_

Customize Workflow (aka Recipes)
------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   configure-workflows/build-workflow
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

      **How to write a new .json configuration**

      How to write a new workflow from scratch

      :octicon:`arrow-right;1em;sd-text-info` `Configure workflow <configure-workflows/build-workflow.html>`_

   .. grid-item-card::

      **How to configure model**

      How to define input model for a new workflow

      :octicon:`arrow-right;1em;sd-text-info` `Configure models <configure-workflows/how-to-configure-model.html>`_

   .. grid-item-card::

      **How to configure passes**

      How to customize a pass parameters

      :octicon:`arrow-right;1em;sd-text-info` `Configure pass <configure-workflows/pass-configuration.html>`_

   .. grid-item-card::

      **How to configure data**

      How to setup custom dataset for calibration and evaluation

      :octicon:`arrow-right;1em;sd-text-info` `Configure data <configure-workflows/how-to-configure-data.html>`_

   .. grid-item-card::

      **How to configure metrics**

      How to define evaluation metrics such as accuracy, latency, throughput, and your own custom metrics

      :octicon:`arrow-right;1em;sd-text-info` `Configure metrics <configure-workflows/metrics-configuration.html>`_

   .. grid-item-card::

      **How to package models**

      How to package output model for deployment

      :octicon:`arrow-right;1em;sd-text-info` `Model packaging <configure-workflows/model-packaging.html>`_

   .. grid-item-card::

      **How to configure systems**

      How to define `host` or `target` systems

      :octicon:`arrow-right;1em;sd-text-info` `Configure systems <configure-workflows/systems.html>`_

   .. grid-item-card::

      **How to configure engine**

      How to configure Olive `engine`

      :octicon:`arrow-right;1em;sd-text-info` `Configure engine <configure-workflows/engine-configuration.html>`_

