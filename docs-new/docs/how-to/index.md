---
hide:
  - toc
---

# How-to guides

## Set-up 

<div class="grid cards" markdown>

-   :octicons-download-24:{ .lg .middle } __How to install Olive__

    ---

    Learn how to install `olive-ai`.

    [:octicons-arrow-right-24: Install Olive](installation.md)

</div>

## Working with the CLI

The Olive CLI provides a set of primitives such as `quantize`, `finetune`, `onnx-graph-capture`, `auto-opt` that enable you to *easily* optimize models and experiment with different cutting-edge optimization strategies without the need to define workflows.

!!! tip
    For users new to Olive, we recommend that you start with the CLI.

<div class="grid cards" markdown>

-   :material-auto-fix:{ .lg .middle } __Auto Optimizer__

    ---

    Learn how to use the `olive auto-opt` command to take a PyTorch/Hugging Face model and turn it into an optimized ONNX model.

    [:octicons-arrow-right-24: `olive auto-opt`](cli/cli-auto-opt.md)

-   :material-tune:{ .lg .middle } __Finetune__

    ---

    Learn how to use the `olive finetune` command to create (Q)LoRA adapters.

    [:octicons-arrow-right-24: `olive finetune`](cli/cli-finetune.md)


-   :fontawesome-solid-compress:{ .lg .middle } __Quantize__

    ---

    Learn how to use the `olive quantize` command to quantize your model with different precisions and techniques such as AWQ.

    [:octicons-arrow-right-24: `olive quantize`](cli/cli-quantize.md)

-   :octicons-workflow-24:{ .lg .middle } __Execute Olive Workflows__

    ---

    Learn how to use the `olive run` command to execute an Olive workflow.

    [:octicons-arrow-right-24: `olive run`](cli/cli-run.md)

</div>


## Configure Workflows (Advanced)

For more complex scenarios, you can create fully customize workflows where you can run any of the 40+ supported optimization techniques in a sequence.

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } __How to configure data__

    ---

    Learn how to configure data such as pre and post processing instructions

    [:octicons-arrow-right-24: Configure data](configure-workflows/how-to-configure-data.md)

-   :material-sort-numeric-ascending:{ .lg .middle } __How to configure metrics__

    ---

    Learn how to configure metrics such as accuracy, latency, throughput, and your own custom metrics.

    [:octicons-arrow-right-24: Configure metrics](configure-workflows/metrics-configuration.md)

-   :octicons-package-24:{ .lg .middle } __How to package models__

    ---

    Learn how to package models for deployment.

    [:octicons-arrow-right-24: Model packaging](configure-workflows/model-packaging.md)

-   :fontawesome-solid-computer:{ .lg .middle } __How to configure systems__

    ---

    Learn how to configure systems such as local compute and remote compute to be a *host* (machine that executes optimization) and/or a *target* (machine that model will inference on).

    [:octicons-arrow-right-24: Configure systems](configure-workflows/systems.md)

-   :material-auto-fix:{ .lg .middle } __How to use Auto Optimizer__

    ---

    Learn how to use Auto Optimizer - a tool that automatically creates the best model for you - in a workflow.

    [:octicons-arrow-right-24: Auto Optimizer](configure-workflows/auto-opt.md)

<div class="grid cards" markdown>

-   :material-microsoft-azure:{ .lg .middle } __How to integrate with Azure AI__

    ---

    Learn how to use integrations with Azure AI, such as model catalog, remote compute and data/job artefacts.

    [:octicons-arrow-right-24: Integrate with Azure AI](configure-workflows/azure-ai.md)

-   :simple-huggingface:{ .lg .middle } __How to integrate with Hugging Face__

    ---

    Learn how to use integrations with Hugging Face, such as models, data and metrics.

    [:octicons-arrow-right-24: Integrate with Hugging Face](configure-workflows/azure-ai.md)

</div>
