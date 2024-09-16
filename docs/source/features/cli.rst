.. _command_line_tools:

Command Line Tools
==================

Olive provides command line tools that can be invoked using the ``olive`` command. |
The command line tools are used to perform various tasks such as running an Olive workflow, |
managing AzureML compute, and more.

If ``olive`` is not in your PATH, you can run the command line tools by replacing ``olive`` with ``python -m olive``.

Input Model
-----------

Olive Cli Procuded Model
^^^^^^^^^^^^^^^^^^^

The Olive command-line tools support using a model produced by Olive CLI as an input model. You can specify the model file path using the ``-m <output_model>`` option, where ``<output_model>`` is the output folder defined by ``-o <output_model>`` in the previous cli command.

Local PyTorch Model
^^^^^^^^^^^^^^^^^^^

Olive command line tools accept a local PyTorch model as an input model. You can specify the model file path using the ``-m model.pt`` option, and the associated model script using the ``--model_script script.py`` option.

Olive reserves several function names to provide specific inputs for the PyTorch model. These functions should be defined in your model script:

Available Functions
-------------------

Below are the functions that Olive expects in the model script and their purposes:

- **Model Loader Function (`_model_loader`)**:
  Loads the PyTorch model. If the model file path is provided using the `-m` option, it takes higher priority than the model loader function.

  .. code-block:: python

      def _model_loader():
          ...
          return model

- **IO Config Function (`_io_config`)**:
  Returns the IO configuration for the model. Either `_io_config` or `_dummy_inputs` is required for the `capture-onnx-graph` CLI command.

  .. code-block:: python

      def _io_config(model: PyTorchModelHandler):
          ...
          return io_config

- **Dummy Inputs Function (`_dummy_inputs`)**:
  Provides dummy input tensors for the model. Either `_io_config` or `_dummy_inputs` is required for the `capture-onnx-graph` CLI command.

  .. code-block:: python

      def _dummy_inputs(model: PyTorchModelHandler):
          ...
          return dummy_inputs

- **Model Format Function (`_model_file_format`)**:
  Specifies the format of the model. The default value is `PyTorch.EntireModel`. For more available options, refer to `this <https://github.com/microsoft/Olive/blob/main/olive/constants.py#L23-L26>`_.

  .. code-block:: python

      def _model_file_format():
          ...
          return model_file_format

Example Usage
-------------

To use the Olive CLI with a local PyTorch model:

1. Provide the model path and the script:

   .. code-block:: bash

      python -m olive capture-onnx-graph -m model.pt --model_script script.py

2. Ensure that the script contains the above functions to handle loading, input/output configuration, dummy inputs, and model format specification as needed.


Argparse Documentation
----------------------

Below is the argparse documentation for the Olive command-line interface:

.. argparse::
    :module: olive.cli.launcher
    :func: get_cli_parser
    :prog: olive
