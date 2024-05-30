.. _command_line_tools:

Command Line Tools
==================

Olive provides command line tools that can be invoked using the ``olive`` command. The command line tools are used to perform various tasks such as running an Olive workflow, managing AzureML compute, and more.

If ``olive`` is not in your PATH, you can run the command line tools by replacing ``olive`` with ``python -m olive``.

.. argparse::
    :module: olive.cli.launcher
    :func: get_cli_parser
    :prog: olive
