# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This is to support running Olive CLI as a module in case olive-cli command
# is not available in the PATH.
# Example: python -m olive.cli
if __name__ == "__main__":
    from olive.cli.olive_cli import main

    main(called_as_console_script=False)
