# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from olive.cli.launcher import legacy_call

    legacy_call("olive.workflows.run", "run", *sys.argv[1:])
