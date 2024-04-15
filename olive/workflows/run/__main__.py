# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # there is no circular dependency since run is imported lazily by the command runner
    from olive.cli.launcher import legacy_call

    legacy_call("olive.workflows.run", "run")
