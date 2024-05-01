# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# A separate __main__.py is implemented since CodeQL complains about circular imports otherwise.
if __name__ == "__main__":
    import sys

    from olive.cli.launcher import legacy_call

    legacy_call("olive.platform_sdk.qualcomm.configure", "configure-qualcomm-sdk", *sys.argv[1:])
