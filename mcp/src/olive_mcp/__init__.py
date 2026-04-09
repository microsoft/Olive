# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import olive_mcp.tools  # noqa: F401 — registers @mcp.tool() and @mcp.prompt() on import
from olive_mcp.server import mcp


def main():
    mcp.run()
