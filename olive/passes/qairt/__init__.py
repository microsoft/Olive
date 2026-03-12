# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

try:
    import qairt
    import qairt.gen_ai_api as qairt_genai
except ImportError as exc:
    raise ImportError(
        "Failed to import QAIRT GenAIBuilder API - please install olive-ai[qairt] to use QAIRT passes."
        "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
    ) from exc