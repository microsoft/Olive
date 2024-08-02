# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)


class OnnxEpValidateMixin:
    """Provide the ORT execution provider validation functionality for the model handler."""

    def is_valid_ep(self, filepath: str, ep: str = None):
        # TODO(shaahji): should be remove if future accelerators is implemented
        # It should be a bug for onnxruntime where the execution provider is not be fallback.
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        if self.use_ort_extensions:
            # register custom ops for onnxruntime_extensions
            from onnxruntime_extensions import get_library_path

            sess_options.register_custom_ops_library(get_library_path())

        try:
            ort.InferenceSession(filepath, sess_options, providers=[ep])
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Error: %s Olive will ignore this %(ep)s."
                "Please make sure the environment with %(ep)s has the required dependencies.",
                e,
                ep=ep,
            )
            return False
        return True
