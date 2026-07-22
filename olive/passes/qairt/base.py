# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Base class for all QAIRT Olive passes."""

import logging
from abc import abstractmethod
from typing import Optional

from olive.model.handler.base import OliveModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig
from olive.passes.qairt.run_metadata import append_pass_entry, load_metadata, write_metadata

logger = logging.getLogger(__name__)


class QairtPass(Pass):
    """Base class for QAIRT passes.

    Wraps _run_qairt_pass() with automatic olive_run_metadata.json accumulation.
    Subclasses implement _run_qairt_pass() instead of _run_for_config().
    """

    def _get_recipe_path(self, config: type[BasePassConfig]) -> Optional[str]:
        """Return the resolved path to the Olive recipe .json for this pass, if applicable.

        Override in subclasses that accept a recipe parameter (e.g. QairtPipelinePass).
        The base implementation returns None, meaning no recipe_metadata will be seeded
        from a recipe file on this pass's invocation.
        """
        return None

    @abstractmethod
    def _run_qairt_pass(
        self,
        model: OliveModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> OliveModelHandler:
        """Run the pass-specific logic. Replaces _run_for_config in subclasses."""

    def _run_for_config(
        self,
        model: OliveModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> OliveModelHandler:
        output_model = self._run_qairt_pass(model, config, output_model_path)

        metadata = load_metadata(model.model_attributes)
        append_pass_entry(
            metadata,
            self.name,
            self.__class__.__name__,
            recipe_path=self._get_recipe_path(config),
        )
        attrs = dict(output_model.model_attributes or {})
        write_metadata(metadata, output_model_path, attrs)
        output_model.model_attributes = attrs

        return output_model
