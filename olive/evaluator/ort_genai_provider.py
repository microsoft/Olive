# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
from typing import Any

logger = logging.getLogger(__name__)

_REGISTERED_PROVIDER_LIBRARIES = set()


def _get_windows_ml_api():
    import winui3.microsoft.windows.ai.machinelearning as winml
    from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
        InitializeOptions,
        initialize,
    )

    return winml, InitializeOptions, initialize


def _configured_provider_names(config: dict[str, Any]) -> set[str]:
    """Collect execution-provider names from GenAI session options."""
    providers = set()

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "provider_options" and isinstance(child, list):
                    for provider_options in child:
                        if isinstance(provider_options, dict):
                            providers.update(provider_options)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    return providers


def register_configured_execution_provider_libraries(og, config: dict[str, Any]) -> None:
    """Register configured GenAI providers exposed by the Windows ML catalog.

    GenAI's built-in providers require no registration. ABI providers installed
    through Windows ML expose a library path through ExecutionProviderCatalog.
    """
    provider_names = _configured_provider_names(config)
    if not provider_names or platform.system() != "Windows":
        return

    try:
        winml, initialize_options, initialize = _get_windows_ml_api()
    except ImportError:
        logger.debug("Windows ML provider discovery is unavailable.")
        return

    with initialize(options=initialize_options.ON_NO_MATCH_SHOW_UI):
        catalog_providers = {
            provider.name.casefold(): provider
            for provider in winml.ExecutionProviderCatalog.get_default().find_all_providers()
        }

        for configured_name in sorted(provider_names):
            candidates = (configured_name, f"{configured_name}ExecutionProvider")
            provider = next(
                (catalog_providers[name.casefold()] for name in candidates if name.casefold() in catalog_providers),
                None,
            )
            if provider is None or provider.name in _REGISTERED_PROVIDER_LIBRARIES:
                continue

            result = provider.ensure_ready_async().get()
            if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
                raise RuntimeError(
                    f"Windows ML execution provider {provider.name} is unavailable: "
                    f"status={result.status}, reason={result.diagnostic_text}, "
                    f"error_code={result.extended_error.value}"
                )

            og.register_execution_provider_library(provider.name, str(provider.library_path))
            _REGISTERED_PROVIDER_LIBRARIES.add(provider.name)
            logger.info("Registered GenAI execution provider %s from %s", provider.name, provider.library_path)
