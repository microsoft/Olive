# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
REGISTRY = {}


def model_handler_registry(model_type):
    """Decorate and register all OliveModelHandler subclasses.

    Args:
        model_type (str): The model type registration name. Is case-insensitive and stored in lowercase.

    Returns:
        cls: The class of register.

    """
    model_type = model_type.lower()

    def decorator_model_class(cls):
        if model_type in REGISTRY:
            raise ValueError("Cannot have two model handlers with the same name")

        REGISTRY[model_type] = cls
        cls.model_type = model_type
        return cls

    return decorator_model_class


def get_model_handler(model_type):
    if not is_valid_model_type(model_type):
        raise ValueError(f"Unknown model type {model_type}")
    return REGISTRY[model_type.lower()]


def is_valid_model_type(model_type):
    return model_type.lower() in REGISTRY
