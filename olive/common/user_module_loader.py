# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from types import FunctionType, MethodType
from typing import Any, Callable, Union

from olive.common.import_lib import import_user_module
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, create_resource_path, get_local_path


class UserModuleLoader:
    """
    Load user module and call object in it.

    Only used for objects that are not json serializable.
    """

    def __init__(self, user_script: OLIVE_RESOURCE_ANNOTATIONS, script_dir: OLIVE_RESOURCE_ANNOTATIONS = None):
        self.user_script = get_local_path(create_resource_path(user_script))
        self.script_dir = get_local_path(create_resource_path(script_dir))
        if self.user_script:
            self.user_module = import_user_module(self.user_script, self.script_dir)
        else:
            self.user_module = None

    def call_object(self, obj: Union[str, Callable, Any], *args, **kwargs):
        """
        Call obj with given arguments if it is a function, otherwise just return the object
        """

        obj = self.load_object(obj)
        # We check for FunctionType, MethodType here instead of Callable since objects with __call__ methods
        # like torch.nn.module are also Callables
        if isinstance(obj, FunctionType) or isinstance(obj, MethodType):
            return obj(*args, **kwargs)
        return obj

    def load_object(self, obj: Union[str, Callable, Any]):
        """
        Get obj from user_module if it is string name else return obj
        """

        if isinstance(obj, str):
            assert self.user_module is not None, "There is no user module to load object from."
            return getattr(self.user_module, obj)
        return obj
