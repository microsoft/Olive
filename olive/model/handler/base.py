import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Union

from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.mixin import CompositeMixin, IoConfigMixin, JsonMixin, ResourceMixin
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


class OliveModelHandler(ABC, ResourceMixin, IoConfigMixin, JsonMixin, CompositeMixin):
    """Abstraction for logical "Model", it contains model path and related metadata.

    Each technique accepts Model as input, return Model as output.
    """

    resource_keys: ClassVar[list] = ["model_path"]

    def __init__(
        self,
        framework: Framework,
        model_file_format: ModelFileFormat,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.framework = framework
        self.model_file_format = model_file_format
        self.composite_parent = None
        self.model_attributes = model_attributes
        self.io_config = None

        # store resource paths
        self.resource_paths: Dict[str, str] = {}
        resources = {}
        resources["model_path"] = model_path
        self.add_resources(resources)

    @property
    def model_path(self) -> str:
        """Return local model path."""
        return self.get_resource("model_path")

    @abstractmethod
    def load_model(self, rank: int = None) -> object:
        """Load model from disk, return in-memory model object.

        Derived class should implement its specific logic if needed.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        """Prepare inference session for Olive model, return in-memory inference session.

        Derived class should implement its specific logic if needed.
        """
        raise NotImplementedError
