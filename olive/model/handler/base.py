import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from olive.common.config_utils import validate_config
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config import IoConfig
from olive.model.handler.mixin import IoConfigMixin, JsonMixin, ResourceMixin
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


class OliveModelHandler(ABC, ResourceMixin, IoConfigMixin, JsonMixin):
    """Abstraction for logical "Model", it contains model path and related metadata.

    Each technique accepts Model as input, return Model as output.
    The major responsibility of this base class is to provide a unified interface model loading
        * load_model: load model from disk, return in-memory model object.
            For PyTorch model, the in-memory model object is torch.nn.Module.
            For ONNX model, the in-memory model object is onnx.ModelProto.
        * prepare_session: prepare inference session for Olive model, return in-memory inference session.
            If the model is PyTorch model, it will return a torch.nn.Module object.
            If the model is ONNX model, it will return a onnxruntime.InferenceSession object.
            If the model is from Huggingface Optimum, it will return ORTModel object.

    If you add new Mixin into model handler, please make sure the member variable
    is initialized properly in model handler.
    """

    model_type: Optional[str] = None
    resource_keys: Tuple[str, ...] = ("model_path",)

    def __init__(
        self,
        framework: Framework,
        model_file_format: ModelFileFormat,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[Dict[str, Any]] = None,
        io_config: Union[Dict[str, Any], "IoConfig", str, Callable] = None,
        generative: bool = False,
    ):
        self.framework = framework
        self.model_file_format = model_file_format
        self.composite_parent = None
        self.model_attributes = model_attributes
        self._io_config = (
            validate_config(io_config, IoConfig).dict(exclude_none=True)
            if isinstance(io_config, (IoConfig, dict))
            else io_config
        )
        self.generative = generative

        # store resource paths
        self.resource_paths: Dict[str, str] = {}

        # Only update the resource_paths when the resource_key is model_path.
        # All other case will be handled by subclass.
        if len(self.resource_keys) == 1 and self.resource_keys[0] == "model_path":
            self.add_resources(locals())

    @property
    def model_path(self) -> str:
        """Return local model path."""
        return self.get_resource("model_path")

    @abstractmethod
    def load_model(self, rank: int = None, cache_model: bool = True) -> object:
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

    @abstractmethod
    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        """Run inference session for Olive model, returns in-memory inference results.

        Derived class should implement its specific logic if needed.
        """
        raise NotImplementedError
