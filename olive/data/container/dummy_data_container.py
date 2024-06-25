# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import ClassVar

from olive.data.constants import DataComponentType, DataContainerType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class DummyDataContainer(DataContainer):
    """Dummy data container.

    The way to create a dummy data container:
        dummy_data_config = DataConfig(
            name="dummy",
            type="DummyDataContainer",
            "load_dataset_config"={
                "params": {
                    "input_names": metric.user_config.input_names,
                    "input_shapes": metric.user_config.input_shapes,
                    "input_types": metric.user_config.input_types,
                }
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "dummy_dataset",
        DataComponentType.DATALOADER.value: "skip_dataloader",
    }


@Registry.register(DataContainerType.DATA_CONTAINER)
class TransformerDummyDataContainer(DummyDataContainer):
    """Dummy data container for transformer model.

    The way to create a dummy data container for transformer model:
        dummy_data_config = DataConfig(
            name="dummy",
            type="TransformerDummyDataContainer",
            "load_dataset_config"={
                "params": {
                    "batch_size": 1,
                    "seq_len": 128,
                    "past_seq_len": 128,
                    "max_seq_len": 1024,
                    "model_framework": Framework.ONNX,
                    "use_fp16": False,
                    "shared_kv": False,
                    "generative": False,
                    "ort_past_key_name":"past_key_values.<id>.key",
                    "ort_past_value_name":"past_key_values.<id>.value",
                )
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "transformers_dummy_dataset",
        DataComponentType.DATALOADER.value: "no_auto_batch_dataloader",
    }
