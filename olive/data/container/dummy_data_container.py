# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import ClassVar

from olive.data.constants import DataComponentType, DataContainerType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry

TRANSFORMER_DUMMY_DATA_CONTAINER = (
    "TransformersDummyDataContainer",
    "TransformersPromptDummyDataContainer",
    "TransformersTokenDummyDataContainer",
)


@Registry.register(DataContainerType.DATA_CONTAINER)
class DummyDataContainer(DataContainer):
    """Dummy data container.

    The way to create a dummy data container:
        dummy_data_config = DataConfig(
            name="dummy",
            type="DummyDataContainer",
            load_dataset_config=DataComponentConfig(
                params={
                    "input_names": metric.user_config.input_names,
                    "input_shapes": metric.user_config.input_shapes,
                    "input_types": metric.user_config.input_types,
                }
            ),
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "dummy_dataset",
        DataComponentType.DATALOADER.value: "no_auto_batch_dataloader",
    }


@Registry.register(DataContainerType.DATA_CONTAINER)
class TransformersDummyDataContainer(DummyDataContainer):
    """Dummy data container for transformer model.

    The way to create a dummy data container for transformer model:
        dummy_data_config = DataConfig(
            name="dummy",
            type="TransformersDummyDataContainer",
            "load_dataset_config"={
                "params": {
                    "seq_len": 128,
                    "past_seq_len": 128,
                    "max_seq_len": 1024,
                    "model_framework": Framework.ONNX,
                    "use_fp16": False,
                    "shared_kv": False,
                    "generative": False,
                    "ort_past_key_name":"past_key_values.<id>.key",
                    "ort_past_value_name":"past_key_values.<id>.value",
                    "trust_remote_code": None,
                    "max_samples": 32,
                )
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "transformers_dummy_dataset",
    }


@Registry.register(DataContainerType.DATA_CONTAINER)
class TransformersPromptDummyDataContainer(DummyDataContainer):
    """Dummy data container for transformer model.

    The way to create a dummy data container for transformer model:
        dummy_data_config = DataConfig(
            name="dummy",
            type="TransformersPromptDummyDataContainer",
            "load_dataset_config"={
                "params": {
                    "seq_len": 8,
                    "past_seq_len": 0,
                    "max_seq_len": 2048,
                    "model_framework": Framework.ONNX,
                    "use_fp16": False,
                    "shared_kv": False,
                    "generative": False,
                    "ort_past_key_name":"past_key_values.<id>.key",
                    "ort_past_value_name":"past_key_values.<id>.value",
                    "trust_remote_code": None,
                    "max_samples": 32,
                )
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "transformers_prompt_dummy_dataset",
    }


@Registry.register(DataContainerType.DATA_CONTAINER)
class TransformersTokenDummyDataContainer(DummyDataContainer):
    """Dummy data container for transformer model.

    The way to create a dummy data container for transformer model:
        dummy_data_config = DataConfig(
            name="dummy",
            type="TransformersTokenDummyDataContainer",
            "load_dataset_config"={
                "params": {
                    "seq_len": 1,
                    "past_seq_len": 8,
                    "max_seq_len": 2048,
                    "model_framework": Framework.ONNX,
                    "use_fp16": False,
                    "shared_kv": False,
                    "generative": False,
                    "ort_past_key_name":"past_key_values.<id>.key",
                    "ort_past_value_name":"past_key_values.<id>.value",
                    "trust_remote_code": None,
                    "max_samples": 32,
                )
            },
            "dataloader_config"={
                "params": {
                    "batch_size": 1,
                    "fields_no_batch": "step",
                }
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "transformers_token_dummy_dataset",
        DataComponentType.DATALOADER.value: "dataloader_with_ignored_batch_fields",
    }
