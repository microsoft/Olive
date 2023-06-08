# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import ClassVar

from olive.data.constants import DataComponentType, DataContainerType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class RawDataContainer(DataContainer):
    """
    The way to create a raw data container:
        raw_data_config = DataConfig(
            name="raw",
            type="RawDataContainer",
            params_config={
                "data_dir": data_dir,
                "input_names": input_names,
                "input_shapes": input_shapes,
                "input_types": input_types, # optional
                "input_dirs": input_dirs, # optional
                "input_suffix": input_suffix, # optional
                "input_order_file": input_order_file, # optional
                "annotations_file": annotations_file, # optional
            }
        )
    """

    default_components_type: ClassVar[dict] = {DataComponentType.LOAD_DATASET.value: "raw_dataset"}
