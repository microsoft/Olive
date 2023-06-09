# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import tempfile
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_dataloader()
def skip_dataloader(_dataset):
    return _dataset


@Registry.register_default_dataloader()
def default_dataloader(_dataset, batch_size=1, **kwargs):
    return DataLoader(_dataset, batch_size=batch_size, **kwargs)


@Registry.register_dataloader()
def no_auto_batch_dataloader(_dataset, **kwargs):
    # torch dataloader will automatically batch if batch_size is not None
    # this dataloader will not batch. Assumes that the dataset already returns a batch
    return DataLoader(_dataset, batch_size=None, **kwargs)


@Registry.register_dataloader()
def default_calibration_dataloader(_dataloader, **kwargs):
    # TODO: consider other quantization calibration interface in SNPE/INC/OpenVINO/Torch and etc.
    from onnxruntime.quantization import CalibrationDataReader

    dataloader = _dataloader

    class _CalibrationDataReader(CalibrationDataReader):
        def __init__(self, dataloader, **kwargs):
            self.dataloader = dataloader
            self.kwargs = kwargs
            self.data_iter = iter(self.dataloader)

        def get_next(self):
            if self.data_iter is None:
                self.data_iter = iter(self.dataloader)
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None
            if isinstance(batch, list):
                batch = batch[0]
            if isinstance(batch, dict):
                batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
            return batch

        def rewind(self):
            self.data_iter = None

    return _CalibrationDataReader(dataloader, **kwargs)


@Registry.register_dataloader()
def default_snpe_dataloader(_dataset, model_io_config):
    from olive.snpe.data_loader import SNPEProcessedDataLoader
    from olive.snpe.utils.input_list import create_input_list

    class _SNPEProcessedDataLoader(SNPEProcessedDataLoader):
        def __init__(self, dataset, model_io_config):
            logger.debug("Creating SNPEProcessedDataLoader from dataset")
            input_specs = {}
            for input_name, input_shape in zip(model_io_config["input_names"], model_io_config["input_shapes"]):
                input_specs[input_name] = {"target_shape": input_shape}

            # get single data sample
            input_data, _ = dataset[0]
            # source input names
            for input_name in input_specs:
                if input_name in input_data:
                    source_name = input_name
                elif input_name.strip(":0") in input_data:
                    source_name = input_name.strip(":0")
                else:
                    raise ValueError(f"Input name {input_name} not found in dataset")
                input_specs[input_name]["source_name"] = source_name

            # source input_shapes and permutations
            for input_name, input_spec in input_specs.items():
                # get source shape
                source_shape = list(input_data[input_spec["source_name"]].shape)
                input_specs[input_name]["source_shape"] = source_shape

                # get permutation from source shape to target shape
                target_shape = input_spec["target_shape"]
                assert len(source_shape) == len(
                    target_shape
                ), f"Source shape {source_shape} and target shape {target_shape} must have the same length"

                # find the permutation of the source shape that matches the target shape
                # e.g. source_shape = [1, 3, 224, 224], target_shape = [1, 224, 224, 3]
                #      -> permutation = [0, 2, 3, 1]
                # NCDHW -> NDHWC, NCHW -> NHWC, NFC -> NCF
                channel_permutation = [0] + list(range(2, len(source_shape))) + [1]
                # NTF -> TNF
                # TODO: confirm if it is NTF -> TNF or TNF -> NTF. Doesn't really matter since the first two
                # dimensions are transposed anyway
                time_permutation = [1, 0] + list(range(2, len(source_shape)))
                if source_shape == target_shape:
                    permutation = None  # no permutation needed
                elif target_shape == [source_shape[idx] for idx in channel_permutation]:
                    permutation = channel_permutation
                elif target_shape == [source_shape[idx] for idx in time_permutation]:
                    permutation = time_permutation
                else:
                    raise ValueError(
                        f"Cannot find a valid permutation of the source shape {source_shape} that matches the target"
                        f" shape {target_shape}"
                    )

                input_specs[input_name]["permutation"] = permutation
            logger.debug(f"Input specs: {input_specs}")

            self.tmp_main_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")
            data_dir = Path(self.tmp_main_dir.name) / "data"
            data_dir.mkdir()  # create data dir

            input_order = []
            annotations = []
            num_samples = len(dataset)
            sample_digits = len(str(num_samples))
            for i in range(num_samples):
                input_data, annotation = dataset[i]
                input_file_name = f"{i}.bin".zfill(sample_digits + 4)
                input_order.append(input_file_name)
                for input_name, input_spec in input_specs.items():
                    data = input_data[input_spec["source_name"]]
                    # snpe data loader only supports float32
                    data = np.array(data, dtype=np.float32)

                    # permute if necessary
                    if input_spec["permutation"] is not None:
                        data = np.transpose(data, input_spec["permutation"])

                    # save input data
                    input_dir_path = data_dir / input_spec["source_name"]
                    input_dir_path.mkdir(exist_ok=True)
                    input_file_path = input_dir_path / input_file_name
                    data.tofile(input_file_path)

                annotations.append(annotation)

            # save annotations if any
            annotations = None if annotations[0] is None else np.array(annotations)
            annotation_file = None
            if annotations is not None:
                annotation_file = str(data_dir / "annotations.npy")
                np.save(annotation_file, annotations)

            # create input list file
            input_list_file = create_input_list(
                data_dir=str(data_dir),
                input_names=list(input_specs.keys()),
                input_dirs=[input_specs[input_name]["source_name"] for input_name in input_specs.keys()],
                add_input_names=len(input_specs) > 1,
                add_output_names=len(model_io_config["output_names"]) > 1,
                output_names=model_io_config["output_names"],
            )

            super().__init__(data_dir=str(data_dir), input_list_file=input_list_file, annotations_file=annotation_file)

    return _SNPEProcessedDataLoader(_dataset, model_io_config)
