# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from olive.platform_sdk.qualcomm.utils import input_list as input_list_utils

logger = logging.getLogger(__name__)


class FileListDataLoader(ABC):
    """Abstraction for logical "FileListDataLoader", it contains data path and related metadata."""

    def __init__(self, config: dict, batch_size: int = None):
        # TODO(anyone): try to add file_chunk_size to distinguish the concept of batch_size and file_chunk_size
        """:param config: data loader specific config."""
        self.config = config
        self.batch_size = batch_size
        self.tmp_dir = None
        self.data_dir, self.input_list, self.annotation = self.load_data()
        self.prepare_batches()

    @abstractmethod
    def load_data(self) -> Tuple[str, str, Any]:
        """Return path to data directory, input list file and loaded annotation.

        Derived class should override this method
        """
        raise NotImplementedError

    def prepare_batches(self):
        if self.batch_size is None:
            self.num_batches = 1
            return

        if self.tmp_dir is None:
            self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")  # pylint: disable=consider-using-with
        self.batch_dir = str(Path(self.tmp_dir.name) / "batch")

        batch_input_list = input_list_utils.resolve_input_list(
            self.batch_dir, self.input_list, self.tmp_dir.name, self.data_dir, "batch_input_list.txt"
        )

        self.batch_input_list_headers = []
        self.batch_input_list_contents = []
        with open(self.input_list) as f, open(batch_input_list) as f_batch:
            for line, line_batch in zip(f, f_batch):
                if line.startswith(("#", "%")):
                    self.batch_input_list_headers.append(line.strip())
                else:
                    self.batch_input_list_contents.append((line.strip(), line_batch.strip()))

        self.batch_input_list_metadata = []
        for line, line_batch in self.batch_input_list_contents:
            to_copy = []
            for input_str, input_batch in zip(line.split(), line_batch.split()):
                if ":=" in input_str:
                    to_copy.append((input_str.split(":=")[1], input_batch.split(":=")[1]))
                else:
                    to_copy.append((input_str, input_batch))
            self.batch_input_list_metadata.append((line_batch, to_copy))

        self.batches = []
        for i in range(0, len(self.batch_input_list_metadata), self.batch_size):
            self.batches.append(self.batch_input_list_metadata[i : i + self.batch_size])  # noqa: E203, RUF100
        self.num_batches = len(self.batches)

    def get_batch(self, batch_id):
        if batch_id >= self.num_batches:
            raise ValueError("batch_id should be less than {}".format(self.num_batches))

        if self.batch_size is None:
            return self.data_dir, self.input_list, self.annotation
        else:
            annotation = None
            if self.annotation is not None:
                annotation = self.annotation[
                    self.batch_size * batch_id : self.batch_size * (batch_id + 1)  # noqa: E203, RUF100
                ]
            batch = self.batches[batch_id]

            shutil.rmtree(self.batch_dir, ignore_errors=True)
            for _, to_copy in batch:
                for input_str, input_batch in to_copy:
                    Path(input_batch).parent.mkdir(parents=True, exist_ok=True)
                    # Path(input_batch).symlink_to(input)
                    shutil.copy(input_str, input_batch)

            batch_input_list = Path(self.batch_dir) / "input_list.txt"
            with batch_input_list.open("w") as f:
                for header in self.batch_input_list_headers:
                    f.write(f"{header}\n")
                for line_batch, _ in batch:
                    f.write(f"{line_batch}\n")

            return self.batch_dir, str(batch_input_list), annotation

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # pylint: disable=attribute-defined-outside-init
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batches:
            batch = self.get_batch(self.n)
            self.n += 1
            return batch
        else:
            raise StopIteration

    def get_data_dir(self):
        return self.data_dir

    def get_input_list(self):
        return self.input_list


class FileListProcessedDataLoader(FileListDataLoader):
    def __init__(
        self,
        data_dir: str,
        input_list_file: str = "input_list.txt",
        annotations_file: str = None,
        batch_size: int = None,
    ):
        config = {"data_dir": data_dir, "input_list_file": input_list_file, "annotations_file": annotations_file}
        super().__init__(config, batch_size)

    def load_data(self) -> Tuple[str, str, np.ndarray]:
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")  # pylint: disable=consider-using-with
        input_list = input_list_utils.get_input_list(
            self.config["data_dir"], self.config["input_list_file"], self.tmp_dir.name
        )

        annotations = None
        if self.config["annotations_file"] is not None:
            annotations_path = Path(self.config["data_dir"]) / self.config["annotations_file"]
            if not annotations_path.is_file():
                raise FileNotFoundError(
                    f"{self.config['annotations_file']} not found in data directory {self.config['data_dir']}"
                )
            if annotations_path.suffix == ".npy":
                annotations = np.load(annotations_path)
            else:
                raise ValueError(f"Unsupported annotations file format {annotations_path.suffix}")

        return self.config["data_dir"], input_list, annotations


class FileListRandomDataLoader(FileListDataLoader):
    def __init__(
        self,
        io_config: dict,
        num_samples: int,
        data_dir: str = None,
        input_list_file: str = "input_list.txt",
        append_0: bool = False,
        batch_size: int = None,
    ):
        config = {
            "io_config": io_config,
            "num_samples": num_samples,
            "data_dir": data_dir,
            "input_list_file": input_list_file,
            "append_0": append_0,
        }
        super().__init__(config, batch_size)

    def load_data(self) -> Tuple[str, str, np.ndarray]:
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")  # pylint: disable=consider-using-with

        # get data_dir
        if self.config["data_dir"] is None:
            data_dir = Path(self.tmp_dir.name).resolve() / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
        else:
            data_dir = Path(self.config["data_dir"]).resolve()

        # create random data
        for input_name, input_shape in zip(
            self.config["io_config"]["input_names"], self.config["io_config"]["input_shapes"]
        ):
            input_dir = data_dir / input_name
            input_dir.mkdir(parents=True, exist_ok=True)
            raw_shape = np.product(input_shape)  # noqa: NPY003
            name_length = len(str(self.config["num_samples"]))
            for i in range(self.config["num_samples"]):
                input_file = input_dir / (f"{i}".zfill(name_length) + ".raw")
                raw_input = np.random.uniform(-1.0, 1.0, raw_shape).astype(np.float32)
                raw_input.tofile(input_file)

        # create input_list
        input_list_utils.create_input_list(
            str(data_dir),
            self.config["io_config"]["input_names"],
            input_list_file=str((data_dir / self.config["input_list_file"]).resolve()),
            add_input_names=len(self.config["io_config"]["input_names"]) > 1,
            add_output_names=len(self.config["io_config"]["output_names"]) > 1,
            output_names=self.config["io_config"]["output_names"],
            append_0=self.config["append_0"],
        )

        input_list = input_list_utils.get_input_list(data_dir, self.config["input_list_file"], self.tmp_dir.name)

        return str(data_dir), input_list, None


class FileListCommonDataLoader(FileListDataLoader):
    """FileList dataloader created from a common dataloader such as torch.data.DataLoader."""

    def __init__(self, dataloader: Any, io_config: dict, batch_size: int = None):
        """Initialize FileList Dataloader.

        :param dataloader: dataloader object. Dataloader must be iterable and return a tuple of (input, target).
        input is a dictionary of input names and input tensors.
        :param io_config: dictionary containing input and output names and shapes of the model.
        :param batch_size: batch size for the FileList dataloader. This is not the same as the batch size of the common
        dataloader.
        """
        config = {"dataloader": dataloader, "io_config": io_config}
        super().__init__(config, batch_size)

    def load_data(self) -> Tuple[str, str, np.ndarray]:
        logger.debug("Converting dataloader of type %s to FileList dataloader", type(self.config["dataloader"]))
        input_specs = {}
        for input_name, input_shape in zip(
            self.config["io_config"]["input_names"], self.config["io_config"]["input_shapes"]
        ):
            input_specs[input_name] = {"target_shape": input_shape}

        # get single data sample
        input_data, _ = next(iter(self.config["dataloader"]))
        # source input names
        for input_name, input_spec in input_specs.items():
            if input_name in input_data:
                source_name = input_name
            elif input_name.strip(":0") in input_data:
                source_name = input_name.strip(":0")
            else:
                raise ValueError(f"Input name {input_name} not found in dataset")
            input_spec["source_name"] = source_name

        # source input_shapes and permutations
        for input_spec in input_specs.values():
            # get source shape
            source_shape = list(input_data[input_spec["source_name"]].shape)
            input_spec["source_shape"] = source_shape

            # get permutation from source shape to target shape
            target_shape = input_spec["target_shape"]
            assert len(source_shape) == len(
                target_shape
            ), f"Source shape {source_shape} and target shape {target_shape} must have the same length"

            # find the permutation of the source shape that matches the target shape
            # e.g. source_shape = [1, 3, 224, 224], target_shape = [1, 224, 224, 3]
            #      -> permutation = [0, 2, 3, 1]
            # NCDHW -> NDHWC, NCHW -> NHWC, NFC -> NCF
            channel_permutation = [0, *list(range(2, len(source_shape))), 1]
            # NTF -> TNF
            # TODO(jambayk): confirm if it is NTF -> TNF or TNF -> NTF. Doesn't really matter since the first two
            # dimensions are transposed anyway
            time_permutation = [1, 0, *list(range(2, len(source_shape)))]
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

            input_spec["permutation"] = permutation
        logger.debug("Input specs: %s", input_specs)

        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")  # pylint: disable=consider-using-with
        data_dir = Path(self.tmp_dir.name) / "data"
        data_dir.mkdir()  # create data dir

        input_order = []
        annotations = []
        num_samples = len(self.config["dataloader"])
        sample_digits = len(str(num_samples))
        for i, (input_data_i, annotation) in enumerate(self.config["dataloader"]):
            if isinstance(input_data_i, tuple):
                input_data = dict(zip(input_specs.keys(), input_data_i))
            elif isinstance(input_data_i, (torch.Tensor, np.ndarray)):
                input_data = dict(zip(input_specs.keys(), [input_data_i]))
            else:
                input_data = input_data_i
                assert isinstance(
                    input_data, dict
                ), f"Input data must be a tuple, torch.Tensor, np.ndarray, or dict. Got {type(input_data)}"

            input_file_name = f"{i}.bin".zfill(sample_digits + 4)
            input_order.append(input_file_name)
            for input_spec in input_specs.values():
                data = input_data[input_spec["source_name"]]
                # FileList data loader only supports float32
                data = np.array(data, dtype=np.float32)

                # permute if necessary
                if input_spec["permutation"] is not None:
                    data = np.transpose(data, input_spec["permutation"])

                # save input data
                input_dir_path = data_dir / input_spec["source_name"]
                input_dir_path.mkdir(exist_ok=True)
                input_file_path = input_dir_path / input_file_name
                data.tofile(input_file_path)

            annotations.append(annotation.tolist())

        annotations = None if annotations[0] is None else np.array(annotations)

        # create input_list
        input_list_file = input_list_utils.create_input_list(
            data_dir=str(data_dir),
            input_names=list(input_specs.keys()),
            input_dirs=[input_spec["source_name"] for input_spec in input_specs.values()],
            add_input_names=len(input_specs) > 1,
            add_output_names=len(self.config["io_config"]["output_names"]) > 1,
            output_names=self.config["io_config"]["output_names"],
        )

        input_list = input_list_utils.get_input_list(str(data_dir), input_list_file, self.tmp_dir.name)

        return str(data_dir), input_list, annotations
