# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import numpy as np

import olive.snpe.utils.input_list as input_list_utils


class SNPEDataLoader(ABC):
    """
    Abstraction for logical "SNPEDataLoader", it contains data path and related metadata.
    """

    def __init__(self, config: dict, batch_size: int = None):
        """
        :param config: data loader specific config
        """
        self.config = config
        self.batch_size = batch_size
        self.tmp_dir = None
        self.data_dir, self.input_list, self.annotation = self.load_data()
        self.prepare_batches()

    @abstractmethod
    def load_data(self) -> Tuple[str, str, Any]:
        """
        Return path to data directory, input list file and loaded annotation
        Derived class should override this method
        """
        raise NotImplementedError()

    def prepare_batches(self):
        if self.batch_size is None:
            self.num_batches = 1
            return

        if self.tmp_dir is None:
            self.tmp_dir = tempfile.TemporaryDirectory(dir=str(Path.cwd()), prefix="olive_tmp_")
        self.batch_dir = str(Path(self.tmp_dir.name) / "batch")

        batch_input_list = input_list_utils.resolve_input_list(
            self.batch_dir, self.input_list, self.tmp_dir.name, self.data_dir, "batch_input_list.txt"
        )

        self.batch_input_list_headers = []
        self.batch_input_list_contents = []
        with open(self.input_list, "r") as f, open(batch_input_list, "r") as f_batch:
            for line, line_batch in zip(f, f_batch):
                if line.startswith("#") or line.startswith("%"):
                    self.batch_input_list_headers.append(line.strip())
                else:
                    self.batch_input_list_contents.append((line.strip(), line_batch.strip()))

        self.batch_input_list_metadata = []
        for (line, line_batch) in self.batch_input_list_contents:
            to_copy = []
            for (input, input_batch) in zip(line.split(), line_batch.split()):
                if ":=" in input:
                    to_copy.append((input.split(":=")[1], input_batch.split(":=")[1]))
                else:
                    to_copy.append((input, input_batch))
            self.batch_input_list_metadata.append((line_batch, to_copy))

        self.batches = []
        for i in range(0, len(self.batch_input_list_metadata), self.batch_size):
            self.batches.append(self.batch_input_list_metadata[i : i + self.batch_size])  # noqa: E203
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
                    self.batch_size * batch_id : self.batch_size * (batch_id + 1)  # noqa: E203
                ]
            batch = self.batches[batch_id]

            shutil.rmtree(self.batch_dir, ignore_errors=True)
            for (_, to_copy) in batch:
                for (input, input_batch) in to_copy:
                    Path(input_batch).parent.mkdir(parents=True, exist_ok=True)
                    # Path(input_batch).symlink_to(input)
                    shutil.copy(input, input_batch)

            batch_input_list = str(Path(self.batch_dir) / "input_list.txt")
            with open(batch_input_list, "w") as f:
                for header in self.batch_input_list_headers:
                    f.write(f"{header}\n")
                for (line_batch, _) in batch:
                    f.write(f"{line_batch}\n")

            return self.batch_dir, batch_input_list, annotation

    def __len__(self):
        return self.num_batches

    def __iter__(self):
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


class SNPEProcessedDataLoader(SNPEDataLoader):
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
        self.tmp_dir = tempfile.TemporaryDirectory(dir=str(Path.cwd()), prefix="olive_tmp_")
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


class SNPERandomDataLoader(SNPEDataLoader):
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
        self.tmp_dir = tempfile.TemporaryDirectory(dir=str(Path.cwd()), prefix="olive_tmp_")

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
            raw_shape = np.product(input_shape)
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
