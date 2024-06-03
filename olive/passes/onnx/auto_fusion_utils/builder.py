# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import sys
import tarfile
import tempfile
import urllib
from pathlib import Path
from typing import List, Union

import triton

from olive.common.utils import copy_dir, run_subprocess
from olive.passes.onnx.auto_fusion_utils.codegen.ort_generator import create_custom_op, join_custom_ops
from olive.passes.onnx.auto_fusion_utils.codegen.triton_generator import create_kernel
from olive.passes.onnx.auto_fusion_utils.utils import get_env_path

logger = logging.getLogger(__name__)


class Builder:
    def __init__(
        self,
        kernel_infos: List[dict],
        out_dir: Union[str, Path],
        lib_name: str = "libcustom_op",
        constant_overrides: dict = None,
        ort_headers_dir: Union[str, Path] = None,
        ort_version: str = None,
    ):
        self.kernel_infos = kernel_infos
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lib_name = lib_name
        self.ort_headers_dir = ort_headers_dir
        self.ort_version = ort_version
        if not self.ort_headers_dir:
            assert self.ort_version, "one of ort_headers_dir or ort_version must be provided"
            assert self.check_ort_version(self.ort_version), f"Cannot find a release for ort version {self.ort_version}"

            self.ort_headers_dir = self.out_dir / "ort_includes"
            self.download_ort_headers(self.ort_version, self.ort_headers_dir)

        # create subdirectories
        self.sub_dirs = {}
        for sub_dir in ["csrc", "python"]:
            sub_dir_path = self.out_dir / sub_dir
            if sub_dir_path.exists():
                logger.warning("Directory %s already exists and will be overwritten", sub_dir_path)
                shutil.rmtree(sub_dir_path)
            sub_dir_path.mkdir(parents=True, exist_ok=True)
            self.sub_dirs[sub_dir] = sub_dir_path

        # TODO(jambayk): Consider tuning these constants
        self.constants = {
            "Elementwise": {
                "num_stages": 1,
                "num_warps": 8,
                "BLOCK_SIZE": 256,
            },
        }
        if constant_overrides:
            for op, constants in constant_overrides.items():
                self.constants[op].update(constants)
        logger.info("Builder constants: %s", self.constants)

    @staticmethod
    def get_ort_lib_url(ort_version: str) -> str:
        """Return the download url for onnxruntime release."""
        return f"https://github.com/microsoft/onnxruntime/releases/download/v{ort_version}/onnxruntime-linux-x64-{ort_version}.tgz"

    @classmethod
    def check_ort_version(cls, ort_version: str) -> bool:
        """Check if a release for the ort version exists."""
        try:
            with urllib.request.urlopen(cls.get_ort_lib_url(ort_version)) as url_request:
                return url_request.status == 200
        except urllib.error.URLError:
            return False

    @classmethod
    def download_ort_headers(cls, ort_version: str, destination: Union[str, Path]):
        """Download ort headers into destination."""
        logger.debug("Downloading ort headers for version %s to %s", ort_version, destination)
        url = cls.get_ort_lib_url(ort_version)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # download package
            archive_path = temp_dir_path / Path(url).name
            urllib.request.urlretrieve(url, archive_path)

            # unzip
            with tarfile.open(archive_path) as tar_ref:
                tar_ref.extractall(temp_dir_path)

            # copy over the contents of include
            copy_dir(temp_dir_path / archive_path.stem / "include", destination, dirs_exist_ok=True)

    def prepare_triton_kernels(self):
        for kernel_info in self.kernel_infos:
            # code gen kernel
            kernel_data = create_kernel(kernel_info)
            # write kernel to file
            kernel_file = self.sub_dirs["python"] / f"{kernel_data['kernel_name']}.py"
            with kernel_file.open("w") as f:
                f.write(kernel_data["kernel_code"])
            # c kernel
            kernel_dir = self.sub_dirs["csrc"] / kernel_data["kernel_name"]
            kernel_dir.mkdir(parents=True, exist_ok=True)
            constants = self.constants["Elementwise"]
            signature = kernel_data["signature"].format(**constants)
            grid = kernel_data["grid"].format(**constants)
            # aot compile and link
            self._compile_triton_kernel(
                kernel_file=kernel_file,
                kernel_name=kernel_data["kernel_name"],
                signature=signature,
                grid=grid,
                num_stages=constants["num_stages"],
                num_warps=constants["num_warps"],
                out_dir=kernel_dir,
            )

    @staticmethod
    def _compile_triton_kernel(
        kernel_file: Path, kernel_name: str, signature: str, grid: str, num_stages: int, num_warps: int, out_dir: Path
    ):
        # aot compile
        compiler_path = Path(triton.tools.__path__[0]) / "compile.py"

        run_subprocess(
            f"{sys.executable} {compiler_path} -n {kernel_name} --signature '{signature}' --out-name {kernel_name} -o"
            f" {kernel_name} -ns {num_stages} -w {num_warps} -g '{grid}' {kernel_file}",
            check=True,
            cwd=out_dir,
        )

        # link all desired configs
        linker_path = Path(triton.tools.__path__[0]) / "link.py"

        h_files = [str(file) for file in Path(out_dir).glob(f"{kernel_name}*.h")]
        run_subprocess(f"{sys.executable} {linker_path} {' '.join(h_files)} -o {kernel_name}", check=True, cwd=out_dir)

        # need to add extern C to the header file to avoid name mangling
        # header is used in c++ code also
        extern_c_start = ["#ifdef __cplusplus\n", 'extern "C" {\n', "#endif\n", "\n"]
        extern_c_end = ["\n", "#ifdef __cplusplus\n", "}\n", "#endif\n"]

        linked_kernel_header = out_dir / f"{kernel_name}.h"
        lines = []
        with linked_kernel_header.open() as f:
            lines = f.readlines()
        lines = lines[:2] + extern_c_start + lines[2:] + extern_c_end
        with linked_kernel_header.open("w") as f:
            f.writelines(lines)

    def prepare_custom_ops(self):
        # copy over the custom op files
        shutil.copytree(Path(__file__).parent / "codegen" / "custom_op_src", self.sub_dirs["csrc"], dirs_exist_ok=True)

        # write fusion op implementation
        custom_ops_data = [create_custom_op(kernel_info) for kernel_info in self.kernel_infos]
        custom_ops_file = self.sub_dirs["csrc"] / "fusion_ops.cc"
        with custom_ops_file.open("w") as f:
            f.write(join_custom_ops(custom_ops_data))

    def build(self) -> Path:
        self.prepare_triton_kernels()
        self.prepare_custom_ops()

        src_files = []
        for pattern in ["**/*.c", "*.cc"]:
            src_files.extend([str(file) for file in self.sub_dirs["csrc"].glob(pattern)])

        include_dirs = [
            f"-I {get_env_path('CUDA_HOME')/ 'include'}",
            f"-I {self.sub_dirs['csrc']}",
            f"-I {self.ort_headers_dir}",
        ]
        link_dirs = [f"-L {get_env_path('CUDA_HOME') / 'lib64'}"]
        link_libs = ["-l cuda"]

        version_script = self.sub_dirs["csrc"] / "custom_op_library.lds"

        # build shared library
        lib_path = self.out_dir / "lib" / f"{self.lib_name}.so"
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        run_subprocess(
            f"gcc -fPIC -shared -o {lib_path}"
            f" {' '.join(src_files)} {' '.join(include_dirs)} {' '.join(link_dirs)} {' '.join(link_libs)} -Xlinker"
            f" --version-script {version_script}",
            check=True,
        )
        return lib_path
