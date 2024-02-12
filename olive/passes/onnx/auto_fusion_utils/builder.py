# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Union

import triton

from olive.common.utils import run_subprocess
from olive.passes.onnx.auto_fusion_utils.codegen.ort_generator import join_custom_ops
from olive.passes.onnx.auto_fusion_utils.fuser import Fusion
from olive.passes.onnx.auto_fusion_utils.utils import get_env_path

logger = logging.getLogger(__name__)


class Builder:
    def __init__(self, fusions: List[Fusion], out_dir: Union[str, Path], lib_name: str = "libcustom_op"):
        self.fusions = fusions
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lib_name = lib_name

        # create subdirectories
        self.sub_dirs = {}
        for sub_dir in ["csrc", "python"]:
            sub_dir_path = self.out_dir / sub_dir
            if sub_dir_path.exists():
                logger.warning(f"Directory {sub_dir_path} already exists and will be overwritten")
                shutil.rmtree(sub_dir_path)
            sub_dir_path.mkdir(parents=True, exist_ok=True)
            self.sub_dirs[sub_dir] = sub_dir_path

        # TODO(jambayk): Consider tuning these constants
        self.constants = {
            "MatMul": {
                "num_stages": 5,
                "num_warps": 2,
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            "Elementwise": {
                "num_stages": 5,
                "num_warps": 4,
                "BLOCK_SIZE": 128,
            },
        }

    def prepare_triton_kernels(self):
        for fusion in self.fusions:
            # code gen kernel
            kernel_data = fusion.get_triton_kernel()
            # write kernel to file
            kernel_file = self.sub_dirs["python"] / f"{kernel_data['kernel_name']}.py"
            with kernel_file.open("w") as f:
                f.write(kernel_data["kernel_code"])
            # c kernel
            kernel_dir = self.sub_dirs["csrc"] / kernel_data["kernel_name"]
            kernel_dir.mkdir(parents=True, exist_ok=True)
            constants = self.constants.get(fusion.base_op, self.constants["Elementwise"])
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
        run_subprocess(f"python {linker_path} {' '.join(h_files)} -o {kernel_name}", check=True, cwd=out_dir)

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
        custom_ops_data = [fusion.get_custom_op() for fusion in self.fusions]
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
            f"-I {get_env_path('ONNXRUNTIME_DIR') / 'include' / 'onnxruntime'}",
            f"-I {get_env_path('ONNXRUNTIME_DIR') / 'include' / 'onnxruntime' / 'core' / 'session'}",
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
