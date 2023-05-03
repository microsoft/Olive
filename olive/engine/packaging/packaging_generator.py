# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import platform
import shutil
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path
from string import Template

import pkg_resources

from olive.common.utils import run_subprocess
from olive.engine.footprint import Footprint
from olive.engine.packaging.packaging_config import PackagingConfig, PackagingType

logger = logging.getLogger(__name__)


def generate_output_artifacts(
    packaging_config: PackagingConfig, foot_print: Footprint, pf_footprint: Footprint, output_dir: Path
):
    if packaging_config.type == PackagingType.Zipfile:
        _generate_zipfile_output(packaging_config, foot_print, pf_footprint, output_dir)


def _generate_zipfile_output(
    packaging_config: PackagingConfig, footprint: Footprint, pf_footprint: Footprint, output_dir: Path
) -> None:
    logger.info("Packaging Zipfile output artifacts")
    cur_path = Path(__file__).parent
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        _package_sample_code(cur_path, tempdir, pf_footprint)
        _package_candidate_models(tempdir, footprint, pf_footprint)
        _package_onnxruntime_packages(tempdir, pf_footprint)
        shutil.make_archive(packaging_config.name, "zip", tempdir)
        shutil.move(f"{packaging_config.name}.zip", output_dir / f"{packaging_config.name}.zip")


def _package_sample_code(cur_path, tempdir, pf_footprint: Footprint):
    shutil.copytree(cur_path / "sample_code", tempdir / "SampleCode")


def _package_candidate_models(tempdir, footprint: Footprint, pf_footprint: Footprint) -> None:
    candidate_models_dir = tempdir / "CandidateModels"
    candidate_models_dir.mkdir()
    model_rank = 1
    for model_id, node in pf_footprint.nodes.items():
        model_dir = candidate_models_dir / f"BestCandidateModel_{model_rank}"
        model_dir.mkdir()
        model_rank += 1
        # Copy model file
        model_path = pf_footprint.get_model_path(model_id)
        model_type = pf_footprint.get_model_type(model_id)
        if model_type == "ONNXModel":
            shutil.copy2(model_path, model_dir / "model.onnx")
        elif model_type == "OpenVINOModel":
            shutil.copytree(model_path, model_dir / "model")
        else:
            raise ValueError(f"Unsupported model type: {model_type} for packaging")

        # Copy inference config
        inference_config_path = str(model_dir / "inference_config.json")
        inference_config = pf_footprint.get_model_inference_config(model_id)
        if inference_config:
            inference_config["inference_settings"]["use_ort_extensions"] = pf_footprint.get_use_ort_extensions(model_id)
        with open(inference_config_path, "w") as f:
            json.dump(pf_footprint.get_model_inference_config(model_id), f)

        # Copy Passes configurations
        configuration_path = str(model_dir / "configurations.json")
        with open(configuration_path, "w") as f:
            json.dump(OrderedDict(reversed(footprint.trace_back_run_history(model_id).items())), f)

        # Copy metrics
        # TODO: Add target info to metrics file
        metric_path = str(model_dir / "metrics.json")
        with open(metric_path, "w") as f:
            json.dump(node.metrics.value, f)


def _package_onnxruntime_packages(tempdir, pf_footprint: Footprint):

    NIGHTLY_PYTHON_CPU_COMMAND = Template(
        "python -m pip download -i "
        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ "
        "ort-nightly==$ort_version --no-deps -d $python_download_path"
    )
    STABLE_PYTHON_CPU_COMMAND = Template(
        "python -m pip download onnxruntime==$ort_version --no-deps -d $python_download_path"
    )
    NIGHTLY_PYTHON_GPU_COMMAND = Template(
        "python -m pip download -i "
        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ "
        "ort-nightly-gpu==$ort_version --no-deps -d $python_download_path"
    )
    STABLE_PYTHON_GPU_COMMAND = Template(
        "python -m pip download onnxruntime-gpu==$ort_version --no-deps -d $python_download_path"
    )

    installed_packages = pkg_resources.working_set
    onnxruntime_pkg = [i for i in installed_packages if i.key == "onnxruntime"]
    ort_nightly_pkg = [i for i in installed_packages if i.key == "ort-nightly"]
    is_nightly = True if ort_nightly_pkg else False
    is_stable = True if onnxruntime_pkg else False

    if not is_nightly and not is_stable:
        logger.warning("ONNXRuntime package is not installed. Skip packaging ONNXRuntime package.")
        return

    # If both nightly and stable are installed, use nightly
    ort_version = ort_nightly_pkg[0].version if is_nightly else onnxruntime_pkg[0].version

    should_package_ort_cpu = False
    should_package_ort_gpu = False
    use_ort_extensions = False

    for model_id, _ in pf_footprint.nodes.items():
        if pf_footprint.get_use_ort_extensions(model_id):
            use_ort_extensions = True
        inference_config = pf_footprint.get_model_inference_config(model_id)
        if not inference_config:
            should_package_ort_cpu = True
        else:
            inference_settings = inference_config["inference_settings"]
            ep_list = inference_settings["execution_provider"]
            for ep_config in ep_list:
                ep = ep_config[0]
                if ep == "CUDAExecutionProvider":
                    should_package_ort_gpu = True
                else:
                    should_package_ort_cpu = True

    try:
        # Download Python onnxruntime package
        python_download_path = tempdir / "ONNXRuntimePackages" / "Python"
        python_download_path.mkdir(parents=True, exist_ok=True)
        python_download_path = str(python_download_path)
        _download_ort_extensions_package(use_ort_extensions, python_download_path)

        if should_package_ort_cpu:
            if is_nightly:
                download_command = NIGHTLY_PYTHON_CPU_COMMAND.substitute(
                    ort_version=ort_version, python_download_path=python_download_path
                )
            else:
                download_command = STABLE_PYTHON_CPU_COMMAND.substitute(
                    ort_version=ort_version, python_download_path=python_download_path
                )
            run_subprocess(download_command)
        if should_package_ort_gpu:
            if is_nightly:
                download_command = NIGHTLY_PYTHON_GPU_COMMAND.substitute(
                    ort_version=ort_version, python_download_path=python_download_path
                )
            else:
                download_command = STABLE_PYTHON_GPU_COMMAND.substitute(
                    ort_version=ort_version, python_download_path=python_download_path
                )
            run_subprocess(download_command)

        # Download CPP onnxruntime package
        cpp_ort_download_path = tempdir / "ONNXRuntimePackages" / "cpp"
        cpp_ort_download_path.mkdir(parents=True, exist_ok=True)
        if should_package_ort_cpu:
            cpp_download_path = str(cpp_ort_download_path / f"microsoft.ml.onnxruntime.{ort_version}.nupkg")
            _download_c_packages(True, is_nightly, ort_version, cpp_download_path)
        if should_package_ort_gpu:
            cpp_download_path = str(cpp_ort_download_path / f"microsoft.ml.onnxruntime.gpu.{ort_version}.nupkg")
            _download_c_packages(False, is_nightly, ort_version, cpp_download_path)

        # Download CS onnxruntime package
        cs_ort_download_path = tempdir / "ONNXRuntimePackages" / "cs"
        cs_ort_download_path.mkdir(parents=True, exist_ok=True)
        if should_package_ort_cpu:
            cs_download_path = str(cs_ort_download_path / f"microsoft.ml.onnxruntime.{ort_version}.nupkg")
            _download_c_packages(True, is_nightly, ort_version, cs_download_path)
        if should_package_ort_gpu:
            _download_c_packages(False, is_nightly, ort_version, cs_download_path)

    except Exception as e:
        logger.error(f"Failed to download onnxruntime package. Please manually download onnxruntime package. {e}")


def _download_ort_extensions_package(use_ort_extensions: bool, download_path: str):
    if use_ort_extensions:
        try:
            import onnxruntime_extensions
        except ImportError:
            logger.warning(
                "ONNXRuntime-Extensions package is not installed. Skip packaging ONNXRuntime-Extensions package."
            )
            return
        version = onnxruntime_extensions.__version__
        # Hardcode the nightly version number for now until we have a better way to identify nightly version
        if version.startswith("0.8.0."):
            system = platform.system()
            if system == "Windows":
                download_command = (
                    "python -m pip download -i "
                    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ "
                    f"onnxruntime-extensions=={version} --no-deps -d {download_path}"
                )
                run_subprocess(download_command)
            elif system == "Linux":
                logger.warning(
                    "ONNXRuntime-Extensions nightly package is not available for Linux. "
                    "Skip packaging ONNXRuntime-Extensions package. Please manually install ONNXRuntime-Extensions."
                )
        else:
            download_command = f"python -m pip download onnxruntime-extensions=={version} --no-deps -d {download_path}"
            run_subprocess(download_command)


def _download_c_packages(is_cpu: bool, is_nightly: bool, ort_version: str, download_path: str):
    NIGHTLY_C_CPU_LINK = Template(
        "https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/NuGet/"
        "Microsoft.ML.OnnxRuntime/overview/$ort_version"
    )
    STABLE_C_CPU_LINK = Template("https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/$ort_version")
    NIGHTLY_C_GPU_LINK = Template(
        "https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/NuGet/"
        "Microsoft.ML.OnnxRuntime.Gpu/overview/$ort_version"
    )
    STABLE_C_GPU_LINK = Template("https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/$ort_version")
    if is_cpu:
        if is_nightly:
            urllib.request.urlretrieve(NIGHTLY_C_CPU_LINK.substitute(ort_version=ort_version), download_path)
        else:
            urllib.request.urlretrieve(STABLE_C_CPU_LINK.substitute(ort_version=ort_version), download_path)
    else:
        if is_nightly:
            urllib.request.urlretrieve(NIGHTLY_C_GPU_LINK.substitute(ort_version=ort_version), download_path)
        else:
            urllib.request.urlretrieve(STABLE_C_GPU_LINK.substitute(ort_version=ort_version), download_path)
