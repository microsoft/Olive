# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import itertools
import json
import logging
import platform
import shutil
import sys
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Dict, List

import pkg_resources

from olive.common.utils import copy_dir, run_subprocess
from olive.engine.packaging.packaging_config import PackagingConfig, PackagingType
from olive.model import ONNXModelHandler
from olive.resource_path import ResourceType, create_resource_path
from olive.systems.utils import get_package_name_from_ep

if TYPE_CHECKING:
    from olive.engine.footprint import Footprint
    from olive.hardware import AcceleratorSpec

logger = logging.getLogger(__name__)

# ruff: noqa: N806


def generate_output_artifacts(
    packaging_config: PackagingConfig,
    footprints: Dict["AcceleratorSpec", "Footprint"],
    pf_footprints: Dict["AcceleratorSpec", "Footprint"],
    output_dir: Path,
):
    if sum(len(f.nodes) if f.nodes else 0 for f in pf_footprints.values()) == 0:
        logger.warning("No model is selected. Skip packaging output artifacts.")
        return
    if packaging_config.type == PackagingType.Zipfile:
        _generate_zipfile_output(packaging_config, footprints, pf_footprints, output_dir)


def _generate_zipfile_output(
    packaging_config: PackagingConfig,
    footprints: Dict["AcceleratorSpec", "Footprint"],
    pf_footprints: Dict["AcceleratorSpec", "Footprint"],
    output_dir: Path,
) -> None:
    logger.info("Packaging Zipfile output artifacts")
    cur_path = Path(__file__).parent
    with tempfile.TemporaryDirectory() as temp_dir:
        tempdir = Path(temp_dir)
        _package_sample_code(cur_path, tempdir)
        _package_models_rank(tempdir, pf_footprints)
        for accelerator_spec, pf_footprint in pf_footprints.items():
            if pf_footprint.nodes and footprints[accelerator_spec].nodes:
                _package_candidate_models(
                    tempdir,
                    footprints[accelerator_spec],
                    pf_footprint,
                    accelerator_spec,
                    packaging_config.export_in_mlflow_format,
                )
        _package_onnxruntime_packages(tempdir, next(iter(pf_footprints.values())))
        shutil.make_archive(packaging_config.name, "zip", tempdir)
        package_file = f"{packaging_config.name}.zip"
        shutil.move(package_file, output_dir / package_file)


def _package_models_rank(tempdir, footprints: Dict["AcceleratorSpec", "Footprint"]):
    metrics_dict = next(iter(footprints.values())).objective_dict
    sorted_nodes = sorted(
        itertools.chain.from_iterable(f.nodes.values() for f in footprints.values()),
        key=lambda x: tuple(
            x.metrics.value[metric].value if x.metrics.cmp_direction[metric] == 1 else -x.metrics.value[metric].value
            for metric in metrics_dict
        ),
        reverse=True,
    )
    rank = 1
    model_info_list = []
    for node in sorted_nodes:
        model_info = {
            "rank": rank,
            "model_config": node.model_config,
            "metrics": node.metrics.value.to_json() if node.metrics else None,
        }
        model_info_list.append(model_info)
        rank += 1
    with (tempdir / "models_rank.json").open("w") as f:
        f.write(json.dumps(model_info_list))


def _package_sample_code(cur_path, tempdir):
    copy_dir(cur_path / "sample_code", tempdir / "SampleCode")


def _package_candidate_models(
    tempdir,
    footprint: "Footprint",
    pf_footprint: "Footprint",
    accelerator_spec: "AcceleratorSpec",
    export_in_mlflow_format=False,
) -> None:
    candidate_models_dir = tempdir / "CandidateModels"
    model_rank = 1
    input_node = footprint.get_input_node()
    for model_id, node in pf_footprint.nodes.items():
        model_dir = candidate_models_dir / str(accelerator_spec) / f"BestCandidateModel_{model_rank}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_rank += 1

        # Copy inference config
        inference_config_path = model_dir / "inference_config.json"
        inference_config = pf_footprint.get_model_inference_config(model_id) or {}

        # Add use_ort_extensions to inference config if needed
        use_ort_extensions = pf_footprint.get_use_ort_extensions(model_id)
        if use_ort_extensions:
            inference_config["use_ort_extensions"] = True

        with inference_config_path.open("w") as f:
            json.dump(inference_config, f)

        # Copy model file
        model_path = pf_footprint.get_model_path(model_id)
        model_resource_path = create_resource_path(model_path) if model_path else None
        model_type = pf_footprint.get_model_type(model_id)
        if model_type.lower() == "onnxmodel":
            with tempfile.TemporaryDirectory(dir=model_dir, prefix="olive_tmp") as model_tempdir:
                # save to model_tempdir first since model_path may be a folder
                temp_resource_path = create_resource_path(model_resource_path.save_to_dir(model_tempdir, "model", True))
                # save to model_dir
                if temp_resource_path.type == ResourceType.LocalFile:
                    # if model_path is a file, rename it to model_dir / model.onnx
                    Path(temp_resource_path.get_path()).rename(model_dir / "model.onnx")
                elif temp_resource_path.type == ResourceType.LocalFolder:
                    # if model_path is a folder, save all files in the folder to model_dir / file_name
                    # file_name for .onnx file is model.onnx, otherwise keep the original file name
                    model_config = pf_footprint.get_model_config(model_id)
                    onnx_file_name = model_config.get("onnx_file_name")
                    onnx_model = ONNXModelHandler(temp_resource_path, onnx_file_name)
                    model_name = Path(onnx_model.model_path).name
                    for file in Path(temp_resource_path.get_path()).iterdir():
                        if file.name == model_name:
                            file_name = "model.onnx"
                        else:
                            file_name = file.name
                        Path(file).rename(model_dir / file_name)
                if export_in_mlflow_format:
                    try:
                        import mlflow
                    except ImportError:
                        raise ImportError("Exporting model in MLflow format requires mlflow>=2.4.0") from None
                    from packaging.version import Version

                    if Version(mlflow.__version__) < Version("2.4.0"):
                        logger.warning(
                            "Exporting model in MLflow format requires mlflow>=2.4.0. "
                            "Skip exporting model in MLflow format."
                        )
                    else:
                        _generate_onnx_mlflow_model(model_dir, inference_config)

        elif model_type.lower() == "openvinomodel":
            model_resource_path.save_to_dir(model_dir, "model", True)
        else:
            raise ValueError(f"Unsupported model type: {model_type} for packaging")

        # Copy Passes configurations
        configuration_path = model_dir / "configurations.json"
        with configuration_path.open("w") as f:
            json.dump(OrderedDict(reversed(footprint.trace_back_run_history(model_id).items())), f)

        # Copy metrics
        # TODO(xiaoyu): Add target info to metrics file
        if node.metrics:
            metric_path = model_dir / "metrics.json"
            with metric_path.open("w") as f:
                metrics = {
                    "input_model_metrics": input_node.metrics.value.to_json() if input_node.metrics else None,
                    "candidate_model_metrics": node.metrics.value.to_json(),
                }
                json.dump(metrics, f, indent=4)


def _generate_onnx_mlflow_model(model_dir, inference_config):
    import mlflow
    import onnx

    logger.info("Exporting model in MLflow format")
    execution_mode_mappping = {0: "SEQUENTIAL", 1: "PARALLEL"}

    session_dict = {}
    if inference_config.get("session_options"):
        session_dict = {k: v for k, v in inference_config.get("session_options").items() if v is not None}
        if "execution_mode" in session_dict:
            session_dict["execution_mode"] = execution_mode_mappping[session_dict["execution_mode"]]

    onnx_model_path = model_dir / "model.onnx"
    model_proto = onnx.load(onnx_model_path)
    onnx_model_path.unlink()

    # MLFlow will save models with default config save_as_external_data=True
    # https://github.com/mlflow/mlflow/blob/1d6eaaa65dca18688d9d1efa3b8b96e25801b4e9/mlflow/onnx.py#L175
    # There will be an aphanumeric file generated in the same folder as the model file
    mlflow.onnx.save_model(
        model_proto,
        model_dir / "mlflow_model",
        onnx_execution_providers=inference_config.get("execution_provider"),
        onnx_session_options=session_dict,
    )


def _package_onnxruntime_packages(tempdir, pf_footprint: "Footprint"):
    # pylint: disable=not-an-iterable
    installed_packages = pkg_resources.working_set
    onnxruntime_pkg = [i for i in installed_packages if i.key.startswith("onnxruntime")]
    ort_nightly_pkg = [i for i in installed_packages if i.key.startswith("ort-nightly")]
    is_nightly = bool(ort_nightly_pkg)
    is_stable = bool(onnxruntime_pkg)

    if not is_nightly and not is_stable:
        logger.warning("ONNXRuntime package is not installed. Skip packaging ONNXRuntime package.")
        return

    if is_nightly and is_stable:
        logger.warning("Both ONNXRuntime and ort-nightly packages are installed. Package ort-nightly package only.")

    ort_version = ort_nightly_pkg[0].version if is_nightly else onnxruntime_pkg[0].version
    use_ort_extensions = False

    for model_id in pf_footprint.nodes:
        if pf_footprint.get_use_ort_extensions(model_id):
            use_ort_extensions = True
        inference_settings = pf_footprint.get_model_inference_config(model_id)
        package_name_list = []
        if not inference_settings:
            package_name_list.append(("onnxruntime", "ort-nightly"))
        else:
            ep_list = inference_settings["execution_provider"]
            package_name_list.extend([get_package_name_from_ep(ep[0]) for ep in ep_list])
            package_name_list = set(package_name_list)

    try:
        # Download Python onnxruntime package
        NIGHTLY_PYTHON_COMMAND = Template(
            f"{sys.executable} -m pip download -i "
            "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ "
            "$package_name==$ort_version --no-deps -d $python_download_path"
        )
        STABLE_PYTHON_COMMAND = Template(
            f"{sys.executable} -m pip download $package_name==$ort_version --no-deps -d $python_download_path"
        )
        python_download_path = tempdir / "ONNXRuntimePackages" / "python"
        python_download_path.mkdir(parents=True, exist_ok=True)
        python_download_path = str(python_download_path)
        _download_ort_extensions_package(use_ort_extensions, python_download_path)

        if is_nightly:
            download_command_list = [
                NIGHTLY_PYTHON_COMMAND.substitute(
                    package_name=package_name[1], ort_version=ort_version, python_download_path=python_download_path
                )
                for package_name in package_name_list
                if package_name[1] is not None
            ]
        else:
            download_command_list = [
                STABLE_PYTHON_COMMAND.substitute(
                    package_name=package_name[0], ort_version=ort_version, python_download_path=python_download_path
                )
                for package_name in package_name_list
            ]

        for download_command in download_command_list:
            run_subprocess(download_command)

        # Download CPP && CS onnxruntime package
        lang_list = ("cpp", "cs")
        for language in lang_list:
            ort_download_path = tempdir / "ONNXRuntimePackages" / language
            ort_download_path.mkdir(parents=True, exist_ok=True)
            if is_nightly:
                _skip_download_c_package(ort_download_path)
            else:
                _download_c_packages(package_name_list, ort_version, ort_download_path)

    except Exception:
        logger.exception("Failed to download onnxruntime package. Please manually download onnxruntime package.")


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
                    f"{sys.executable} -m pip download -i "
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
            download_command = (
                f"{sys.executable} -m pip download onnxruntime-extensions=={version} --no-deps -d {download_path}"
            )
            run_subprocess(download_command)


def _download_c_packages(package_name_list: List[str], ort_version: str, ort_download_path: str):
    PACKAGE_DOWNLOAD_LINK_MAPPING = {
        "onnxruntime": Template("https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/$ort_version"),
        "onnxruntime-gpu": Template("https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/$ort_version"),
        "onnxruntime-directml": Template(
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$ort_version"
        ),
        "onnxruntime-openvino": None,
    }
    for package_name_tuple in package_name_list:
        package_name = package_name_tuple[0]
        download_link = PACKAGE_DOWNLOAD_LINK_MAPPING[package_name]
        download_path = str(ort_download_path / f"microsoft.ml.{package_name}.{ort_version}.nupkg")
        if download_link:
            urllib.request.urlretrieve(download_link.substitute(ort_version=ort_version), download_path)
        else:
            logger.warning(
                "Package %s is not available for packaging. Please manually install the package.", package_name
            )


def _skip_download_c_package(package_path):
    warning_msg = (
        "Found ort-nightly package installed. Please manually download "
        "ort-nightly package from https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly"
    )
    logger.warning(warning_msg)
    readme_path = package_path / "README.md"
    with readme_path.open("w") as f:
        f.write(warning_msg)
