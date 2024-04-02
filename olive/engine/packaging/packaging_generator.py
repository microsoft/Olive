# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
from typing import TYPE_CHECKING, Dict, List, Union

import pkg_resources

from olive.common.utils import copy_dir, retry_func, run_subprocess
from olive.engine.packaging.packaging_config import (
    AzureMLDataPackagingConfig,
    AzureMLModelsPackagingConfig,
    PackagingConfig,
    PackagingType,
)
from olive.model import ONNXModelHandler
from olive.resource_path import ResourceType, create_resource_path
from olive.systems.utils import get_package_name_from_ep

if TYPE_CHECKING:
    from olive.azureml.azureml_client import AzureMLClientConfig
    from olive.engine.footprint import Footprint, FootprintNode
    from olive.hardware import AcceleratorSpec

logger = logging.getLogger(__name__)

# ruff: noqa: N806


def generate_output_artifacts(
    packaging_configs: Union[PackagingConfig, List[PackagingConfig]],
    footprints: Dict["AcceleratorSpec", "Footprint"],
    pf_footprints: Dict["AcceleratorSpec", "Footprint"],
    output_dir: Path,
    azureml_client_config: "AzureMLClientConfig" = None,
):
    if sum(len(f.nodes) if f.nodes else 0 for f in pf_footprints.values()) == 0:
        logger.warning("No model is selected. Skip packaging output artifacts.")
        return
    packaging_config_list = packaging_configs if isinstance(packaging_configs, list) else [packaging_configs]
    for packaging_config in packaging_config_list:
        _package_candidate_models(packaging_config, output_dir, footprints, pf_footprints, azureml_client_config)


def _package_candidate_models(
    packaging_config: PackagingConfig,
    output_dir: Path,
    footprints: Dict["AcceleratorSpec", "Footprint"],
    pf_footprints: Dict["AcceleratorSpec", "Footprint"],
    azureml_client_config: "AzureMLClientConfig" = None,
):
    packaging_type = packaging_config.type
    output_name = packaging_config.name
    config = packaging_config.config
    export_in_mlflow_format = config.export_in_mlflow_format

    logger.info("Packaging output models to %s", packaging_type)

    with tempfile.TemporaryDirectory() as temp_dir:

        tempdir = Path(temp_dir)

        if packaging_type == PackagingType.Zipfile:
            cur_path = Path(__file__).parent
            _package_sample_code(cur_path, tempdir)
            _package_onnxruntime_packages(tempdir, next(iter(pf_footprints.values())))

        for accelerator_spec, pf_footprint in pf_footprints.items():
            footprint = footprints[accelerator_spec]
            if pf_footprint.nodes and footprint.nodes:
                model_rank = 1
                input_node = footprint.get_input_node()
                for model_id, node in pf_footprint.nodes.items():
                    model_name = f"{output_name}_{accelerator_spec}_{model_rank}"
                    if packaging_type == PackagingType.Zipfile:
                        model_dir = (
                            tempdir / "CandidateModels" / str(accelerator_spec) / f"BestCandidateModel_{model_rank}"
                        )
                    else:
                        model_dir = tempdir / model_name

                    model_dir.mkdir(parents=True, exist_ok=True)

                    # Copy inference config
                    inference_config_path = model_dir / "inference_config.json"
                    inference_config = pf_footprint.get_model_inference_config(model_id) or {}

                    _copy_inference_config(inference_config_path, inference_config)
                    _copy_configurations(model_dir, footprint, model_id)
                    _copy_metrics(model_dir, input_node, node)
                    model_path = _save_model(
                        pf_footprint, model_id, model_dir, inference_config, export_in_mlflow_format
                    )

                    model_info_list = []
                    relative_path = str(model_path.relative_to(tempdir))
                    model_info = _get_model_info(node, model_rank, relative_path, packaging_type)
                    model_info_list.append(model_info)
                    _copy_model_info(model_dir, model_info)

                    if packaging_type == PackagingType.AzureMLModels:
                        _upload_to_azureml_models(azureml_client_config, model_dir, model_name, config)
                    elif packaging_type == PackagingType.AzureMLData:
                        _upload_to_azureml_data(azureml_client_config, model_dir, model_name, config)

                model_rank += 1

        if packaging_type == PackagingType.Zipfile:
            _copy_models_rank(tempdir, model_info_list)
            _package_zipfile_model(output_dir, output_name, tempdir)


def _upload_to_azureml_models(
    azureml_client_config: "AzureMLClientConfig",
    model_path: Path,
    model_name: str,
    config: AzureMLModelsPackagingConfig,
):
    """Upload model to AzureML workspace Models."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.core.exceptions import ServiceResponseError

    ml_client = azureml_client_config.create_client()
    model = Model(
        path=model_path,
        type=AssetTypes.MLFLOW_MODEL if config.export_in_mlflow_format else AssetTypes.CUSTOM_MODEL,
        name=model_name,
        version=str(config.version),
        description=config.description,
    )
    retry_func(
        ml_client.models.create_or_update,
        [model],
        max_tries=azureml_client_config.max_operation_retries,
        delay=azureml_client_config.operation_retry_interval,
        exceptions=ServiceResponseError,
    )


def _upload_to_azureml_data(
    azureml_client_config: "AzureMLClientConfig", model_path: Path, model_name: str, config: AzureMLDataPackagingConfig
):
    """Upload model as Data to AzureML workspace Data."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data
    from azure.core.exceptions import ServiceResponseError

    ml_client = azureml_client_config.create_client()
    data = Data(
        path=str(model_path),
        type=AssetTypes.URI_FILE if model_path.is_file() else AssetTypes.URI_FOLDER,
        description=config.description,
        name=model_name,
        version=str(config.version),
    )
    retry_func(
        ml_client.data.create_or_update,
        [data],
        max_tries=azureml_client_config.max_operation_retries,
        delay=azureml_client_config.operation_retry_interval,
        exceptions=ServiceResponseError,
    )


def _get_model_info(node: "FootprintNode", model_rank: int, relative_path: str, packaging_type: PackagingType):
    model_config = node.model_config
    if packaging_type == PackagingType.Zipfile:
        model_config["config"]["model_path"] = relative_path
    return {
        "rank": model_rank,
        "model_config": model_config,
        "metrics": node.metrics.value.to_json() if node.metrics else None,
    }


def _copy_models_rank(tempdir: Path, model_info_list: List[Dict]):
    with (tempdir / "models_rank.json").open("w") as f:
        f.write(json.dumps(model_info_list))


def _package_sample_code(cur_path: Path, tempdir: Path):
    copy_dir(cur_path / "sample_code", tempdir / "SampleCode")


def _package_zipfile_model(output_dir: Path, output_name: str, model_dir: Path):
    shutil.make_archive(output_name, "zip", model_dir)
    package_file = f"{output_name}.zip"
    shutil.move(package_file, output_dir / package_file)


def _copy_model_info(model_dir: Path, model_info: Dict):
    model_info_path = model_dir / "model_info.json"
    with model_info_path.open("w") as f:
        json.dump(model_info, f, indent=4)


def _copy_inference_config(path: Path, inference_config: Dict):
    with path.open("w") as f:
        json.dump(inference_config, f, indent=4)


def _copy_configurations(model_dir: Path, footprint: "Footprint", model_id: str):
    configuration_path = model_dir / "configurations.json"
    with configuration_path.open("w") as f:
        json.dump(OrderedDict(reversed(footprint.trace_back_run_history(model_id).items())), f, indent=4)


# TODO(xiaoyu): Add target info to metrics file
def _copy_metrics(model_dir: Path, input_node: "FootprintNode", node: "FootprintNode"):
    metric_path = model_dir / "metrics.json"
    if node.metrics:
        with metric_path.open("w") as f:
            metrics = {
                "input_model_metrics": input_node.metrics.value.to_json() if input_node.metrics else None,
                "candidate_model_metrics": node.metrics.value.to_json(),
            }
            json.dump(metrics, f, indent=4)


def _save_model(
    pf_footprint: "Footprint",
    model_id: str,
    saved_model_path: Path,
    inference_config: Dict,
    export_in_mlflow_format: bool,
):
    model_path = pf_footprint.get_model_path(model_id)
    model_resource_path = create_resource_path(model_path) if model_path else None
    model_type = pf_footprint.get_model_type(model_id)

    if model_type.lower() == "onnxmodel":
        with tempfile.TemporaryDirectory(dir=saved_model_path, prefix="olive_tmp") as model_tempdir:
            # save to model_tempdir first since model_path may be a folder
            temp_resource_path = create_resource_path(model_resource_path.save_to_dir(model_tempdir, "model", True))
            # save to model_dir
            if temp_resource_path.type == ResourceType.LocalFile:
                # if model_path is a file, rename it to model_dir / model.onnx
                Path(temp_resource_path.get_path()).rename(saved_model_path / "model.onnx")
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
                    Path(file).rename(saved_model_path / file_name)
            if export_in_mlflow_format:
                _generate_onnx_mlflow_model(saved_model_path, inference_config)
                return saved_model_path / "mlflow_model"
            return (
                saved_model_path
                if model_resource_path.type == ResourceType.LocalFolder
                else saved_model_path / "model.onnx"
            )

    elif model_type.lower() == "openvinomodel":
        model_resource_path.save_to_dir(saved_model_path, "model", True)
        return saved_model_path
    else:
        raise ValueError(
            f"Unsupported model type: {model_type} for packaging,"
            " you can set `packaging_config` as None to mitigate this issue."
        )


def _generate_onnx_mlflow_model(model_dir: Path, inference_config: Dict):
    try:
        import mlflow
    except ImportError:
        raise ImportError("Exporting model in MLflow format requires mlflow>=2.4.0") from None
    from packaging.version import Version

    if Version(mlflow.__version__) < Version("2.4.0"):
        logger.warning(
            "Exporting model in MLflow format requires mlflow>=2.4.0. Skip exporting model in MLflow format."
        )
        return None

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
    mlflow_model_path = model_dir / "mlflow_model"

    # MLFlow will save models with default config save_as_external_data=True
    # https://github.com/mlflow/mlflow/blob/1d6eaaa65dca18688d9d1efa3b8b96e25801b4e9/mlflow/onnx.py#L175
    # There will be an aphanumeric file generated in the same folder as the model file
    mlflow.onnx.save_model(
        model_proto,
        mlflow_model_path,
        onnx_execution_providers=inference_config.get("execution_provider"),
        onnx_session_options=session_dict,
    )
    return mlflow_model_path


def _package_onnxruntime_packages(tempdir: Path, pf_footprint: "Footprint"):
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


def _skip_download_c_package(package_path: Path):
    warning_msg = (
        "Found ort-nightly package installed. Please manually download "
        "ort-nightly package from https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly"
    )
    logger.warning(warning_msg)
    readme_path = package_path / "README.md"
    with readme_path.open("w") as f:
        f.write(warning_msg)
