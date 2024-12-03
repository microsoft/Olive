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
from typing import TYPE_CHECKING, Any, Dict, List, Set, Union

import pkg_resources

from olive.common.constants import OS
from olive.common.utils import retry_func, run_subprocess
from olive.engine.footprint import get_best_candidate_node
from olive.engine.packaging.packaging_config import (
    AzureMLDeploymentPackagingConfig,
    DockerfilePackagingConfig,
    InferencingServerType,
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
        if packaging_config.type == PackagingType.AzureMLDeployment:
            _package_azureml_deployment(packaging_config, footprints, pf_footprints, azureml_client_config)
        elif packaging_config.type == PackagingType.Dockerfile:
            _package_dockerfile(packaging_config, footprints, pf_footprints, output_dir)
        else:
            _package_candidate_models(packaging_config, output_dir, footprints, pf_footprints, azureml_client_config)


def _package_dockerfile(
    packaging_config: PackagingConfig,
    footprints: Dict["AcceleratorSpec", "Footprint"],
    pf_footprints: Dict["AcceleratorSpec", "Footprint"],
    output_dir: Path,
):
    config: DockerfilePackagingConfig = packaging_config.config
    logger.info("Packaging output models to Dockerfile")
    base_image = config.base_image
    best_node = get_best_candidate_node(pf_footprints, footprints)

    docker_context_path = "docker_content"
    content_path = output_dir / docker_context_path
    if content_path.exists():
        shutil.rmtree(content_path)
    content_path.mkdir(parents=True)

    if config.requirements_file:
        shutil.copy(config.requirements_file, content_path / "requirements.txt")

    model_config = best_node.model_config
    _save_model(
        model_config["config"].get("model_path", None),
        model_config["type"],
        model_config,
        content_path,
        model_config["config"].get("inference_settings", None),
        False,
    )
    is_generative = _is_generative_model(best_node.model_config["config"])
    if packaging_config.include_runtime_packages:
        if is_generative:
            _package_onnxruntime_genai_runtime_dependencies(content_path, False)
        else:
            _package_onnxruntime_runtime_dependencies(content_path, next(iter(pf_footprints.values())), "310", False)

    dockerfile_base_path = Path(__file__).parent / "Dockerfile.base"
    with open(dockerfile_base_path) as file:
        filedata = file.read()

    filedata = filedata.replace("<BASE_IMAGE>", base_image)
    filedata = filedata.replace("<DIR>", docker_context_path)

    dockerfile_path = output_dir / "Dockerfile"
    with open(dockerfile_path, "w") as file:
        file.writelines(filedata)


def _package_azureml_deployment(
    packaging_config: PackagingConfig,
    footprints: Dict["AcceleratorSpec", "Footprint"],
    pf_footprints: Dict["AcceleratorSpec", "Footprint"],
    azureml_client_config: "AzureMLClientConfig" = None,
):
    from azure.ai.ml.entities import (
        AzureMLBatchInferencingServer,
        AzureMLOnlineInferencingServer,
        BaseEnvironment,
        BatchDeployment,
        BatchEndpoint,
        CodeConfiguration,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        ModelConfiguration,
        ModelPackage,
    )
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, ServiceResponseError

    config: AzureMLDeploymentPackagingConfig = packaging_config.config
    if config.export_in_mlflow_format:
        logger.warning("Exporting model in MLflow format is not supported for AzureML endpoint packaging.")

    try:
        # Get best model from footprint
        best_node = get_best_candidate_node(pf_footprints, footprints)

        with tempfile.TemporaryDirectory() as temp_dir:
            tempdir = Path(temp_dir)

            model_config = best_node.model_config
            _save_model(
                model_config["config"].get("model_path", None),
                model_config["type"],
                model_config,
                tempdir,
                model_config["config"].get("inference_settings", None),
                False,
            )

            # Register model to AzureML
            _upload_to_azureml_models(
                azureml_client_config,
                tempdir,
                config.model_name,
                config.model_version,
                config.model_description,
                False,
            )

        ml_client = azureml_client_config.create_client()

        # AzureML package config
        model_package_config = config.model_package

        code_folder = Path(model_package_config.inferencing_server.code_folder)
        assert code_folder.exists(), f"Code folder {code_folder} does not exist."

        scoring_script = code_folder / model_package_config.inferencing_server.scoring_script
        assert scoring_script.exists(), f"Scoring script {scoring_script} does not exist."

        code_configuration = CodeConfiguration(
            code=model_package_config.inferencing_server.code_folder,
            scoring_script=model_package_config.inferencing_server.scoring_script,
        )

        inferencing_server = None
        if model_package_config.inferencing_server.type == InferencingServerType.AzureMLOnline:
            inferencing_server = AzureMLOnlineInferencingServer(code_configuration=code_configuration)
        elif model_package_config.inferencing_server.type == InferencingServerType.AzureMLBatch:
            inferencing_server = AzureMLBatchInferencingServer(code_configuration=code_configuration)

        model_configuration = None
        if model_package_config.model_configurations:
            model_configuration = ModelConfiguration(
                mode=model_package_config.model_configurations.mode,
                mount_path=model_package_config.model_configurations.mount_path,
            )

        base_environment_source = BaseEnvironment(
            type="EnvironmentAsset", resource_id=model_package_config.base_environment_id
        )

        package_request = ModelPackage(
            target_environment=model_package_config.target_environment,
            inferencing_server=inferencing_server,
            base_environment_source=base_environment_source,
            target_environment_version=model_package_config.target_environment_version,
            model_configuration=model_configuration,
            environment_variables=model_package_config.environment_variables,
        )

        # invoke model package operation
        model_package = retry_func(
            func=ml_client.models.package,
            kwargs={"name": config.model_name, "version": config.model_version, "package_request": package_request},
            max_tries=azureml_client_config.max_operation_retries,
            delay=azureml_client_config.operation_retry_interval,
            exceptions=ServiceResponseError,
        )

        logger.info(
            "Target environment created successfully: name: %s, version: %s",
            model_package_config.target_environment,
            model_package_config.target_environment_version,
        )

        # Deploy model package
        deployment_config = config.deployment_config

        # Get endpoint
        try:
            endpoint = retry_func(
                ml_client.online_endpoints.get,
                [deployment_config.endpoint_name],
                max_tries=azureml_client_config.max_operation_retries,
                delay=azureml_client_config.operation_retry_interval,
                exceptions=ServiceResponseError,
            )
            logger.info(
                "Endpoint %s already exists. The scoring_uri is: %s",
                deployment_config.endpoint_name,
                endpoint.scoring_uri,
            )
        except ResourceNotFoundError:
            logger.info("Endpoint %s does not exist. Creating a new endpoint...", deployment_config.endpoint_name)
            if model_package_config.inferencing_server.type == InferencingServerType.AzureMLOnline:
                endpoint = ManagedOnlineEndpoint(
                    name=deployment_config.endpoint_name,
                    description="this is an endpoint created by Olive automatically",
                )
            elif model_package_config.inferencing_server.type == InferencingServerType.AzureMLBatch:
                endpoint = BatchEndpoint(
                    name=deployment_config.endpoint_name,
                    description="this is an endpoint created by Olive automatically",
                )

            endpoint = retry_func(
                ml_client.online_endpoints.begin_create_or_update,
                [endpoint],
                max_tries=azureml_client_config.max_operation_retries,
                delay=azureml_client_config.operation_retry_interval,
                exceptions=ServiceResponseError,
            ).result()
            logger.info(
                "Endpoint %s created successfully. The scoring_uri is: %s",
                deployment_config.endpoint_name,
                endpoint.scoring_uri,
            )

        deployment = None
        extra_config = deployment_config.extra_config or {}
        if model_package_config.inferencing_server.type == InferencingServerType.AzureMLOnline:
            deployment = ManagedOnlineDeployment(
                name=deployment_config.deployment_name,
                endpoint_name=deployment_config.endpoint_name,
                environment=model_package,
                instance_type=deployment_config.instance_type,
                instance_count=deployment_config.instance_count,
                **extra_config,
            )

        elif model_package_config.inferencing_server.type == InferencingServerType.AzureMLBatch:
            deployment = BatchDeployment(
                name=deployment_config.deployment_name,
                endpoint_name=deployment_config.endpoint_name,
                environment=model_package,
                compute=deployment_config.compute,
                mini_batch_size=deployment_config.mini_batch_size,
                **extra_config,
            )
        deployment = retry_func(
            ml_client.online_deployments.begin_create_or_update,
            [deployment],
            max_tries=azureml_client_config.max_operation_retries,
            delay=azureml_client_config.operation_retry_interval,
            exceptions=ServiceResponseError,
        ).result()
        logger.info("Deployment %s created successfully", deployment.name)

    except ResourceNotFoundError:
        logger.exception(
            "Failed to package AzureML deployment. The resource is not found. Please check the exception details."
        )
        raise
    except ResourceExistsError:
        logger.exception(
            "Failed to package AzureML deployment. The resource already exists. Please check the exception details."
        )
        raise
    except Exception:
        logger.exception("Failed to package AzureML deployment. Please check the exception details.")
        raise


def _is_generative_model(config: Dict[str, Any]) -> bool:
    model_attributes = config.get("model_attributes") or {}
    return model_attributes.get("generative", False)


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
            best_node: FootprintNode = get_best_candidate_node(pf_footprints, footprints)
            is_generative = _is_generative_model(best_node.model_config["config"])

            if packaging_config.include_runtime_packages:
                if is_generative:
                    _package_onnxruntime_genai_runtime_dependencies(tempdir)
                else:
                    _package_onnxruntime_runtime_dependencies(
                        tempdir, next(iter(pf_footprints.values())), _get_python_version()
                    )

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
                        pf_footprint.get_model_path(model_id),
                        pf_footprint.get_model_type(model_id),
                        pf_footprint.get_model_config(model_id),
                        model_dir,
                        inference_config,
                        export_in_mlflow_format,
                    )

                    model_info_list = []
                    relative_path = str(model_path.relative_to(tempdir))
                    model_info = _get_model_info(node, model_rank, relative_path, packaging_type)
                    model_info_list.append(model_info)
                    _copy_model_info(model_dir, model_info)

                    if packaging_type == PackagingType.AzureMLModels:
                        _upload_to_azureml_models(
                            azureml_client_config,
                            model_dir,
                            model_name,
                            config.version,
                            config.description,
                            export_in_mlflow_format,
                        )
                    elif packaging_type == PackagingType.AzureMLData:
                        _upload_to_azureml_data(
                            azureml_client_config, model_dir, model_name, config.version, config.description
                        )

                    model_rank += 1

        if packaging_type == PackagingType.Zipfile:
            _copy_models_rank(tempdir, model_info_list)
            _package_zipfile_model(output_dir, output_name, tempdir)


def _upload_to_azureml_models(
    azureml_client_config: "AzureMLClientConfig",
    model_path: Path,
    model_name: str,
    version: Union[int, str],
    description: str,
    export_in_mlflow_format: bool,
):
    """Upload model to AzureML workspace Models."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.core.exceptions import ServiceResponseError

    ml_client = azureml_client_config.create_client()
    model = Model(
        path=model_path,
        type=AssetTypes.MLFLOW_MODEL if export_in_mlflow_format else AssetTypes.CUSTOM_MODEL,
        name=model_name,
        version=str(version),
        description=description,
    )
    retry_func(
        ml_client.models.create_or_update,
        [model],
        max_tries=azureml_client_config.max_operation_retries,
        delay=azureml_client_config.operation_retry_interval,
        exceptions=ServiceResponseError,
    )


def _upload_to_azureml_data(
    azureml_client_config: "AzureMLClientConfig",
    model_path: Path,
    model_name: str,
    version: Union[int, str],
    description: str,
):
    """Upload model as Data to AzureML workspace Data."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data
    from azure.core.exceptions import ServiceResponseError

    ml_client = azureml_client_config.create_client()
    data = Data(
        path=str(model_path),
        type=AssetTypes.URI_FILE if model_path.is_file() else AssetTypes.URI_FOLDER,
        description=description,
        name=model_name,
        version=str(version),
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
    model_path: str,
    model_type: str,
    model_config: Dict,
    saved_model_path: Path,
    inference_config: Dict,
    export_in_mlflow_format: bool,
):
    model_resource_path = create_resource_path(model_path) if model_path else None

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


def create_python_download_command(base_url=None):
    command = f"{sys.executable} -m pip download"
    if base_url:
        command += f" -i {base_url}"
    command += " $package_name==$version --no-deps -d $python_download_path --python-version=$python_version"
    return Template(command)


def _package_onnxruntime_genai_runtime_dependencies(save_path: Path, download_c_packages: bool = True):
    # pylint: disable=not-an-iterable
    installed_packages = [
        pkg
        for pkg in pkg_resources.working_set
        if pkg.key.startswith("onnxruntime-genai") or pkg.project_name.startswith("onnxruntime-genai")
    ]
    if not installed_packages:
        logger.warning("ONNXRuntime-GenAI package is not installed. Skip packaging runtime packages.")
        return

    DOWNLOAD_COMMAND_TEMPLATE = create_python_download_command()
    python_download_path = save_path / "ONNXRuntimePackages" / "python"
    python_download_path.mkdir(parents=True, exist_ok=True)
    python_download_path = str(python_download_path)

    for pkg in installed_packages:
        pkg_name = pkg.key if pkg.key.startswith("onnxruntime-genai") else pkg.project_name
        download_command = DOWNLOAD_COMMAND_TEMPLATE.substitute(
            package_name=pkg_name, version=pkg.version, python_download_path=python_download_path
        )

        try:
            run_subprocess(download_command)
        except Exception:
            logger.exception(
                "Failed to download %s package. Please manually download & install the required package.", pkg_name
            )

    # Download CPP && CS onnxruntime-genai packages
    if download_c_packages:
        ort_version = installed_packages[0].version
        lang_list = ("cpp", "cs")
        for language in lang_list:
            ort_download_path = save_path / "ONNXRuntimePackages" / language
            ort_download_path.mkdir(parents=True, exist_ok=True)
            _download_native_onnx_packages(installed_packages, ort_version, ort_download_path)


def _package_onnxruntime_runtime_dependencies(
    save_path: Path, pf_footprint: "Footprint", python_version: str, download_c_packages: bool = True
):
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
    package_name_list = set()
    use_ort_extensions = False
    for model_id in pf_footprint.nodes:
        if pf_footprint.get_use_ort_extensions(model_id):
            use_ort_extensions = True

        inference_settings = pf_footprint.get_model_inference_config(model_id)
        if inference_settings:
            ep_list = inference_settings["execution_provider"]
            for ep in ep_list:
                pkg_tuple = get_package_name_from_ep(ep[0])
                pkg_name = pkg_tuple[1] if is_nightly else pkg_tuple[0]
                package_name_list.update([pkg_name])
        else:
            pkg_name = "ort-nightly" if is_nightly else "onnxruntime"
            package_name_list.update([pkg_name])

    try:
        # Download Python onnxruntime package
        NIGHTLY_PYTHON_URL = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/"
        NIGHTLY_PYTHON_COMMAND = create_python_download_command(NIGHTLY_PYTHON_URL)
        STABLE_PYTHON_COMMAND = create_python_download_command()
        python_download_path = save_path / "ONNXRuntimePackages" / "python"
        python_download_path.mkdir(parents=True, exist_ok=True)
        python_download_path = str(python_download_path)
        _download_ort_extensions_package(use_ort_extensions, python_download_path, python_version)
        if is_nightly:
            download_command_list = [
                NIGHTLY_PYTHON_COMMAND.substitute(
                    package_name=package_name, version=ort_version, python_download_path=python_download_path
                )
                for package_name in package_name_list
            ]
        else:
            download_command_list = [
                STABLE_PYTHON_COMMAND.substitute(
                    package_name=package_name,
                    version=ort_version,
                    python_download_path=python_download_path,
                    python_version=python_version,
                )
                for package_name in package_name_list
            ]
        for download_command in download_command_list:
            run_subprocess(download_command)

        # Download CPP && CS onnxruntime package
        if download_c_packages:
            lang_list = ("cpp", "cs")
            for language in lang_list:
                ort_download_path = save_path / "ONNXRuntimePackages" / language
                ort_download_path.mkdir(parents=True, exist_ok=True)
                if is_nightly:
                    _skip_download_c_package(ort_download_path)
                else:
                    _download_native_onnx_packages(package_name_list, ort_version, ort_download_path)

    except Exception:
        logger.exception("Failed to download onnxruntime package. Please manually download onnxruntime package.")


def _download_ort_extensions_package(use_ort_extensions: bool, download_path: str, python_version: str):
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
            if system == OS.WINDOWS:
                NIGHTLY_URL = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/"
                download_command = create_python_download_command(NIGHTLY_URL).substitute(
                    package_name="onnxruntime_extensions",
                    version=version,
                    python_download_path=download_path,
                    python_version=python_version,
                )
                run_subprocess(download_command)
            elif system == OS.LINUX:
                logger.warning(
                    "ONNXRuntime-Extensions nightly package is not available for Linux. "
                    "Skip packaging ONNXRuntime-Extensions package. Please manually install ONNXRuntime-Extensions."
                )
        else:
            download_command = create_python_download_command().substitute(
                package_name="onnxruntime_extensions",
                version=version,
                python_download_path=download_path,
                python_version=python_version,
            )
            run_subprocess(download_command)


def _download_native_onnx_packages(package_name_list: Set[str], ort_version: str, ort_download_path: str):
    PACKAGE_DOWNLOAD_LINK_MAPPING = {
        "onnxruntime": Template("https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/$ort_version"),
        "onnxruntime-gpu": Template("https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/$ort_version"),
        "onnxruntime-directml": Template(
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$ort_version"
        ),
        "onnxruntime-openvino": None,
        "onnxruntime-genai": Template(
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntimeGenAI.Managed/$ort_version"
        ),
        "onnxruntime-genai-cuda": Template(
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.OnnxRuntimeGenAI.Cuda/$ort_version"
        ),
        "onnxruntime-genai-directml": Template(
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.OnnxRuntimeGenAI.DirectML/$ort_version"
        ),
    }
    for package_name in package_name_list:
        download_link = PACKAGE_DOWNLOAD_LINK_MAPPING.get(package_name)
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


def _get_python_version():
    major_version = sys.version_info.major
    minor_version = sys.version_info.minor

    return f"{major_version}{minor_version}"
