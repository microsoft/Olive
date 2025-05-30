# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Union

from olive.common.utils import retry_func
from olive.engine.output import ModelOutput, WorkflowOutput
from olive.engine.packaging.packaging_config import (
    AzureMLDeploymentPackagingConfig,
    DockerfilePackagingConfig,
    InferencingServerType,
    PackagingConfig,
    PackagingType,
)
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.resource_path import ResourceType, create_resource_path

if TYPE_CHECKING:
    from olive.azureml.azureml_client import AzureMLClientConfig

logger = logging.getLogger(__name__)

# ruff: noqa: N806


def generate_output_artifacts(
    packaging_configs: Union[PackagingConfig, list[PackagingConfig]],
    workflow_output: WorkflowOutput,
    output_dir: Path,
    azureml_client_config: "AzureMLClientConfig" = None,
):
    packaging_config_list = packaging_configs if isinstance(packaging_configs, list) else [packaging_configs]
    for packaging_config in packaging_config_list:
        if packaging_config.type == PackagingType.AzureMLDeployment:
            _package_azureml_deployment(packaging_config, workflow_output, azureml_client_config)
        elif packaging_config.type == PackagingType.Dockerfile:
            _package_dockerfile(packaging_config, workflow_output, output_dir)
        else:
            _package_candidate_models(packaging_config, output_dir, workflow_output, azureml_client_config)


def _package_dockerfile(
    packaging_config: PackagingConfig,
    workflow_output: WorkflowOutput,
    output_dir: Path,
):
    config: DockerfilePackagingConfig = packaging_config.config
    logger.info("Packaging output models to Dockerfile")
    base_image = config.base_image
    model_config = workflow_output.get_best_candidate().olive_model_config

    docker_context_path = "docker_content"
    content_path = output_dir / docker_context_path
    if content_path.exists():
        shutil.rmtree(content_path)
    content_path.mkdir(parents=True)

    if config.requirements_file:
        shutil.copy(config.requirements_file, content_path / "requirements.txt")

    _save_model(
        model_config["config"].get("model_path", None),
        model_config["type"],
        model_config,
        content_path,
        model_config["config"].get("inference_settings", None),
        False,
    )

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
    workflow_output: WorkflowOutput,
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
        # Get best model from workflow output
        model_config = workflow_output.get_best_candidate().olive_model_config

        with tempfile.TemporaryDirectory() as temp_dir:
            tempdir = Path(temp_dir)

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


def _package_candidate_models(
    packaging_config: PackagingConfig,
    output_dir: Path,
    workflow_output: WorkflowOutput,
    azureml_client_config: "AzureMLClientConfig" = None,
):
    packaging_type = packaging_config.type
    output_name = packaging_config.name
    config = packaging_config.config
    export_in_mlflow_format = config.export_in_mlflow_format

    logger.info("Packaging output models to %s", packaging_type)

    with tempfile.TemporaryDirectory() as temp_dir:
        tempdir = Path(temp_dir)

        output_model_list = workflow_output.get_output_models()
        model_rank = 1
        model_info_list = []
        for model_output in output_model_list:
            from_device = model_output.from_device()
            from_ep = model_output.from_execution_provider()
            accelerator_spec = AcceleratorSpec(accelerator_type=from_device, execution_provider=from_ep)
            model_name = f"{output_name}_{accelerator_spec}_{model_rank}"
            if packaging_type == PackagingType.Zipfile:
                model_dir = tempdir / "CandidateModels" / str(accelerator_spec) / f"BestCandidateModel_{model_rank}"
            else:
                model_dir = tempdir / model_name

            model_dir.mkdir(parents=True, exist_ok=True)

            # Copy inference config
            inference_config_path = model_dir / "inference_config.json"
            inference_config = model_output.get_inference_config()

            _copy_inference_config(inference_config_path, inference_config)
            _copy_configurations(model_dir, workflow_output, model_output.model_id)
            _copy_metrics(model_dir, workflow_output, model_output.metrics_value)

            model_path = _save_model(
                model_output.model_path,
                model_output.model_type,
                model_output.model_config,
                model_dir,
                inference_config,
                export_in_mlflow_format,
            )

            relative_path = str(model_path.relative_to(tempdir))
            model_info = _get_model_info(model_output, model_rank, relative_path, packaging_type)
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

        if model_info_list and packaging_type == PackagingType.Zipfile:
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


def _get_model_info(model_output: "ModelOutput", model_rank: int, relative_path: str, packaging_type: PackagingType):
    olive_model_config = model_output.olive_model_config
    if packaging_type == PackagingType.Zipfile:
        olive_model_config["config"]["model_path"] = relative_path
    return {"rank": model_rank, "model_config": olive_model_config, "metrics": model_output.metrics_value}


def _copy_models_rank(tempdir: Path, model_info_list: list[dict]):
    with (tempdir / "models_rank.json").open("w") as f:
        f.write(json.dumps(model_info_list))


def _package_zipfile_model(output_dir: Path, output_name: str, model_dir: Path):
    shutil.make_archive(output_name, "zip", model_dir)
    package_file = f"{output_name}.zip"
    shutil.move(package_file, output_dir / package_file)


def _copy_model_info(model_dir: Path, model_info: dict):
    model_info_path = model_dir / "model_info.json"
    with model_info_path.open("w") as f:
        json.dump(model_info, f, indent=4)


def _copy_inference_config(path: Path, inference_config: dict):
    with path.open("w") as f:
        json.dump(inference_config, f, indent=4)


def _copy_configurations(model_dir: Path, workflow_output: WorkflowOutput, model_id: str):
    configuration_path = model_dir / "configurations.json"
    with configuration_path.open("w") as f:
        json.dump(OrderedDict(reversed(workflow_output.trace_back_run_history(model_id).items())), f, indent=4)


# TODO(xiaoyu): Add target info to metrics file
def _copy_metrics(model_dir: Path, workflow_output: WorkflowOutput, output_model_metrics: dict):
    metric_path = model_dir / "metrics.json"
    if output_model_metrics:
        with metric_path.open("w") as f:
            metrics = {
                "input_model_metrics": workflow_output.get_input_model_metrics(),
                "candidate_model_metrics": output_model_metrics,
            }
            json.dump(metrics, f, indent=4)


def _save_model(
    model_path: str,
    model_type: str,
    model_config: dict,
    saved_model_path: Path,
    inference_config: dict,
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


def _generate_onnx_mlflow_model(model_dir: Path, inference_config: dict):
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
    execution_mode_mapping = {0: "SEQUENTIAL", 1: "PARALLEL"}

    session_dict = {}
    if inference_config.get("session_options"):
        session_dict = {k: v for k, v in inference_config.get("session_options").items() if v is not None}
        if "execution_mode" in session_dict:
            session_dict["execution_mode"] = execution_mode_mapping[session_dict["execution_mode"]]

    onnx_model_path = model_dir / "model.onnx"
    model_proto = onnx.load(onnx_model_path)
    onnx_model_path.unlink()
    mlflow_model_path = model_dir / "mlflow_model"

    # MLFlow will save models with default config save_as_external_data=True
    # https://github.com/mlflow/mlflow/blob/1d6eaaa65dca18688d9d1efa3b8b96e25801b4e9/mlflow/onnx.py#L175
    # There will be an alphanumeric file generated in the same folder as the model file
    mlflow.onnx.save_model(
        model_proto,
        mlflow_model_path,
        onnx_execution_providers=inference_config.get("execution_provider"),
        onnx_session_options=session_dict,
    )
    return mlflow_model_path


def _get_python_version():
    major_version = sys.version_info.major
    minor_version = sys.version_info.minor

    return f"{major_version}{minor_version}"
