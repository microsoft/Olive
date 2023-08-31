# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import BuildContext, Environment, Model
from azure.core.exceptions import HttpResponseError, ServiceResponseError

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.cache import normalize_data_path
from olive.common.config_utils import ParamCategory, validate_config
from olive.common.utils import retry_func
from olive.constants import Framework
from olive.evaluator.metric import Metric, MetricResult
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig, OliveModel
from olive.passes.olive_pass import Pass
from olive.resource_path import (
    AZUREML_RESOURCE_TYPES,
    LOCAL_RESOURCE_TYPES,
    OLIVE_RESOURCE_ANNOTATIONS,
    AzureMLModel,
    ResourcePath,
    ResourceType,
    create_resource_path,
)
from olive.systems.common import AzureMLDockerConfig, SystemType
from olive.systems.olive_system import OliveSystem

logger = logging.getLogger(__name__)

RESOURCE_TYPE_TO_ASSET_TYPE = {
    ResourceType.LocalFile: AssetTypes.URI_FILE,
    ResourceType.LocalFolder: AssetTypes.URI_FOLDER,
    ResourceType.StringName: None,
    ResourceType.AzureMLModel: AssetTypes.CUSTOM_MODEL,
    ResourceType.AzureMLDatastore: None,
    ResourceType.AzureMLJobOutput: AssetTypes.CUSTOM_MODEL,
}


def get_asset_type_from_resource_path(resource_path: ResourcePath):
    resource_path = create_resource_path(resource_path)  # just in case
    if not resource_path:
        # this is a placeholder for optional input
        return AssetTypes.URI_FILE

    if RESOURCE_TYPE_TO_ASSET_TYPE.get(resource_path.type):
        return RESOURCE_TYPE_TO_ASSET_TYPE[resource_path.type]

    if resource_path.type == ResourceType.AzureMLDatastore:
        return AssetTypes.URI_FILE if resource_path.is_file() else AssetTypes.URI_FOLDER

    # these won't be uploaded to azureml, so we use URI_FILE as a placeholder
    return AssetTypes.URI_FILE


class AzureMLSystem(OliveSystem):
    system_type = SystemType.AzureML

    def __init__(
        self,
        azureml_client_config: AzureMLClientConfig,
        aml_compute: str,
        aml_docker_config: Union[Dict[str, Any], AzureMLDockerConfig],
        instance_count: int = 1,
        is_dev: bool = False,
        accelerators: List[str] = None,
    ):
        super().__init__(accelerators)
        self._assert_not_none(aml_docker_config)
        aml_docker_config = validate_config(aml_docker_config, AzureMLDockerConfig)
        azureml_client_config = validate_config(azureml_client_config, AzureMLClientConfig)
        self.azureml_client_config = azureml_client_config
        self.compute = aml_compute
        self.environment = self._create_environment(aml_docker_config)
        self.instance_count = instance_count
        self.is_dev = is_dev

    def _create_environment(self, docker_config: AzureMLDockerConfig):
        if docker_config.build_context_path:
            return Environment(
                build=BuildContext(dockerfile_path=docker_config.dockerfile, path=docker_config.build_context_path)
            )
        if docker_config.base_image:
            return Environment(image=docker_config.base_image, conda_file=docker_config.conda_file_path)
        raise ValueError("Please specify DockerConfig.")

    def _assert_not_none(self, object):
        if object is None:
            raise ValueError(f"{object.__class__.__name__} is missing in the inputs!")

    def run_pass(
        self,
        the_pass: Pass,
        model: OliveModel,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """
        ml_client = self.azureml_client_config.create_client()
        point = point or {}
        config = the_pass.config_at_search_point(point)
        pass_config = the_pass.to_json(check_object=True)
        pass_config["config"].update(the_pass.serialize_config(config, check_object=True))

        with tempfile.TemporaryDirectory() as tempdir:
            pipeline_job = self._create_pipeline_for_pass(data_root, tempdir, model, pass_config, the_pass.path_params)

            # submit job
            logger.debug("Submitting pipeline")
            job = retry_func(
                ml_client.jobs.create_or_update,
                [pipeline_job],
                {"experiment_name": "olive-pass", "tags": {"Pass": pass_config["type"]}},
                max_tries=self.azureml_client_config.max_operation_retries,
                delay=self.azureml_client_config.operation_retry_interval,
                exceptions=HttpResponseError,
            )
            logger.info(f"Pipeline submitted. Job name: {job.name}. Job link: {job.studio_url}")
            ml_client.jobs.stream(job.name)

            # get output
            output_dir = Path(tempdir) / "pipeline_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Downloading pipeline output to {output_dir}")
            retry_func(
                ml_client.jobs.download,
                [job.name],
                {"output_name": "pipeline_output", "download_path": output_dir},
                max_tries=self.azureml_client_config.max_operation_retries,
                delay=self.azureml_client_config.operation_retry_interval,
                exceptions=ServiceResponseError,
            )

            pipeline_output_path = output_dir / "named-outputs" / "pipeline_output"

            return self._load_model(model, output_model_path, pipeline_output_path)

    def _create_model_inputs(self, model_resource_paths: Dict[str, ResourcePath]):
        inputs = {"model_config": Input(type=AssetTypes.URI_FILE)}
        # loop through all the model resource paths
        # create an input for each one using the resource type, with the name model_<resource_name>
        for path_name, resource_path in model_resource_paths.items():
            inputs[f"model_{path_name}"] = Input(type=get_asset_type_from_resource_path(resource_path), optional=True)
        return inputs

    def _create_args_from_resource_path(self, rp: OLIVE_RESOURCE_ANNOTATIONS):
        model_resource_path = create_resource_path(rp)
        if not model_resource_path:
            # no argument for this resource, placeholder for optional input
            return None
        asset_type = get_asset_type_from_resource_path(model_resource_path)

        if model_resource_path.type == ResourceType.AzureMLDatastore:
            asset_type = AssetTypes.URI_FILE if model_resource_path.is_file() else AssetTypes.URI_FOLDER

        if model_resource_path.type in AZUREML_RESOURCE_TYPES:
            # ensure that the model is in the same workspace as the system
            model_workspace_config = model_resource_path.get_aml_client_config().get_workspace_config()
            system_workspace_config = self.azureml_client_config.get_workspace_config()
            for key in model_workspace_config:
                if model_workspace_config[key] != system_workspace_config[key]:
                    raise ValueError(
                        f"Model workspace {model_workspace_config} is different from system workspace"
                        f" {system_workspace_config}. Olive will download the model to local storage, then upload it to"
                        "the system workspace."
                    )

        if model_resource_path.type == ResourceType.AzureMLJobOutput:
            # there is no direct way to use the output of a job as input to another job
            # so we create a dummy aml model and use it as input
            ml_client = self.azureml_client_config.create_client()

            # create aml model
            logger.debug(f"Creating aml model for job output {model_resource_path}")
            aml_model = retry_func(
                ml_client.models.create_or_update,
                [
                    Model(
                        path=model_resource_path.get_path(),
                        name="olive-backend-model",
                        description="Model created by Olive backend. Ignore this model.",
                        type=AssetTypes.CUSTOM_MODEL,
                    )
                ],
                max_tries=self.azureml_client_config.max_operation_retries,
                delay=self.azureml_client_config.operation_retry_interval,
                exceptions=ServiceResponseError,
            )
            model_resource_path = create_resource_path(
                AzureMLModel(
                    {
                        "azureml_client": self.azureml_client_config,
                        "name": aml_model.name,
                        "version": aml_model.version,
                    }
                )
            )
        # we keep the model path as a string in the config file
        if model_resource_path.type != ResourceType.StringName:
            return Input(type=asset_type, path=model_resource_path.get_path())

        return None

    def _create_model_args(self, model_json: dict, model_resource_paths: Dict[str, ResourcePath], tmp_dir: Path):
        args = {}
        # keep track of resource names in model_json that are uploaded/mounted
        model_json["resource_names"] = []

        for resource_name, resource_path in model_resource_paths.items():
            arg = self._create_args_from_resource_path(resource_path)
            if arg:
                model_json["config"][resource_name] = None
                model_json["resource_names"].append(resource_name)
            args[f"model_{resource_name}"] = arg

        # save the model json to a file
        model_config_path = tmp_dir / "model_config.json"
        with model_config_path.open("w") as f:
            json.dump(model_json, f, sort_keys=True, indent=4)
        args["model_config"] = Input(type=AssetTypes.URI_FILE, path=model_config_path)

        return args

    def _create_pass_inputs(self, pass_path_params: List[Tuple[str, bool, ParamCategory]]):
        inputs = {"pass_config": Input(type=AssetTypes.URI_FILE)}
        for param, required, category in pass_path_params:
            # aml supports uploading file/folder even though this is typed as URI_FOLDER
            inputs[f"pass_{param}"] = Input(type=AssetTypes.URI_FOLDER, optional=not required)

        return inputs

    def _create_pass_args(
        self, pass_config: dict, pass_path_params: List[Tuple[str, bool, ParamCategory]], data_root: str, tmp_dir: Path
    ):
        pass_args = {}
        for param, _, category in pass_path_params:
            param_val = pass_config["config"].get(param, None)
            if category == ParamCategory.DATA:
                if param_val:
                    # convert the dict to a resource path
                    param_val = create_resource_path(param_val)
                param_val = normalize_data_path(data_root, param_val)
            if not param_val:
                continue
            pass_args[f"pass_{param}"] = self._create_args_from_resource_path(param_val)
            pass_config["config"][param] = None

        pass_config_path = tmp_dir / "pass_config.json"
        with pass_config_path.open("w") as f:
            json.dump(pass_config, f, sort_keys=True, indent=4)

        return {"pass_config": Input(type=AssetTypes.URI_FILE, path=pass_config_path), **pass_args}

    def _create_step(
        self, name, display_name, description, aml_environment, code, compute, instance_count, inputs, script_name
    ):
        parameters = []
        for param, input in inputs.items():
            if isinstance(input, Input) and input.optional:
                parameters.append(f"$[[--{param} ${{{{inputs.{param}}}}}]]")
            else:
                parameters.append(f"--{param} ${{{{inputs.{param}}}}}")
        parameters.append("--pipeline_output ${{outputs.pipeline_output}}")

        cmd_line = f"python {script_name} {' '.join(parameters)}"

        component = command(
            name=name,
            display_name=display_name,
            description=description,
            command=cmd_line,
            environment=aml_environment,
            code=str(code),
            inputs=inputs,
            outputs=dict(pipeline_output=Output(type=AssetTypes.URI_FOLDER)),
            instance_count=instance_count,
            compute=compute,
        )

        return component

    def _create_pipeline_for_pass(
        self,
        data_root: str,
        tmp_dir,
        model: OliveModel,
        pass_config: dict,
        pass_path_params: List[Tuple[str, bool, ParamCategory]],
    ):
        tmp_dir = Path(tmp_dir)

        # prepare code
        script_name = "aml_pass_runner.py"
        cur_dir = Path(__file__).resolve().parent
        code_file = cur_dir / script_name
        code_root = tmp_dir / "code"
        code_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(code_file), str(code_root))
        if self.is_dev:
            logger.warning(
                "Dev mode is only enabled for CI pipeline! "
                + "It will overwrite the Olive package in AML computer with latest code."
            )
            project_folder = cur_dir.parent.parent
            shutil.copytree(project_folder, code_root / "olive", ignore=shutil.ignore_patterns("__pycache__"))

        accelerator_info = {
            "pass_accelerator_type": pass_config["accelerator"]["accelerator_type"],
            "pass_execution_provider": pass_config["accelerator"]["execution_provider"],
        }
        # prepare inputs
        inputs = {
            **self._create_model_inputs(model.resource_paths),
            **self._create_pass_inputs(pass_path_params),
            **accelerator_info,
        }

        # pass type
        pass_type = pass_config["type"]

        # aml command object
        cmd = self._create_step(
            name=pass_type,
            display_name=pass_type,
            description=f"Run olive {pass_type} pass",
            aml_environment=self.environment,
            code=str(code_root),
            compute=self.compute,
            instance_count=self.instance_count,
            inputs=inputs,
            script_name=script_name,
        )

        # model json
        model_json = model.to_json(check_object=True)

        # input argument values
        args = {
            **self._create_model_args(model_json, model.resource_paths, tmp_dir),
            **self._create_pass_args(pass_config, pass_path_params, data_root, tmp_dir),
            **accelerator_info,
        }

        @pipeline()
        def pass_runner_pipeline():
            outputs = {}
            component = cmd(**args)
            outputs["pipeline_output"] = component.outputs.pipeline_output
            return outputs

        pipeline_job = pass_runner_pipeline()

        return pipeline_job

    def _load_model(self, input_model, output_model_path, pipeline_output_path):
        model_json_path = pipeline_output_path / "output_model_config.json"
        with model_json_path.open("r") as f:
            model_json = json.load(f)

        # set the resources that are the same as the input model
        same_resources_as_input = model_json.pop("same_resources_as_input")
        for resource_name in same_resources_as_input:
            # get the resource path from the input model
            # do direct indexing to catch errors, should never happen
            model_json["config"][resource_name] = input_model.resource_paths[resource_name]
        # resolve resource names that are relative paths and save them to the output folder
        relative_resource_names = model_json.pop("resource_names")
        for resource_name in relative_resource_names:
            resource_json = model_json["config"][resource_name]
            # can only be local file or folder
            resource_type = resource_json["type"]
            assert resource_type in LOCAL_RESOURCE_TYPES, f"Expected local file or folder, got {resource_type}"
            # to be safe when downloading we will use the whole of output_model_path as a directory
            # and create subfolders for each resource
            # this is fine since the engine calls the system with a unique output_model_path which is a folder
            output_dir = Path(output_model_path).with_suffix("")
            output_name = resource_name.replace("_path", "")
            # if the model is downloaded from job, we need to copy it to the output folder
            # get the downloaded model path
            downloaded_path = pipeline_output_path / resource_json["config"]["path"]
            # create a resource path object for the downloaded path
            downloaded_resource_path = deepcopy(resource_json)
            downloaded_resource_path["config"]["path"] = str(downloaded_path)
            downloaded_resource_path = create_resource_path(downloaded_resource_path)
            # save the downloaded model to the output folder
            output_path = downloaded_resource_path.save_to_dir(output_dir, output_name, True)
            # create a resource path object for the output model
            output_resource_path = deepcopy(resource_json)
            output_resource_path["config"]["path"] = str(output_path)
            output_resource_path = create_resource_path(output_resource_path)
            model_json["config"][resource_name] = output_resource_path

        return ModelConfig(**model_json).create_model()

    def _create_metric_inputs(self):
        return {
            "metric_config": Input(type=AssetTypes.URI_FILE),
            "metric_user_script": Input(type=AssetTypes.URI_FILE, optional=True),
            "metric_script_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
            "metric_data_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
        }

    def _create_metric_args(self, data_root: str, metric_config: dict, tmp_dir: Path) -> Tuple[List[str], dict]:
        metric_user_script = metric_config["user_config"]["user_script"]
        if metric_user_script:
            metric_user_script = Input(type=AssetTypes.URI_FILE, path=metric_user_script)
            metric_config["user_config"]["user_script"] = None

        metric_script_dir = metric_config["user_config"]["script_dir"]
        if metric_script_dir:
            metric_script_dir = Input(type=AssetTypes.URI_FOLDER, path=metric_script_dir)
            metric_config["user_config"]["script_dir"] = None

        metric_data_dir = metric_config["user_config"]["data_dir"]
        # convert the dict to a resource path object
        metric_data_dir = create_resource_path(metric_data_dir)
        metric_data_dir = normalize_data_path(data_root, metric_data_dir)
        if metric_data_dir:
            metric_data_dir = self._create_args_from_resource_path(metric_data_dir)
            if metric_data_dir:
                metric_config["user_config"]["data_dir"] = None

        metric_config_path = tmp_dir / "metric_config.json"
        with metric_config_path.open("w") as f:
            json.dump(metric_config, f, sort_keys=True, indent=4)
        metric_config = Input(type=AssetTypes.URI_FILE, path=metric_config_path)

        return {
            "metric_config": metric_config,
            "metric_user_script": metric_user_script,
            "metric_script_dir": metric_script_dir,
            "metric_data_dir": metric_data_dir,
        }

    def evaluate_model(
        self, model: OliveModel, data_root: str, metrics: List[Metric], accelerator: AcceleratorSpec
    ) -> MetricResult:
        if model.framework == Framework.SNPE:
            raise NotImplementedError("SNPE model does not support azureml evaluation")
        if model.framework == Framework.OPENVINO:
            raise NotImplementedError("OpenVINO model does not support azureml evaluation")

        with tempfile.TemporaryDirectory() as tempdir:
            ml_client = self.azureml_client_config.create_client()
            pipeline_job = self._create_pipeline_for_evaluation(data_root, tempdir, model, metrics, accelerator)

            # submit job
            logger.debug("Submitting pipeline")
            job = retry_func(
                ml_client.jobs.create_or_update,
                [pipeline_job],
                {"experiment_name": "olive-evaluation"},
                max_tries=self.azureml_client_config.max_operation_retries,
                delay=self.azureml_client_config.operation_retry_interval,
                exceptions=HttpResponseError,
            )
            logger.info(f"Pipeline submitted. Job name: {job.name}. Job link: {job.studio_url}")
            ml_client.jobs.stream(job.name)

            # get output
            output_dir = Path(tempdir) / "pipeline_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            retry_func(
                ml_client.jobs.download,
                [job.name],
                {"download_path": output_dir, "all": True},
                max_tries=self.azureml_client_config.max_operation_retries,
                delay=self.azureml_client_config.operation_retry_interval,
                exceptions=ServiceResponseError,
            )

            metric_results = {}
            for metric in metrics:
                metric_json = output_dir / "named-outputs" / metric.name / "metric_result.json"
                if metric_json.is_file():
                    with metric_json.open() as f:
                        metric_results.update(json.load(f))

            return MetricResult.parse_obj(metric_results)

    def _create_pipeline_for_evaluation(
        self, data_root: str, tmp_dir: str, model: OliveModel, metrics: List[Metric], accelerator: AcceleratorSpec
    ):
        tmp_dir = Path(tmp_dir)

        # model json
        model_json = model.to_json(check_object=True)

        # model args
        model_args = self._create_model_args(model_json, model.resource_paths, tmp_dir)

        accelerator_config_path: Path = tmp_dir / "accelerator.json"
        with accelerator_config_path.open("w") as f:
            json.dump(accelerator.to_json(), f, sort_keys=True)

        @pipeline
        def evaluate_pipeline():
            outputs = {}
            for metric in metrics:
                metric_tmp_dir = tmp_dir / metric.name
                metric_component = self._create_metric_component(
                    data_root,
                    metric_tmp_dir,
                    metric,
                    model_args,
                    model.resource_paths,
                    accelerator_config_path,
                )
                outputs[metric.name] = metric_component.outputs.pipeline_output
            return outputs

        pipeline_job = evaluate_pipeline()
        pipeline_job.settings.default_compute = self.compute

        return pipeline_job

    def _create_metric_component(
        self,
        data_root: str,
        tmp_dir: Path,
        metric: Metric,
        model_args: Dict[str, Input],
        model_resource_paths: Dict[str, ResourcePath],
        accelerator_config_path: str,
    ):
        metric_json = metric.to_json(check_object=True)

        # prepare code
        script_name = "aml_evaluation_runner.py"
        cur_dir = Path(__file__).resolve().parent
        code_file = cur_dir / script_name
        code_root = tmp_dir / "code"
        code_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(code_file), str(code_root))
        if self.is_dev:
            logger.warning(
                "Dev mode is only enabled for CI pipeline! "
                + "It will overwrite the Olive package in AML computer with latest code."
            )
            project_folder = cur_dir.parent.parent
            shutil.copytree(project_folder, code_root / "olive", ignore=shutil.ignore_patterns("__pycache__"))

        # prepare inputs
        inputs = {
            **self._create_model_inputs(model_resource_paths),
            **self._create_metric_inputs(),
            **{"accelerator_config": Input(type=AssetTypes.URI_FILE)},
        }

        # metric type
        metric_type = metric_json["type"]
        if metric_json["sub_types"] is not None:
            sub_type_name = ",".join([st["name"] for st in metric_json["sub_types"]])
            metric_type = f"{metric_type}-{sub_type_name}"

        # aml command object
        cmd = self._create_step(
            name=metric_type,
            display_name=metric_type,
            description=f"Run olive {metric_type} evaluation",
            aml_environment=self.environment,
            code=str(code_root),
            compute=self.compute,
            instance_count=self.instance_count,
            inputs=inputs,
            script_name=script_name,
        )

        # input argument values
        args = {
            **model_args,
            **self._create_metric_args(data_root, metric_json, tmp_dir),
            **{"accelerator_config": Input(type=AssetTypes.URI_FILE, path=accelerator_config_path)},
        }

        # metric component
        metric_component = cmd(**args)

        return metric_component
