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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import BuildContext, Environment, Model, UserIdentityConfiguration
from azure.ai.ml.exceptions import JobException
from azure.core.exceptions import HttpResponseError, ServiceResponseError

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import validate_config
from olive.common.utils import copy_dir, get_dict_value, retry_func, set_dict_value
from olive.evaluator.metric_result import MetricResult
from olive.model import ModelConfig
from olive.resource_path import (
    AZUREML_RESOURCE_TYPES,
    LOCAL_RESOURCE_TYPES,
    AzureMLModel,
    ResourcePath,
    ResourceType,
    create_resource_path,
    find_all_resources,
)
from olive.systems.common import AcceleratorConfig, AzureMLDockerConfig, AzureMLEnvironmentConfig, SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import AzureMLTargetUserConfig

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.passes.olive_pass import Pass


logger = logging.getLogger(__name__)

RESOURCE_TYPE_TO_ASSET_TYPE = {
    ResourceType.LocalFile: AssetTypes.URI_FILE,
    ResourceType.LocalFolder: AssetTypes.URI_FOLDER,
    ResourceType.StringName: None,
    ResourceType.AzureMLModel: AssetTypes.CUSTOM_MODEL,
    ResourceType.AzureMLRegistryModel: AssetTypes.CUSTOM_MODEL,
    ResourceType.AzureMLDatastore: None,
    ResourceType.AzureMLJobOutput: AssetTypes.CUSTOM_MODEL,
}


def get_asset_type_from_resource_path(resource_path: ResourcePath):
    resource_path = create_resource_path(resource_path)  # just in case

    if RESOURCE_TYPE_TO_ASSET_TYPE.get(resource_path.type):
        return RESOURCE_TYPE_TO_ASSET_TYPE[resource_path.type]

    if resource_path.type == ResourceType.AzureMLDatastore:
        return AssetTypes.URI_FILE if resource_path.is_file() else AssetTypes.URI_FOLDER

    # these won't be uploaded to azureml, so we use URI_FILE as a placeholder
    return AssetTypes.URI_FILE


# TODO(jambayk): remove data_root. Not used at all and makes the logic unnecessarily complex
class AzureMLSystem(OliveSystem):
    system_type = SystemType.AzureML
    olive_config = None

    def __init__(
        self,
        azureml_client_config: AzureMLClientConfig,
        aml_compute: str,
        aml_docker_config: Union[Dict[str, Any], AzureMLDockerConfig] = None,
        aml_environment_config: Union[Dict[str, Any], AzureMLEnvironmentConfig] = None,
        tags: Dict = None,
        resources: Dict = None,
        instance_count: int = 1,
        is_dev: bool = False,
        accelerators: List[AcceleratorConfig] = None,
        hf_token: bool = None,
        **kwargs,
    ):
        super().__init__(accelerators, hf_token=hf_token)

        self.config = AzureMLTargetUserConfig(**locals(), **kwargs)

        self.instance_count = instance_count
        self.tags = tags or {}
        self.resources = resources
        self.is_dev = is_dev
        self.compute = aml_compute
        self.azureml_client_config = validate_config(azureml_client_config, AzureMLClientConfig)
        if not aml_docker_config and not aml_environment_config:
            raise ValueError("either aml_docker_config or aml_environment_config should be provided.")

        self.environment = None
        if aml_environment_config:
            from azure.core.exceptions import ResourceNotFoundError

            aml_environment_config = validate_config(aml_environment_config, AzureMLEnvironmentConfig)
            try:
                self.environment = self._get_environment_from_config(aml_environment_config)
            except ResourceNotFoundError:
                if not aml_docker_config:
                    raise

        if self.environment is None and aml_docker_config:
            aml_docker_config = validate_config(aml_docker_config, AzureMLDockerConfig)
            self.environment = self._create_environment(aml_docker_config)
        self.env_vars = self._get_hf_token_env(self.azureml_client_config.keyvault_name) if self.hf_token else None
        self.temp_dirs = []

    def _get_hf_token_env(self, keyvault_name: Optional[str]):
        if keyvault_name is None:
            raise ValueError(
                "hf_token is set to True but keyvault name is not provided. "
                "Please provide a keyvault name to use HF_TOKEN."
            )
        env_vars = {"HF_LOGIN": True}
        env_vars.update({"KEYVAULT_NAME": keyvault_name})
        return env_vars

    def _get_environment_from_config(self, aml_environment_config: AzureMLEnvironmentConfig):
        ml_client = self.azureml_client_config.create_client()
        return retry_func(
            ml_client.environments.get,
            [aml_environment_config.name, aml_environment_config.version, aml_environment_config.label],
            max_tries=self.azureml_client_config.max_operation_retries,
            delay=self.azureml_client_config.operation_retry_interval,
            exceptions=ServiceResponseError,
        )

    def _create_environment(self, docker_config: AzureMLDockerConfig):
        if docker_config.build_context_path:
            return Environment(
                name=docker_config.name,
                version=docker_config.version,
                build=BuildContext(dockerfile_path=docker_config.dockerfile, path=docker_config.build_context_path),
            )
        elif docker_config.base_image:
            return Environment(
                name=docker_config.name,
                version=docker_config.version,
                image=docker_config.base_image,
                conda_file=docker_config.conda_file_path,
            )
        raise ValueError("Please specify DockerConfig.")

    def _assert_not_none(self, obj):
        if obj is None:
            raise ValueError(f"{obj.__class__.__name__} is missing in the inputs!")

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: ModelConfig,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """Run the pass on the model at a specific point in the search space."""
        ml_client = self.azureml_client_config.create_client()

        # serialize pass
        point = point or {}
        config = the_pass.config_at_search_point(point)
        pass_config = the_pass.to_json(check_object=True)
        pass_config["config"].update(the_pass.serialize_config(config, check_object=True))

        with tempfile.TemporaryDirectory() as tempdir:
            pipeline_job = self._create_pipeline_for_pass(tempdir, model_config.to_json(check_object=True), pass_config)

            # submit job
            named_outputs_dir = self._run_job(
                ml_client,
                pipeline_job,
                "olive-pass",
                tempdir,
                tags={"Pass": pass_config["type"]},
                output_name="pipeline_output",
            )
            pipeline_output_path = named_outputs_dir / "pipeline_output"

            return self._load_model(model_config.to_json(check_object=True), output_model_path, pipeline_output_path)

    def create_inputs_and_args(self, name: str, config_json: dict, tmp_dir: Path, ignore_keys=None):
        inputs = {f"{name}_config": Input(type=AssetTypes.URI_FILE)}
        args = {}

        resource_map = {}
        # create inputs and args for each resource in the config
        for resource_key, resource_path in find_all_resources(config_json, ignore_keys=None).items():
            resource_name = f"{name}__{'__'.join(map(str, resource_key))}"
            inputs[resource_name], args[resource_name] = self._create_arg_and_input_from_resource_path(resource_path)
            resource_map[resource_name] = resource_key
            set_dict_value(config_json, resource_key, None)

        if resource_map:
            resource_map_path = tmp_dir / f"{name}_resource_map.json"
            with resource_map_path.open("w") as f:
                json.dump(resource_map, f, sort_keys=True, indent=4)
            inputs[f"{name}_resource_map"] = Input(type=AssetTypes.URI_FILE)
            args[f"{name}_resource_map"] = Input(type=AssetTypes.URI_FILE, path=resource_map_path)

        config_path = tmp_dir / f"{name}_config.json"
        with config_path.open("w") as f:
            json.dump(config_json, f, sort_keys=True, indent=4)
        args[f"{name}_config"] = Input(type=AssetTypes.URI_FILE, path=config_path)

        return inputs, args

    def _create_arg_and_input_from_resource_path(self, resource_path: ResourcePath):
        asset_type = get_asset_type_from_resource_path(resource_path)

        if resource_path.type in AZUREML_RESOURCE_TYPES:
            # ensure that the model is in the same workspace as the system
            model_workspace_config = resource_path.get_aml_client_config().get_workspace_config()
            system_workspace_config = self.azureml_client_config.get_workspace_config()
            for key in model_workspace_config:
                if model_workspace_config[key] != system_workspace_config[key]:
                    raise ValueError(
                        f"Model workspace {model_workspace_config} is different from system workspace"
                        f" {system_workspace_config}. Olive will download the model to local storage, then upload it to"
                        "the system workspace."
                    )

        if resource_path.type == ResourceType.AzureMLJobOutput:
            # there is no direct way to use the output of a job as input to another job
            # so we create a dummy aml model and use it as input
            ml_client = self.azureml_client_config.create_client()

            # create aml model
            logger.debug("Creating aml model for job output %s", resource_path)
            aml_model = retry_func(
                ml_client.models.create_or_update,
                [
                    Model(
                        path=resource_path.get_path(),
                        name="olive-backend-model",
                        description="Model created by Olive backend. Ignore this model.",
                        type=AssetTypes.CUSTOM_MODEL,
                    )
                ],
                max_tries=self.azureml_client_config.max_operation_retries,
                delay=self.azureml_client_config.operation_retry_interval,
                exceptions=ServiceResponseError,
            )
            resource_path = create_resource_path(
                AzureMLModel(
                    {
                        "azureml_client": self.azureml_client_config,
                        "name": aml_model.name,
                        "version": aml_model.version,
                    }
                )
            )

        return Input(type=asset_type), Input(type=asset_type, path=resource_path.get_path())

    def _create_olive_config_file(self, olive_config: Optional[dict], tmp_dir: Path):
        if olive_config is None:
            return None

        olive_config_path = tmp_dir / "olive_config.json"
        with olive_config_path.open("w") as f:
            json.dump(olive_config, f, sort_keys=True, indent=4)
        return olive_config_path

    def _create_step(
        self,
        name,
        display_name,
        description,
        aml_environment,
        code,
        compute,
        resources,
        instance_count,
        inputs,
        outputs,
        script_name,
    ):
        # create arguments for inputs and outputs
        parameters = []
        inputs = inputs or {}
        for param, job_input in inputs.items():
            if isinstance(job_input, Input) and job_input.optional:
                parameters.append(f"$[[--{param} ${{{{inputs.{param}}}}}]]")
            else:
                parameters.append(f"--{param} ${{{{inputs.{param}}}}}")
        outputs = outputs or {}
        parameters.extend([f"--{param} ${{{{outputs.{param}}}}}" for param in outputs])

        cmd_line = f"python {script_name} {' '.join(parameters)}"
        env_vars = deepcopy(self.env_vars) if self.env_vars else {}
        env_vars["OLIVE_LOG_LEVEL"] = logging.getLevelName(logger.getEffectiveLevel())

        # the name need to be lowercase
        # https://github.com/Azure/azure-sdk-for-python/blob/8b5217499caedba762b47fa6a118e51209f6f604/sdk/ml/azure-ai-ml/azure/ai/ml/entities/_builders/base_node.py#L217
        return command(
            name=name.lower(),  # convert to lowercase to avoid AzureML name restrictions
            display_name=display_name,
            description=description,
            command=cmd_line,
            resources=resources,
            environment=aml_environment,
            environment_variables=env_vars,
            code=str(code),
            inputs=inputs,
            outputs=outputs,
            instance_count=instance_count,
            compute=compute,
            identity=UserIdentityConfiguration(),
        )

    def _create_pipeline_for_pass(
        self,
        tmp_dir,
        model_config: dict,
        pass_config: dict,
    ):
        tmp_dir = Path(tmp_dir)

        # prepare code
        script_name = "aml_pass_runner.py"

        cur_dir = Path(__file__).resolve().parent
        code_root = tmp_dir / "code"
        code_files = [cur_dir / script_name]

        olive_config_path = self._create_olive_config_file(self.olive_config, tmp_dir)
        if olive_config_path:
            code_files.append(olive_config_path)

        self.copy_code(code_files, code_root)

        # prepare inputs
        inputs, args = {}, {}
        for name, config_json in [("model", model_config), ("pass", pass_config)]:
            name_inputs, name_args = self.create_inputs_and_args(
                name, config_json, tmp_dir, ignore_keys=["model_attributes"] if name == "model" else None
            )
            inputs.update(name_inputs)
            args.update(name_args)

        # prepare outputs
        outputs = {"pipeline_output": Output(type=AssetTypes.URI_FOLDER)}

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
            resources=self.resources,
            instance_count=self.instance_count,
            inputs=inputs,
            outputs=outputs,
            script_name=script_name,
        )

        @pipeline()
        def pass_runner_pipeline():
            outputs = {}
            component = cmd(**args)
            outputs["pipeline_output"] = component.outputs.pipeline_output
            return outputs

        return pass_runner_pipeline()

    def _run_job(
        self,
        ml_client,
        pipeline_job,
        experiment_name: str,
        tmp_dir: Union[str, Path],
        tags: Dict = None,
        output_name: str = None,
    ) -> Path:
        """Run a pipeline job and return the path to named-outputs."""
        # submit job
        logger.debug("Submitting pipeline")
        tags = {**self.tags, **(tags or {})}
        job = retry_func(
            ml_client.jobs.create_or_update,
            [pipeline_job],
            {"experiment_name": experiment_name, "tags": tags},
            max_tries=self.azureml_client_config.max_operation_retries,
            delay=self.azureml_client_config.operation_retry_interval,
            exceptions=(HttpResponseError, JobException),
        )
        logger.info("Pipeline submitted. Job name: %s. Job link: %s", job.name, job.studio_url)
        ml_client.jobs.stream(job.name)

        # get output
        output_dir = Path(tmp_dir) / "pipeline_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        # whether to download a single output or all outputs
        output_arg = {"download_path": output_dir}
        if output_name:
            output_arg["output_name"] = output_name
        else:
            output_arg["all"] = True
        logger.debug("Downloading pipeline output to %s", output_dir)
        retry_func(
            ml_client.jobs.download,
            [job.name],
            output_arg,
            max_tries=self.azureml_client_config.max_operation_retries,
            delay=self.azureml_client_config.operation_retry_interval,
            exceptions=ServiceResponseError,
        )

        return output_dir / "named-outputs"

    def _load_model(self, input_model_config: dict, output_model_path, pipeline_output_path):
        model_json_path = pipeline_output_path / "output_model_config.json"
        with model_json_path.open("r") as f:
            model_json = json.load(f)

        # set the resources that are the same as the input model
        same_resources_as_input = model_json.pop("same_resources_as_input")
        for resource_key in same_resources_as_input:
            set_dict_value(model_json, resource_key, get_dict_value(input_model_config, resource_key))

        # resolve resource names that are relative paths and save them to the output folder
        relative_resource_names = model_json.pop("resources")
        for resource_key in relative_resource_names:
            resource_json = get_dict_value(model_json, resource_key)
            # can only be local file or folder
            resource_type = resource_json["type"]
            assert resource_type in LOCAL_RESOURCE_TYPES, f"Expected local file or folder, got {resource_type}"
            # to be safe when downloading we will use the whole of output_model_path as a directory
            # and create subfolders for each resource
            # this is fine since the engine calls the system with a unique output_model_path which is a folder
            output_dir = Path(output_model_path).with_suffix("")
            # key is like ("config", "model_path")
            output_name = "--".join(map(str, resource_key)).replace("config--", "").replace("_path", "")
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
            set_dict_value(model_json, resource_key, output_resource_path)

        return ModelConfig(**model_json)

    def evaluate_model(
        self, model_config: ModelConfig, data_root: str, metrics: List["Metric"], accelerator: "AcceleratorSpec"
    ) -> MetricResult:
        if model_config.type.lower() == "SNPEModel".lower():
            raise NotImplementedError("SNPE model does not support azureml evaluation")
        if model_config.type.lower() == "OpenVINOModel".lower():
            raise NotImplementedError("OpenVINO model does not support azureml evaluation")

        with tempfile.TemporaryDirectory() as tempdir:
            ml_client = self.azureml_client_config.create_client()
            pipeline_job = self._create_pipeline_for_evaluation(
                tempdir, model_config.to_json(check_object=True), metrics, accelerator
            )

            # submit job
            named_outputs_dir = self._run_job(ml_client, pipeline_job, "olive-evaluation", tempdir)

            metric_results = {}
            for metric in metrics:
                metric_json = named_outputs_dir / metric.name / "metric_result.json"
                if metric_json.is_file():
                    with metric_json.open() as f:
                        metric_results.update(json.load(f))

            return MetricResult.parse_obj(metric_results)

    def _create_pipeline_for_evaluation(
        self,
        tmp_dir: str,
        model_config: dict,
        metrics: List["Metric"],
        accelerator: "AcceleratorSpec",
    ):
        tmp_dir = Path(tmp_dir)

        # model args
        model_inputs, model_args = self.create_inputs_and_args(
            "model", model_config, tmp_dir, ignore_keys=["model_attributes"]
        )

        accelerator_config_path: Path = tmp_dir / "accelerator_config.json"
        with accelerator_config_path.open("w") as f:
            json.dump(accelerator.to_json(), f, sort_keys=True)

        @pipeline
        def evaluate_pipeline():
            outputs = {}
            for metric in metrics:
                metric_tmp_dir = tmp_dir / metric.name
                metric_component = self._create_metric_component(
                    metric_tmp_dir,
                    metric,
                    model_inputs,
                    model_args,
                    accelerator_config_path,
                )
                outputs[metric.name] = metric_component.outputs.pipeline_output
            return outputs

        pipeline_job = evaluate_pipeline()
        pipeline_job.settings.default_compute = self.compute

        return pipeline_job

    def _create_metric_component(
        self,
        tmp_dir: Path,
        metric: "Metric",
        model_inputs: Dict[str, Input],
        model_args: Dict[str, Input],
        accelerator_config_path: str,
    ):
        metric_json = metric.to_json(check_object=True)

        # prepare code
        script_name = "aml_evaluation_runner.py"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        cur_dir = Path(__file__).resolve().parent
        code_root = tmp_dir / "code"
        code_files = [cur_dir / script_name]

        olive_config_path = self._create_olive_config_file(self.olive_config, tmp_dir)
        if olive_config_path:
            code_files.append(olive_config_path)

        self.copy_code(code_files, code_root)

        # prepare inputs
        metric_inputs, metric_args = self.create_inputs_and_args("metric", metric_json, tmp_dir)
        inputs = {**model_inputs, **metric_inputs, "accelerator_config": Input(type=AssetTypes.URI_FILE)}
        args = {
            **model_args,
            **metric_args,
            "accelerator_config": Input(type=AssetTypes.URI_FILE, path=accelerator_config_path),
        }

        # prepare outputs
        outputs = {"pipeline_output": Output(type=AssetTypes.URI_FOLDER)}

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
            resources=self.resources,
            instance_count=self.instance_count,
            inputs=inputs,
            outputs=outputs,
            script_name=script_name,
        )

        # metric component
        return cmd(**args)

    def copy_code(self, code_files: List, target_path: Path):
        target_path.mkdir(parents=True, exist_ok=True)
        for code_file in code_files:
            shutil.copy2(str(code_file), str(target_path))

        if self.is_dev:
            logger.warning(
                "Dev mode is only enabled for CI pipeline! "
                "It will overwrite the Olive package in AML computer with latest code."
            )
            cur_dir = Path(__file__).resolve().parent
            project_folder = cur_dir.parents[1]
            copy_dir(project_folder, target_path / "olive", ignore=shutil.ignore_patterns("__pycache__"))

    def remove(self):
        if self.temp_dirs:
            logger.info("AzureML system cleanup temp dirs.")
            for temp_dir in self.temp_dirs:
                temp_dir.cleanup()
            self.temp_dirs = []
