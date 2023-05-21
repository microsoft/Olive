# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import BuildContext, Environment
from azure.core.exceptions import HttpResponseError, ServiceResponseError

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import validate_config
from olive.common.utils import retry_func
from olive.constants import Framework
from olive.evaluator.metric import Metric
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig, ModelStorageKind, OliveModel, ONNXModel
from olive.passes.olive_pass import Pass
from olive.systems.common import AzureMLDockerConfig, SystemType
from olive.systems.olive_system import OliveSystem

logger = logging.getLogger(__name__)


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
        raise Exception("Please specify DockerConfig.")

    def _assert_not_none(self, object):
        if object is None:
            raise Exception(f"{object.__class__.__name__} is missing in the inputs!")

    def run_pass(
        self,
        the_pass: Pass,
        model: OliveModel,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """
        ml_client = self.azureml_client_config.create_client()
        point = point or {}
        config = the_pass.config_at_search_point(point)
        pass_config = the_pass.to_json(check_objects=True)
        pass_config["config"].update(the_pass.serialize_config(config, check_objects=True))

        with tempfile.TemporaryDirectory() as tempdir:
            pipeline_job = self._create_pipeline_for_pass(tempdir, model, pass_config, the_pass.path_params)

            # submit job
            logger.debug("Submitting pipeline")
            job = retry_func(
                ml_client.jobs.create_or_update,
                [pipeline_job],
                {"experiment_name": "olive-pass", "tags": {"Pass": pass_config["type"]}},
                max_tries=3,
                delay=5,
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
                max_tries=3,
                delay=5,
                exceptions=ServiceResponseError,
            )

            pipeline_output_path = output_dir / "named-outputs" / "pipeline_output"

            return self._load_model(model, output_model_path, pipeline_output_path)

    def _create_model_inputs(self, model_storage_kind: ModelStorageKind):
        return {
            "model_config": Input(type=AssetTypes.URI_FILE),
            # aml supports uploading file/folder even though model_path is typed as URI_FOLDER
            "model_path": Input(type=AssetTypes.CUSTOM_MODEL)
            if model_storage_kind == ModelStorageKind.AzureMLModel
            else Input(type=AssetTypes.URI_FOLDER, optional=True),
            "model_script": Input(type=AssetTypes.URI_FILE, optional=True),
            "model_script_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
        }

    def _create_model_args(self, model_json: dict, tmp_dir: Path):
        # TODO: consider symlinking model_script and model_script_dir also when we decide
        # the relationship between the two
        model_script = None
        if model_json["config"].get("model_script"):
            model_script = Input(type=AssetTypes.URI_FILE, path=model_json["config"]["model_script"])
            model_json["config"]["model_script"] = None

        model_script_dir = None
        if model_json["config"].get("script_dir"):
            model_script_dir = Input(type=AssetTypes.URI_FOLDER, path=model_json["config"]["script_dir"])
            model_json["config"]["script_dir"] = None

        model_path = None
        if model_json["config"]["model_storage_kind"] == ModelStorageKind.AzureMLModel:
            model_path = Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=model_json["config"]["model_path"],
            )
            model_json["config"]["model_storage_kind"] = str(ModelStorageKind.LocalFile)
            model_json["config"]["version"] = None
        else:
            if model_json["config"].get("model_path"):
                original_model_path = Path(model_json["config"]["model_path"]).resolve()
                if (
                    model_json["type"].lower() == "onnxmodel"
                    and model_json["config"]["model_storage_kind"] == ModelStorageKind.LocalFolder
                ):
                    # onnx model with external data
                    # need to upload the parent directory of .onnx file
                    original_model_path = original_model_path.parent
                # use common name "model" for model_path
                tmp_model_path = (tmp_dir / "model").with_suffix(original_model_path.suffix)
                if original_model_path.is_dir():
                    # copy model directory
                    # symlink doesn't work for directory
                    shutil.copytree(original_model_path, tmp_model_path, symlinks=True)
                    if model_json["type"].lower() == "onnxmodel":
                        # rename .onnx file to model.onnx
                        onnx_model_file = Path(model_json["config"]["model_path"]).resolve().name
                        (tmp_model_path / onnx_model_file).rename(tmp_model_path / "model.onnx")
                else:
                    # symlink model file
                    tmp_model_path.symlink_to(original_model_path)
                model_path = Input(
                    type=AssetTypes.URI_FILE
                    if model_json["config"].get("model_storage_kind") == ModelStorageKind.LocalFile
                    else AssetTypes.URI_FOLDER,
                    path=tmp_model_path,
                )
        model_json["config"]["model_path"] = None

        model_config_path = tmp_dir / "model_config.json"
        with model_config_path.open("w") as f:
            json.dump(model_json, f, sort_keys=True)
        model_config = Input(type=AssetTypes.URI_FILE, path=model_config_path)

        return {
            "model_config": model_config,
            "model_path": model_path,
            "model_script": model_script,
            "model_script_dir": model_script_dir,
        }

    def _create_pass_inputs(self, pass_path_params: List[Tuple[str, bool]]):
        inputs = {"pass_config": Input(type=AssetTypes.URI_FILE)}
        for param, required in pass_path_params:
            # aml supports uploading file/folder even though this is typed as URI_FOLDER
            inputs[f"pass_{param}"] = Input(type=AssetTypes.URI_FOLDER, optional=not required)

        return inputs

    def _create_pass_args(self, pass_config: dict, pass_path_params: List[Tuple[str, bool]], tmp_dir: Path):
        pass_args = {}
        for param, _ in pass_path_params:
            if pass_config["config"].get(param) is None:
                continue
            pass_args[f"pass_{param}"] = Input(
                type=AssetTypes.URI_FILE if Path(pass_config["config"][param]).is_file() else AssetTypes.URI_FOLDER,
                path=pass_config["config"][param],
            )
            pass_config["config"][param] = None

        pass_config_path = tmp_dir / "pass_config.json"
        with pass_config_path.open("w") as f:
            json.dump(pass_config, f, sort_keys=True)

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
            code=code,
            inputs=inputs,
            outputs=dict(pipeline_output=Output(type=AssetTypes.URI_FOLDER)),
            instance_count=instance_count,
            compute=compute,
        )

        return component

    def _create_pipeline_for_pass(
        self,
        tmp_dir,
        model: OliveModel,
        pass_config: dict,
        pass_path_params: List[Tuple[str, bool]],
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
                "This mode is only enabled for CI pipeline! "
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
            **self._create_model_inputs(model.model_storage_kind),
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
            code=code_root,
            compute=self.compute,
            instance_count=self.instance_count,
            inputs=inputs,
            script_name=script_name,
        )

        # model json
        model_json = model.to_json(check_object=True)

        # input argument values
        args = {
            **self._create_model_args(model_json, tmp_dir),
            **self._create_pass_args(pass_config, pass_path_params, tmp_dir),
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

        # resolve model path
        # this is to handle passes like OrtPerfTuning that use the same model file as input
        same_model_path_as_input = model_json.pop("same_model_path_as_input")
        model_path = None
        if same_model_path_as_input:
            model_path = input_model.model_path
            model_json["config"].update(
                {
                    "name": input_model.name,
                    "version": input_model.version,
                    "model_storage_kind": input_model.model_storage_kind,
                }
            )
        elif model_json["config"]["model_path"] is not None:
            downloaded_model_path = pipeline_output_path / model_json["config"]["model_path"]
            if model_json["type"].lower() == "onnxmodel":
                # onnx model can have external data
                output_model_path = ONNXModel.resolve_path(output_model_path)
                if model_json["config"]["model_storage_kind"] == ModelStorageKind.LocalFolder:
                    # has external data
                    # copy the .onnx file along with external data files
                    shutil.copytree(downloaded_model_path.parent, Path(output_model_path).parent, dirs_exist_ok=True)
                    # rename the .onnx file to the output_model_path, the downloaded model has "model.onnx" name
                    (Path(output_model_path).parent / downloaded_model_path.name).rename(output_model_path)
                else:
                    # no external data
                    # just copy over the .onnx file
                    shutil.copy(downloaded_model_path, output_model_path)
            else:
                # handle other model types
                if Path(output_model_path).suffix != Path(downloaded_model_path).suffix:
                    output_model_path += Path(downloaded_model_path).suffix
                if downloaded_model_path.is_file():
                    # model is a file
                    shutil.copy(downloaded_model_path, output_model_path)
                else:
                    # model is a directory
                    shutil.copytree(downloaded_model_path, output_model_path, dirs_exist_ok=True)
            model_path = output_model_path
        model_json["config"]["model_path"] = model_path
        return ModelConfig(**model_json).create_model()

    def _create_metric_inputs(self):
        return {
            "metric_config": Input(type=AssetTypes.URI_FILE),
            "metric_user_script": Input(type=AssetTypes.URI_FILE, optional=True),
            "metric_script_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
            "metric_data_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
        }

    def _create_metric_args(self, metric_config: dict, tmp_dir: Path) -> Tuple[List[str], dict]:
        metric_config["name"] = "result"
        metric_config.pop("goal", None)
        metric_user_script = metric_config["user_config"]["user_script"]
        if metric_user_script:
            metric_user_script = Input(type=AssetTypes.URI_FILE, path=metric_user_script)
            metric_config["user_config"]["user_script"] = None

        metric_script_dir = metric_config["user_config"]["script_dir"]
        if metric_script_dir:
            metric_script_dir = Input(type=AssetTypes.URI_FOLDER, path=metric_script_dir)
            metric_config["user_config"]["script_dir"] = None

        metric_data_dir = metric_config["user_config"]["data_dir"]
        if metric_data_dir:
            metric_data_dir = Input(type=AssetTypes.URI_FOLDER, path=metric_data_dir)
            metric_config["user_config"]["data_dir"] = None

        metric_config_path = tmp_dir / "metric_config.json"
        with metric_config_path.open("w") as f:
            json.dump(metric_config, f, sort_keys=True)
        metric_config = Input(type=AssetTypes.URI_FILE, path=metric_config_path)

        return {
            "metric_config": metric_config,
            "metric_user_script": metric_user_script,
            "metric_script_dir": metric_script_dir,
            "metric_data_dir": metric_data_dir,
        }

    def evaluate_model(self, model: OliveModel, metrics: List[Metric], accelerator: AcceleratorSpec) -> Dict[str, Any]:
        if model.framework == Framework.SNPE:
            raise NotImplementedError("SNPE model does not support azureml evaluation")
        if model.framework == Framework.OPENVINO:
            raise NotImplementedError("OpenVINO model does not support azureml evaluation")

        with tempfile.TemporaryDirectory() as tempdir:
            ml_client = self.azureml_client_config.create_client()
            pipeline_job = self._create_pipeline_for_evaluation(tempdir, model, metrics, accelerator)

            # submit job
            logger.debug("Submitting pipeline")
            job = retry_func(
                ml_client.jobs.create_or_update,
                [pipeline_job],
                {"experiment_name": "olive-evaluation"},
                max_tries=3,
                delay=5,
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
                max_tries=3,
                delay=5,
                exceptions=ServiceResponseError,
            )

            metric_results = {}
            for metric in metrics:
                metric_json = output_dir / "named-outputs" / metric.name / "metric_result.json"
                if metric_json.is_file():
                    with metric_json.open() as f:
                        metric_results[metric.name] = json.load(f)["result"]

            return metric_results

    def _create_pipeline_for_evaluation(
        self, tmp_dir: str, model: OliveModel, metrics: List[Metric], accelerator: AcceleratorSpec
    ):
        tmp_dir = Path(tmp_dir)

        # model json
        model_json = model.to_json(check_object=True)

        # model args
        model_args = self._create_model_args(model_json, tmp_dir)

        accelerator_config_path: Path = tmp_dir / "accelerator.json"
        with accelerator_config_path.open("w") as f:
            json.dump(accelerator.to_json(), f, sort_keys=True)

        @pipeline
        def evaluate_pipeline():
            outputs = {}
            for metric in metrics:
                metric_tmp_dir = tmp_dir / metric.name
                metric_component = self._create_metric_component(
                    metric_tmp_dir, metric, model_args, model.model_storage_kind, accelerator_config_path
                )
                outputs[metric.name] = metric_component.outputs.pipeline_output
            return outputs

        pipeline_job = evaluate_pipeline()
        pipeline_job.settings.default_compute = self.compute

        return pipeline_job

    def _create_metric_component(
        self,
        tmp_dir: Path,
        metric: Metric,
        model_args: Dict[str, Input],
        model_storage_kind: ModelStorageKind,
        accelerator_config_path: str,
    ):
        metric_json = metric.to_json(check_objects=True)

        # prepare code
        script_name = "aml_evaluation_runner.py"
        cur_dir = Path(__file__).resolve().parent
        code_file = cur_dir / script_name
        code_root = tmp_dir / "code"
        code_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(code_file), str(code_root))
        if self.is_dev:
            logger.warning(
                "This mode is only enabled for CI pipeline! "
                + "It will overwrite the Olive package in AML computer with latest code."
            )
            project_folder = cur_dir.parent.parent
            shutil.copytree(project_folder, code_root / "olive", ignore=shutil.ignore_patterns("__pycache__"))

        # prepare inputs
        inputs = {
            **self._create_model_inputs(model_storage_kind),
            **self._create_metric_inputs(),
            **{"accelerator_config": Input(type=AssetTypes.URI_FILE)},
        }

        # metric type
        metric_type = metric_json["type"]
        if metric_json["sub_type"] is not None:
            metric_type = f"{metric_type}-{metric_json['sub_type']}"

        # aml command object
        cmd = self._create_step(
            name=metric_type,
            display_name=metric_type,
            description=f"Run olive {metric_type} evaluation",
            aml_environment=self.environment,
            code=code_root,
            compute=self.compute,
            instance_count=self.instance_count,
            inputs=inputs,
            script_name=script_name,
        )

        # input argument values
        args = {
            **model_args,
            **self._create_metric_args(metric_json, tmp_dir),
            **{"accelerator_config": Input(type=AssetTypes.URI_FILE, path=accelerator_config_path)},
        }

        # metric component
        metric_component = cmd(**args)

        return metric_component
