# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import json
import shutil
from pathlib import Path
from test.unit_test.utils import ONNX_MODEL_PATH, get_accuracy_metric, get_onnx_model_config, get_pytorch_model_config
from unittest.mock import ANY, MagicMock, patch

import pytest

from olive.evaluator.metric import AccuracySubType
from olive.evaluator.metric_result import joint_metric_key
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.perf_tuning import OrtPerfTuning
from olive.systems.common import LocalDockerConfig
from olive.systems.docker.docker_system import DockerSystem
from olive.systems.system_config import DockerTargetUserConfig, SystemConfig
from olive.systems.utils import create_managed_system

# pylint: disable=attribute-defined-outside-init,protected-access


class TestDockerSystem:
    @patch("olive.systems.docker.docker_system.docker.from_env")
    def test__init_image_exist(self, mock_from_env):
        # setup
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        mock_image = MagicMock()
        mock_docker_client.images.get.return_value = mock_image
        docker_config = LocalDockerConfig(
            image_name="image_name", build_context_path="build_context_path", dockerfile="dockerfile"
        )

        # execute
        docker_system = DockerSystem(docker_config, is_dev=True)

        # assert
        assert docker_system.image == mock_image
        mock_docker_client.images.get.assert_called_once_with(docker_config.image_name)

    @patch("olive.systems.docker.docker_system.shutil.copy2")
    @patch("olive.systems.docker.docker_system.shutil.copytree")
    @patch("olive.systems.docker.docker_system.docker.from_env")
    @patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory")
    def test__init_image_dockerfile_build(self, mock_tempdir, mock_from_env, mock_copytree, mock_copy2, tmpdir):
        # setup
        mock_tempdir.return_value.__enter__.return_value = tmpdir

        import docker

        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        docker_config = LocalDockerConfig(
            image_name="image_name", build_context_path="build_context_path", dockerfile="dockerfile"
        )
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("msg")
        mock_docker_client.images.build.return_value = (MagicMock(), [{"stream": "Successfully built mock_image_id"}])

        # execute
        DockerSystem(docker_config, is_dev=True)

        # assert
        expected_build_context_path = tmpdir
        mock_docker_client.images.build.assert_called_once_with(
            path=expected_build_context_path,
            dockerfile=docker_config.dockerfile,
            tag=docker_config.image_name,
            buildargs=docker_config.build_args,
        )

    @patch("olive.systems.docker.docker_system.shutil.copy2")
    @patch("olive.systems.docker.docker_system.shutil.copyfile")
    @patch("olive.systems.docker.docker_system.docker.from_env")
    @patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory")
    def test__init_image_requirements_file_build(self, mock_tempdir, mock_from_env, mock_copyfile, mock_copy, tmpdir):
        # setup
        import docker

        tempdir = tmpdir
        mock_tempdir.return_value.__enter__.return_value = tempdir
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        docker_config = LocalDockerConfig(
            image_name="image_name",
        )
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("msg")
        mock_docker_client.images.build.return_value = (MagicMock(), [{"stream": "Successfully built mock_image_id"}])

        # execute
        docker_system = DockerSystem(docker_config, is_dev=True, requirements_file="requirements_file")

        # assert
        mock_docker_client.images.build.assert_called_once_with(
            path=tempdir,
            dockerfile=docker_system.BASE_DOCKERFILE,
            tag=docker_config.image_name,
            buildargs=docker_config.build_args,
        )

    @pytest.fixture()
    def mock_docker_system_info(self):
        self.mock_from_env = patch("olive.systems.docker.docker_system.docker.from_env").start()
        self.mock_tempdir = patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory").start()
        self.mock_create_eval_config = patch(
            "olive.systems.docker.docker_system.DockerSystem._create_eval_config"
        ).start()
        self.mock_copy = patch("olive.systems.docker.docker_system.copy.deepcopy").start()
        yield
        patch.stopall()

    @pytest.mark.usefixtures("mock_docker_system_info")
    @pytest.mark.parametrize("exit_code", [0, 1])
    def test_evaluate_model(
        self,
        exit_code,
        tmp_path,
    ):
        # setup
        import docker

        output_local_path = Path(__file__).absolute().parent / "output_local_path"
        eval_output_path = "eval_output"
        eval_output_name = "eval_res.json"

        def container_run(image, command=None, stdout=True, stderr=False, remove=False, **kwargs):
            container_obj = MagicMock()
            container_obj.wait.return_value = {"StatusCode": exit_code}
            container_obj.logs.return_value = [b"mock_error"] if exit_code != 0 else []
            shutil.copy2(output_local_path / eval_output_name, tmp_path / eval_output_path / eval_output_name)
            return container_obj

        mock_docker_client = MagicMock()
        mock_docker_client.containers.run.side_effect = container_run
        self.mock_from_env.return_value = mock_docker_client
        self.mock_tempdir.return_value.__enter__.return_value = str(tmp_path)

        self.mock_create_eval_config.return_value = {"metrics": [], "model": {"config": {}}}
        self.mock_copy.side_effect = lambda x: x

        model_config = get_pytorch_model_config()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        docker_config = LocalDockerConfig(
            image_name="image_name", build_context_path="build_context_path", dockerfile="dockerfile"
        )
        docker_system = DockerSystem(docker_config, is_dev=True)
        data_root = None

        if exit_code != 0:
            with pytest.raises(
                docker.errors.ContainerError,
                match=r".*returned non-zero exit status 1: Docker container evaluation failed with: mock_error",
            ):
                _ = docker_system.evaluate_model(model_config, data_root, [metric], DEFAULT_CPU_ACCELERATOR)
        else:
            actual_res = docker_system.evaluate_model(model_config, data_root, [metric], DEFAULT_CPU_ACCELERATOR)

            container_root_path = Path("/olive-ws/")
            eval_local_path = Path(__file__).resolve().parents[4] / "olive" / "systems" / "docker" / "eval.py"
            volumes_list = [
                f"{eval_local_path}:{container_root_path / 'eval.py'}",  # eval script
                f"{tmp_path / 'olive'}:{container_root_path / 'olive'}",  # olive dev
                f"{tmp_path / 'config.json'}:{container_root_path / 'config.json'}",  # config
                f"{tmp_path / eval_output_path}:{container_root_path / eval_output_path}",  # output
            ]
            eval_command = [
                "python",
                str(container_root_path / "eval.py"),
                "--config",
                str(container_root_path / "config.json"),
                "--output_path",
                str(container_root_path / eval_output_path),
                "--output_name",
                eval_output_name,
                "--accelerator_type",
                "cpu",
                "--execution_provider",
                "CPUExecutionProvider",
            ]
            eval_command = " ".join(eval_command)
            mock_docker_client.containers.run.assert_called_once_with(
                image=docker_system.image,
                command=eval_command,
                volumes=volumes_list,
                detach=True,
                environment=ANY,
            )

            for sub_type in metric.sub_types:
                joint_key = joint_metric_key(metric.name, sub_type.name)
                assert actual_res[joint_key].value == 0.99618

    @patch("olive.systems.docker.docker_system.docker.from_env")
    @patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory")
    def test_run_pass(self, mock_tempdir, mock_from_env, tmp_path):
        runner_output_path = "runner_output"
        runner_output_name = "runner_res.json"

        exit_code = 0
        onnx_model = get_onnx_model_config()
        container_root_path = Path("/olive-ws/")

        def container_run(image, command=None, stdout=True, stderr=False, remove=False, **kwargs):
            container_obj = MagicMock()
            container_obj.wait.return_value = {"StatusCode": exit_code}
            container_obj.logs.return_value = [b"mock_error"] if exit_code != 0 else []
            with (tmp_path / runner_output_path / runner_output_name).open("w") as f:
                output_model = copy.deepcopy(onnx_model)
                output_model.config["model_path"] = str(container_root_path / ONNX_MODEL_PATH.name)
                json.dump(output_model.to_json(), f, indent=4)
            return container_obj

        mock_docker_client = MagicMock()
        mock_docker_client.containers.run.side_effect = container_run
        mock_from_env.return_value = mock_docker_client
        mock_tempdir.return_value.__enter__.return_value = str(tmp_path)
        # mock_copy.side_effect = lambda x: x

        docker_config = LocalDockerConfig(
            image_name="image_name", build_context_path="build_context_path", dockerfile="dockerfile"
        )
        docker_system = DockerSystem(docker_config, is_dev=True)
        data_root = "data_root"

        p = create_pass_from_dict(OrtPerfTuning, {"data_dir": "data_dir"}, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        def validate_file_or_folder(v, values, **kwargs):
            return v

        def is_dir_mock(self):
            return self == Path("data_root") / "data_dir"

        with patch("olive.resource_path._validate_file_path", side_effect=validate_file_or_folder), patch(
            "olive.resource_path._validate_folder_path", side_effect=validate_file_or_folder
        ), patch("olive.resource_path._validate_path", side_effect=validate_file_or_folder), patch.object(
            Path, "is_dir", side_effect=is_dir_mock, autospec=True
        ):
            output_model = docker_system.run_pass(p, onnx_model, data_root, output_folder)
            assert output_model is not None

        runner_local_path = Path(__file__).resolve().parents[4] / "olive" / "systems" / "docker" / "runner.py"
        model_path = onnx_model.config["model_path"]
        data_dir = str(p.config["data_dir"])
        volumes_list = [
            f"{runner_local_path}:{container_root_path / 'runner.py'}",  # runner script
            f"{tmp_path / 'olive'}:{container_root_path / 'olive'}",  # olive dev
            f"{Path(model_path).resolve()}:{container_root_path / Path(model_path).name}",  # model
            f"{Path(data_root) / data_dir}:{container_root_path / data_dir}",
            f"{tmp_path / 'config.json'}:{container_root_path / 'config.json'}",  # config
            f"{tmp_path / runner_output_path}:{container_root_path / runner_output_path}",  # output
        ]
        runner_command = [
            "python",
            str(container_root_path / "runner.py"),
            "--config",
            str(container_root_path / "config.json"),
            "--output_path",
            str(container_root_path / runner_output_path),
            "--output_name",
            runner_output_name,
        ]
        runner_command = " ".join(runner_command)
        mock_docker_client.containers.run.assert_called_once_with(
            image=docker_system.image,
            command=runner_command,
            volumes=volumes_list,
            detach=True,
            environment=ANY,
        )

    def test_runner_entry(self, tmp_path):
        from olive.systems.docker import utils as docker_utils
        from olive.systems.docker.runner import runner_entry as docker_runner_entry

        p = create_pass_from_dict(OrtPerfTuning, {"data_dir": "data_dir"}, disable_search=True)
        pass_config = p.to_json(check_object=True)
        config = p.config_at_search_point({})
        pass_config["config"].update(p.serialize_config(config, check_object=True))

        onnx_model = get_onnx_model_config()

        container_root_path = Path("/olive-ws/")
        data = DockerSystem._create_runner_config(
            onnx_model,
            pass_config,
            {"model_path": onnx_model.config["model_path"]},
            {"data_dir": str(container_root_path / "data_dir")},
        )
        docker_utils.create_config_file(tmp_path, data, container_root_path)
        with patch.object(OrtPerfTuning, "run", return_value=onnx_model):
            docker_runner_entry(str(tmp_path / "config.json"), str(tmp_path), "runner_res.json")
            assert (tmp_path / "runner_res.json").exists()

    @patch("olive.systems.docker.docker_system.docker.from_env")
    def test_managed_env(self, mock_from_env):
        import docker

        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("msg")
        mock_docker_client.images.build.return_value = (MagicMock(), [{"stream": "Successfully built mock_image_id"}])

        system_config = SystemConfig(
            type="Docker",
            config=DockerTargetUserConfig(
                accelerators=[{"device": "cpu"}],
                olive_managed_env=True,
                is_dev=True,
            ),
        )
        target = create_managed_system(system_config, DEFAULT_CPU_ACCELERATOR)
        assert target.config.olive_managed_env

        host_system = create_managed_system(system_config, None)
        assert host_system.config.olive_managed_env
