# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest

from olive.systems.docker.docker_system import DockerSystem
from olive.systems.system_config import DockerTargetUserConfig
from test.utils import ONNX_MODEL_PATH

# pylint: disable=attribute-defined-outside-init,protected-access


class TestDockerSystem:
    @staticmethod
    def get_default_docker_config():
        """Get default Docker configuration for tests."""
        return DockerTargetUserConfig(
            dockerfile="Dockerfile",
            build_context_path="build_context",
            image_name="test-image:latest",
            work_dir="/olive-ws",
        )

    @patch("olive.systems.docker.docker_system.docker.from_env")
    def test__init_image_exist(self, mock_from_env):
        # setup
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        mock_image = MagicMock()
        mock_docker_client.images.get.return_value = mock_image
        docker_config = self.get_default_docker_config()

        # execute
        docker_system = DockerSystem(
            image_name=docker_config.image_name,
            build_context_path=docker_config.build_context_path,
            dockerfile=docker_config.dockerfile,
            work_dir=docker_config.work_dir,
            build_args=docker_config.build_args,
            run_params=docker_config.run_params,
            clean_image=docker_config.clean_image,
        )

        # assert
        assert docker_system.image == mock_image
        mock_docker_client.images.get.assert_called_once_with(docker_config.image_name)

    @patch("olive.systems.docker.docker_system.docker.from_env")
    def test__init_image_dockerfile_build(self, mock_from_env):
        # setup
        import docker

        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        docker_config = self.get_default_docker_config()
        mock_docker_client.images.get.side_effect = [docker.errors.ImageNotFound("msg"), MagicMock()]
        mock_docker_client.api.build.return_value = [{"stream": "Successfully built\n"}]

        # execute
        DockerSystem(
            image_name=docker_config.image_name,
            build_context_path=docker_config.build_context_path,
            dockerfile=docker_config.dockerfile,
            work_dir=docker_config.work_dir,
            build_args=docker_config.build_args,
            run_params=docker_config.run_params,
            clean_image=docker_config.clean_image,
        )

        # assert
        mock_docker_client.api.build.assert_called_once()

    @patch("olive.systems.docker.docker_system.docker.from_env")
    @patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory")
    @patch("olive.systems.docker.docker_system.find_all_resources")
    def test_run_workflow(self, mock_find_resources, mock_tempdir, mock_from_env, tmp_path):
        """Test running a workflow in Docker container."""
        # Setup mocks
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        mock_tempdir.return_value.__enter__.return_value = str(tmp_path)
        mock_find_resources.return_value = {}

        # Mock Docker image
        mock_image = MagicMock()
        mock_docker_client.images.get.return_value = mock_image

        # Mock container run
        mock_container = MagicMock()
        mock_container.logs.return_value = iter([b"[Docker] Running workflow\n"])
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_docker_client.containers.run.return_value = mock_container

        # Create test config
        from olive.workflows.run.config import RunConfig

        run_config_dict = {
            "input_model": {"type": "ONNXModel", "config": {"model_path": str(ONNX_MODEL_PATH)}},
            "systems": {
                "docker_system": {
                    "type": "Docker",
                    "config": {
                        "dockerfile": "Dockerfile",
                        "build_context_path": "build_context",
                        "image_name": "test-image:latest",
                        "work_dir": "/olive-ws",
                    },
                },
                "local_system": {"type": "LocalSystem", "config": {}},
            },
            "engine": {"host": "docker_system", "target": "local_system", "output_dir": str(tmp_path / "output")},
        }

        run_config = RunConfig(**run_config_dict)

        # Create DockerSystem
        docker_config = self.get_default_docker_config()
        docker_system = DockerSystem(
            image_name=docker_config.image_name,
            build_context_path=docker_config.build_context_path,
            dockerfile=docker_config.dockerfile,
            work_dir=docker_config.work_dir,
            clean_image=False,
        )

        # Run workflow
        docker_system.run_workflow(run_config)

        # Verify container was run
        mock_docker_client.containers.run.assert_called_once()

        # Verify command includes workflow runner
        command = mock_docker_client.containers.run.call_args[1]["command"]
        assert "workflow_runner.py" in command
        assert "--config" in command

        # Verify cleanup
        mock_container.remove.assert_called_once()

    @patch("olive.systems.docker.docker_system.docker.from_env")
    @patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory")
    @patch("olive.systems.docker.docker_system.find_all_resources")
    def test_run_workflow_container_error(self, mock_find_resources, mock_tempdir, mock_from_env, tmp_path):
        """Test workflow run with container error."""
        # Setup mocks
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        mock_tempdir.return_value.__enter__.return_value = str(tmp_path)
        mock_find_resources.return_value = {}

        # Mock Docker image
        mock_image = MagicMock()
        mock_docker_client.images.get.return_value = mock_image

        # Mock container with error
        mock_container = MagicMock()
        mock_container.logs.return_value = iter([b"[Docker] Error occurred\n"])
        mock_container.wait.return_value = {"StatusCode": 1}  # Non-zero exit code
        mock_container.remove = MagicMock()

        mock_docker_client.containers.run.return_value = mock_container

        # Create test config
        from olive.workflows.run.config import RunConfig

        run_config_dict = {
            "input_model": {"type": "ONNXModel", "config": {"model_path": str(ONNX_MODEL_PATH)}},
            "systems": {
                "docker_system": {
                    "type": "Docker",
                    "config": {
                        "dockerfile": "Dockerfile",
                        "build_context_path": "build_context",
                        "image_name": "test-image:latest",
                        "work_dir": "/olive-ws",
                    },
                },
                "local_system": {"type": "LocalSystem", "config": {}},
            },
            "engine": {"host": "docker_system", "target": "local_system", "output_dir": str(tmp_path / "output")},
        }

        run_config = RunConfig(**run_config_dict)

        # Create DockerSystem
        docker_config = self.get_default_docker_config()
        docker_system = DockerSystem(
            image_name=docker_config.image_name,
            build_context_path=docker_config.build_context_path,
            dockerfile=docker_config.dockerfile,
            work_dir=docker_config.work_dir,
            build_args=docker_config.build_args,
            run_params=docker_config.run_params,
            clean_image=False,
        )

        # Run workflow and expect error
        from docker.errors import ContainerError

        with pytest.raises(ContainerError, match="Docker container run failed"):
            docker_system.run_workflow(run_config)

        # Verify container was still removed
        mock_container.remove.assert_called_once()

    @patch("olive.systems.docker.docker_system.docker.from_env")
    def test_remove_image(self, mock_from_env):
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        mock_image = MagicMock()
        mock_image.tags = ["test-image:latest"]
        mock_docker_client.images.get.return_value = mock_image

        docker_config = self.get_default_docker_config()
        docker_system = DockerSystem(
            image_name=docker_config.image_name,
            build_context_path=docker_config.build_context_path,
            dockerfile=docker_config.dockerfile,
            work_dir=docker_config.work_dir,
        )

        docker_system.remove()

        mock_docker_client.images.remove.assert_called_once_with("test-image:latest", force=True)
