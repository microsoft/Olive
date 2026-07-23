# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

from olive.cli.base import (
    BaseOliveCLICommand,
    _flatten_test_metrics,
    add_discrepancy_check_pass,
    add_hf_test_model_config,
    add_input_model_options,
    add_logging_options,
    add_telemetry_options,
    get_input_model_config,
    mark_test_output_path,
    save_discrepancy_check_results,
    validate_test_output_path,
)
from olive.telemetry import action


class WorkflowRunCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser("run", help="Run an olive workflow")
        sub_parser.add_argument("--run-config", "--config", type=str, help="Path to json config file", required=True)
        sub_parser.add_argument(
            "--list_required_packages", help="List packages required to run the workflow", action="store_true"
        )
        sub_parser.add_argument(
            "--tempdir", type=str, help="Root directory for tempfile directories and files", required=False
        )
        sub_parser.add_argument(
            "--package-config",
            type=str,
            required=False,
            help=(
                "For advanced users. Path to optional package (json) config file with location "
                "of individual pass module implementation and corresponding dependencies. "
                "Configuration might also include user owned/proprietary/private pass implementations."
            ),
        )
        add_logging_options(sub_parser, default=None)
        add_input_model_options(
            sub_parser.add_argument_group("Model options (not required)"),
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            required=False,
        )
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=WorkflowRunCommand)

    @action
    def run(self):
        from olive.common.config_utils import load_config_file
        from olive.workflows import run as olive_run

        # allow the run_config to be a dict already (for api use)
        run_config = self.args.run_config
        if not isinstance(run_config, dict):
            run_config = load_config_file(run_config)
        if "builds" in run_config and self.args.test not in (None, False):
            raise ValueError("--test is not supported with multi-build run configurations.")
        if "builds" in run_config and self.args.output_path is not None:
            raise ValueError(
                "--output_path is not supported with multi-build run configurations. "
                "Use each build's output_dir or its default output/<build_name> directory instead."
            )
        if input_model_config := get_input_model_config(self.args, required=False):
            print("Replacing input model config in run config")
            run_config["input_model"] = input_model_config
        elif self.args.test not in (None, False):
            input_model = run_config.get("input_model")
            if not isinstance(input_model, dict) or input_model.get("type", "").lower() != "hfmodel":
                raise ValueError("--test for olive run requires a Hugging Face input_model in the run config.")
            output_path = (
                self.args.output_path or run_config.get("output_dir") or run_config.get("engine", {}).get("output_dir")
            )
            validate_test_output_path(output_path, self.args.test)
            run_config["input_model"] = add_hf_test_model_config(input_model, self.args.test, output_path)
            test_metrics = _flatten_test_metrics(getattr(self.args, "test_metrics", None))
            run_config = add_discrepancy_check_pass(run_config, test_metrics)

        for arg_key, rc_key in [("output_path", "output_dir"), ("log_level", "log_severity_level")]:
            if (arg_value := getattr(self.args, arg_key)) is not None:
                print(f"Replacing {rc_key} in run config with {arg_value}")
                # remove value from engine config if it exists
                run_config.get("engine", {}).pop(rc_key, None)
                # add value to run config directly
                run_config[rc_key] = arg_value

        output_path = run_config.get("output_dir") or run_config.get("engine", {}).get("output_dir")
        workflow_output = olive_run(
            run_config,
            list_required_packages=self.args.list_required_packages,
            tempdir=self.args.tempdir,
            package_config=self.args.package_config,
        )
        if self.args.test not in (None, False):
            mark_test_output_path(output_path)
            save_discrepancy_check_results(workflow_output, output_path)

        if self.args.list_required_packages is True:
            print("Required packages listed!")
        elif isinstance(workflow_output, dict):
            self._print_build_outputs(run_config, workflow_output)

        return workflow_output

    @staticmethod
    def _print_build_outputs(run_config: dict, workflow_outputs: dict) -> None:
        from olive.workflows.run.builds import get_build_output_dir

        builds = run_config.get("builds") or {}
        build_default = builds.get("_default") or {}
        for build_name, workflow_output in workflow_outputs.items():
            if workflow_output is None or not workflow_output.has_output_model():
                print(f"Build {build_name!r}: no output model produced. Please check the log for details.")
                continue
            configured_output_dir = (builds.get(build_name) or {}).get("output_dir") or build_default.get("output_dir")
            output_dir = get_build_output_dir(build_name, configured_output_dir)
            print(f"Build {build_name!r}: model is saved under {output_dir}")
