# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

import json
import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Dict

import yaml

from olive.auto_optimizer.template_mapping import PERF_TUNING_TEMPLATE
from olive.cli.base import (
    BaseOliveCLICommand,
    add_logging_options,
    add_model_options,
    add_remote_options,
    get_input_model_config,
    get_output_model_number,
    is_remote_run,
    update_remote_option,
)
from olive.common.utils import set_nested_dict_value, set_tempdir
from olive.data.config import DataConfig
from olive.workflows import run as olive_run


class PerfTuningCommand(BaseOliveCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "tune-session-params",
            help=(
                "Automatically tune the session parameters for a given onnx model. "
                "Currently, for onnx model converted from huggingface model and used for "
                "generative tasks, user can simply provide the --model onnx_model_path "
                "--hf_model_name hf_model_name --device device_type to get the tuned session parameters."
            ),
        )
        add_logging_options(sub_parser)

        # model options
        add_model_options(sub_parser)

        # dataset options
        dataset_group = sub_parser.add_argument_group(
            "dataset options, which mutually exclusive with huggingface dataset options"
        )
        dataset_group.add_argument(
            "--data_config_path",
            type=str,
            help="Path to the data config file. It allows to customize the data config(json/yaml) for the model.",
        )

        hf_dataset_group = sub_parser.add_argument_group(
            "huggingface dataset options, if dataset options are not provided, "
            "user should provide the following options to modify the default data config. "
            "Please refer to olive.data.container.TransformersTokenDummyDataContainer for more details."
        )
        hf_dataset_group.add_argument(
            "--predict_with_kv_cache",
            action="store_true",
            help="Whether to use key-value cache for perf_tuning",
        )
        hf_dataset_group.add_argument(
            "--hf_model_name",
            help="Huggingface model name used to load model configs from huggingface.",
        )
        hf_dataset_group.add_argument(
            "--batch_size",
            type=int,
            help="Batch size of the input data.",
        )
        hf_dataset_group.add_argument(
            "--seq_len",
            type=int,
            help="Sequence length to use for the input data.",
        )
        hf_dataset_group.add_argument(
            "--past_seq_len",
            type=int,
            help="Past sequence length to use for the input data.",
        )
        hf_dataset_group.add_argument(
            "--max_seq_len",
            type=int,
            help="Max sequence length to use for the input data.",
        )
        hf_dataset_group.add_argument(
            "--shared_kv",
            action="store_true",
            help="Whether to enable share kv cache in the input data.",
        )
        hf_dataset_group.add_argument(
            "--generative",
            action="store_true",
            help="Whether to enable generative mode in the input data.",
        )
        hf_dataset_group.add_argument(
            "--ort_past_key_name",
            type=str,
            help="Past key name for the input data.",
        )
        hf_dataset_group.add_argument(
            "--ort_past_value_name",
            type=str,
            help="Past value name for the input data.",
        )
        hf_dataset_group.add_argument(
            "--max_samples",
            type=int,
            help="Max samples to use for the input data.",
        )
        hf_dataset_group.add_argument(
            "--fields_no_batch",
            nargs="*",
            help="List of fields that should not be batched.",
        )

        # pass options
        pass_group = sub_parser.add_argument_group("pass options")

        pass_group.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["gpu", "cpu"],
            help="Device to use for the model.",
        )
        pass_group.add_argument(
            "--cpu_cores",
            type=int,
            default=None,
            help="CPU cores used for thread tuning.",
        )
        pass_group.add_argument(
            "--io_bind",
            action="store_true",
            help="Whether enable IOBinding Search for ONNX Runtime inference.",
        )
        pass_group.add_argument(
            "--enable_cuda_graph",
            action="store_true",
            help="Whether enable CUDA Graph for CUDA execution provider.",
        )
        pass_group.add_argument(
            "--providers_list",
            type=str,
            nargs="*",
            help=(
                "List of execution providers to use for ONNX model. They are case sensitive. "
                "If not provided, all available providers will be used."
            ),
        )
        pass_group.add_argument(
            "--execution_mode_list", type=int, nargs="*", help="Parallelism list between operators."
        )
        pass_group.add_argument("--opt_level_list", type=int, nargs="*", help="Optimization level list for ONNX Model.")
        pass_group.add_argument("--trt_fp16_enable", action="store_true", help="Enable TensorRT FP16 mode.")
        pass_group.add_argument(
            "--intra_thread_num_list", type=int, nargs="*", help="List of intra thread number for test."
        )
        pass_group.add_argument(
            "--inter_thread_num_list", type=int, nargs="*", help="List of inter thread number for test."
        )
        pass_group.add_argument(
            "--extra_session_config",
            type=json.loads,
            default=None,
            help=(
                "Extra customized session options during tuning process. It should be a json string."
                'E.g. --extra_session_config \'{"key1": "value1", "key2": "value2"}\''
            ),
        )
        pass_group.add_argument(
            "--disable_force_evaluate_other_eps",
            action="store_true",
            help=(
                "Whether force to evaluate all execution providers"
                " which are different with the associated execution provider."
            ),
        )
        pass_group.add_argument(
            "--enable_profiling",
            action="store_true",
            help="Whether enable profiling for ONNX Runtime inference.",
        )

        sub_parser.add_argument(
            "--output_path",
            type=str,
            default="perf_tuning_output",
            help="Path to save the tuned inference settings.",
        )
        sub_parser.add_argument(
            "--tempdir", default=None, type=str, help="Root directory for tempfile directories and files"
        )

        # remote options
        add_remote_options(sub_parser)

        sub_parser.set_defaults(func=PerfTuningCommand)

    @staticmethod
    def perf_tuning_template():
        with PERF_TUNING_TEMPLATE.open() as f:
            return yaml.safe_load(f)

    def _update_default_data_config_params(self, default_data_config) -> Dict:
        data_config = deepcopy(default_data_config)
        load_dataset_keys = (
            "seq_len",
            "past_seq_len",
            "max_seq_len",
            "shared_kv",
            "generative",
            "ort_past_key_name",
            "ort_past_value_name",
            "trust_remote_code",
            "max_samples",
        )
        dataloader_keys = ("fields_no_batch", "batch_size")
        args_dict = vars(self.args)
        load_dataset_params = {k: args_dict[k] for k in load_dataset_keys if args_dict[k] is not None}
        load_dataset_params["model_name"] = self.args.hf_model_name
        dataloader_params = {k: args_dict[k] for k in dataloader_keys if args_dict[k] is not None}

        data_config["load_dataset_config"] = load_dataset_params
        data_config["dataloader_config"] = dataloader_params

        # special field predict_with_kv_cache to choose the data container type
        # from TransformersPromptDummyDataContainer and TransformersTokenDummyDataContainer
        data_config["type"] = (
            "TransformersTokenDummyDataContainer"
            if self.args.predict_with_kv_cache
            else "TransformersPromptDummyDataContainer"
        )
        return data_config

    def _update_pass_config(self, default_pass_config) -> Dict:
        pass_config = deepcopy(default_pass_config)
        pass_config_keys = (
            "cpu_cores",
            "io_bind",
            "enable_cuda_graph",
            "execution_mode_list",
            "opt_level_list",
            "trt_fp16_enable",
            "intra_thread_num_list",
            "inter_thread_num_list",
            "extra_session_config",
        )
        args_dict = vars(self.args)
        pass_config.update({k: args_dict[k] for k in pass_config_keys if args_dict[k] is not None})
        return pass_config

    def _get_data_config(self, template_config=None) -> Dict:
        if not self.args.data_config_path:
            if template_config is None:
                raise ValueError("Template config is required when data config is not provided.")
            return DataConfig.parse_file_or_obj(
                self._update_default_data_config_params(template_config["data_configs"][0])
            )
        return DataConfig.parse_file_or_obj(self.args.data_config_path)

    def refine_args(self):
        self.args.providers_list = self.args.providers_list or []
        for idx, provider in enumerate(self.args.providers_list):
            if not provider.endswith("ExecutionProvider"):
                self.args.providers_list[idx] = f"{provider}ExecutionProvider"

    def get_run_config(self, tempdir) -> Dict:
        template_config = PerfTuningCommand.perf_tuning_template()
        template_config["input_model"] = get_input_model_config(self.args)
        print(f"input_model: {template_config['input_model']}")

        perf_tuning_key = ("passes", "perf_tuning")
        system_device_key = ("systems", "local_system", "accelerators", 0, "device")

        data_configs = [self._get_data_config(template_config)]
        to_replace = [
            (("data_configs"), data_configs),
            (perf_tuning_key, self._update_pass_config(template_config["passes"]["perf_tuning"])),
            (system_device_key, self.args.device),
            ((*perf_tuning_key, "data_config"), data_configs[0].name),
        ]

        if self.args.providers_list:
            system_ep_key = ("systems", "local_system", "accelerators", 0, "execution_providers")
            to_replace.append((system_ep_key, self.args.providers_list))

        config = deepcopy(template_config)
        for k, v in to_replace:
            if v is None:
                continue
            set_nested_dict_value(config, k, v)

        update_remote_option(config, self.args, "perf-tuning", tempdir)
        config["log_severity_level"] = self.args.log_level

        return config

    def run(self):
        self.refine_args()
        set_tempdir(self.args.tempdir)
        with tempfile.TemporaryDirectory() as tempdir:
            run_config = self.get_run_config(tempdir)
            run_config["output_dir"] = tempdir
            output = olive_run(run_config)

            if is_remote_run(self.args):
                # TODO(jambayk): point user to datastore with outputs or download outputs
                # both are not implemented yet
                return

            if get_output_model_number(output) > 0:
                # need to improve the output structure of olive run
                output_path = Path(self.args.output_path)
                output_path.mkdir(parents=True, exist_ok=True)
                for provider in self.args.providers_list:
                    provider_key = provider.replace("ExecutionProvider", "").lower()
                    infer_setting_output_path = output_path / f"{self.args.device}-{provider_key}.json"
                    rls_json_path = Path(tempdir) / "perf_tuning" / f"{self.args.device}-{provider_key}_model.json"
                    with rls_json_path.open() as f:
                        infer_settings = json.load(f)["config"]["inference_settings"]
                        json.dump(infer_settings, infer_setting_output_path.open("w"), indent=4)
                print(f"Inference session parameters are saved to {output_path.resolve()}")
            else:
                print("Failed to run tune-session-params. Please set the log_level to 1 for more detailed logs.")
