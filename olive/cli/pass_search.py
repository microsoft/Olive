# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import re
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from olive.cli.base import (
    BaseOliveCLICommand,
    add_dataset_options,
    add_logging_options,
    add_save_config_file_options,
    add_telemetry_options,
    update_dataset_options,
)
from olive.telemetry import action

# Endpoint used to submit remote Olive-as-a-Service jobs.
OLIVE_AAS_RUN_URL = "https://oliveaas.azure-api.net/v1/run"


def _hex_string(value: str) -> str:
    """Validate that the value is a 32-character hexadecimal string."""
    if not value or not re.fullmatch(r"[0-9a-fA-F]{32}", value):
        raise ArgumentTypeError("must be a 32-character hexadecimal string")
    return value


class PassSearchCommand(BaseOliveCLICommand):
    """Run an Olive workflow that searches *across passes*, not parameters within a pass.

    This command searches over which passes (and in what combinations) to apply to a model.
    It does NOT tune the individual configuration parameters of any single pass -- each pass
    is applied using its default configuration. Use ``--include-pass-params`` only if you
    additionally want the underlying search strategy to explore per-pass parameters.

    WARNING: Enabling ``--include-pass-params`` expands the search space combinatorially. The
    total number of search points is roughly the product of the searchable parameter
    combinations across every pass in the group, so even a few passes can blow up the search
    space exponentially. Before launching a run, check the engine log line
    "Search space contains N search points ..." to confirm the size is manageable.

    Two mutually exclusive modes of operation are supported:

    1. Config-file mode (``--run-config`` / ``--config``):
       Start from an existing JSON workflow config. Search strategy options (sampler,
       execution order, iteration/time budgets, etc.) are merged into the config's
       ``engine.search_strategy`` section, and the (single) resulting config is run,
       saved (``--save_config_file``/``--dry_run``), or submitted as a remote job.

    2. Build-mode (``-m`` / ``--model_name_or_path``):
       Construct workflow config(s) on the fly from command-line arguments. The ``--passes``
       argument defines ordered pass groups; each group becomes its own generated
       ``config_%02d.json`` (a separate workflow), letting you compare different pass
       pipelines. A slot written as ``(A | B)`` is an either/or alternative the search chooses
       between. Because this is a search command, every generated config always includes a
       search strategy (in no-search mode either/or slots would collapse to their first
       alternative). Model/device/provider, evaluation, and dataset options are also applied.
       Each generated config is then run, or submitted as a remote job.

    In both modes, providing ``--az_apim_subscription_key`` submits the generated config(s)
    to the remote Olive-as-a-Service endpoint instead of running them locally.
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "pass-search",
            help=(
                "Search across passes (which passes/pipelines to apply), not the parameters within an "
                "individual pass. Each pass uses its default configuration."
            ),
        )

        # ------------------------------------------------------------------- #
        # Mode selection: mutually exclusive — config file OR build from args #
        # ------------------------------------------------------------------- #
        mode_group = sub_parser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument(
            "--run-config",
            "--config",
            type=str,
            dest="run_config",
            default=None,
            help="Path to a JSON workflow config file. Mutually exclusive with --model_name_or_path.",
        )
        mode_group.add_argument(
            "-m",
            "--model_name_or_path",
            type=str,
            dest="model_name_or_path",
            default=None,
            help=(
                "Model to search over (HuggingFace id, local path, or AzureML URI). "
                "Mutually exclusive with --run-config."
            ),
        )

        # ------------------------------------------------------------------ #
        # Common options (both modes)                                        #
        # ------------------------------------------------------------------ #
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
                "of individual pass module implementation and corresponding dependencies."
            ),
        )
        sub_parser.add_argument(
            "--az_apim_subscription_key",
            type=_hex_string,
            default=None,
            help=(
                "Azure APIM subscription key (hexadecimal string). When provided, each generated config "
                "is submitted as a remote Olive-as-a-Service job instead of running locally, and the "
                "returned operation id (op_xxxxxxxxx) is printed."
            ),
        )

        # ------------------------------------------------------------------ #
        # Search strategy options (both modes)                               #
        # ------------------------------------------------------------------ #
        search_group = sub_parser.add_argument_group("Search strategy options")
        search_group.add_argument(
            "--execution-order",
            type=str,
            choices=["joint", "pass-by-pass"],
            default=None,
            help=(
                "Order in which passes are searched. "
                "'joint' searches all passes together; 'pass-by-pass' searches each pass sequentially."
            ),
        )
        search_group.add_argument(
            "--sampler",
            type=str,
            choices=["sequential", "random", "tpe"],
            default=None,
            help="Sampler algorithm to use for exploring the search space.",
        )
        search_group.add_argument(
            "--max-iter",
            type=int,
            default=None,
            help="Maximum number of search iterations. Only applies to joint execution order.",
        )
        search_group.add_argument(
            "--max-time",
            type=int,
            default=None,
            help="Maximum search time in seconds. Only applies to joint execution order.",
        )
        search_group.add_argument(
            "--output-model-num",
            type=int,
            default=None,
            help="Number of output models to produce from the search.",
        )
        search_group.add_argument(
            "--stop-when-goals-met",
            action="store_true",
            default=False,
            help="Stop searching once all metric goals are met. Only applies to joint execution order.",
        )
        # WARNING: Enabling --include-pass-params expands the search space combinatorially.
        # The total number of search points is roughly the product of the number of searchable
        # parameter combinations across every pass in the group, so even a handful of passes can
        # cause the search space to blow up exponentially. Before launching a run, check the log
        # line "Search space contains N search points ..." (emitted by the engine) to confirm the
        # size is manageable.
        search_group.add_argument(
            "--include-pass-params",
            action="store_true",
            default=None,
            help=(
                "Include individual pass parameters in the search space. WARNING: this expands the "
                "search space combinatorially -- even a few passes can blow it up exponentially. "
                "Check the engine log ('Search space contains N search points ...') before launching."
            ),
        )

        # ------------------------------------------------------------------ #
        # Evaluation options (lm-eval based, used to score searched models)  #
        # ------------------------------------------------------------------ #
        eval_group = sub_parser.add_argument_group(
            "Evaluation options",
            description=(
                "lm-eval based evaluation used to score models produced during the search. "
                "Provide --eval-tasks to enable evaluation when building a config from arguments."
            ),
        )
        eval_group.add_argument(
            "--eval-tasks",
            type=str,
            nargs="*",
            default=None,
            help="List of lm-eval tasks to evaluate searched models on (e.g. hellaswag winogrande).",
        )
        eval_group.add_argument(
            "--eval-batch-size",
            type=int,
            default=1,
            help="Batch size for evaluation. Default is 1.",
        )
        eval_group.add_argument(
            "--eval-max-length",
            type=int,
            default=1024,
            help="Maximum length of input + output for evaluation. Default is 1024.",
        )
        eval_group.add_argument(
            "--eval-limit",
            type=float,
            default=1,
            help="Number (or percentage of dataset) of samples to use for evaluation. Default is 1.",
        )
        eval_group.add_argument(
            "--eval-backend",
            type=str,
            default="auto",
            choices=["auto", "ort", "ortgenai"],
            help=(
                "Backend for lm-eval model evaluation. 'ort' and 'ortgenai' require ONNX input; "
                "'ortgenai' additionally requires GenAI-packaged model assets (e.g., genai_config.json). "
                "'auto' infers backend from model type."
            ),
        )

        # ------------------------------------------------------------------ #
        # Build-mode options (only used when --model_name_or_path is given)  #
        # ------------------------------------------------------------------ #
        build_group = sub_parser.add_argument_group(
            "Build-mode options",
            description=(
                "These options are used to construct a run config on-the-fly "
                "when --model_name_or_path is provided instead of --run-config."
            ),
        )
        build_group.add_argument(
            "--passes",
            type=str,
            nargs="+",
            metavar="PASSES",
            default=None,
            help=(
                "Ordered pass groups to search over. Each group generates a separate config file named "
                "'config_%%02d.json'. Wrap a multi-pass group in square brackets, e.g. "
                "--passes [OnnxConversion, OrtTransformersOptimization] [OnnxConversion, OnnxQuantization]. "
                "Unbracketed entries are each treated as their own single-pass group, e.g. "
                "--passes OnnxConversion OnnxQuantization generates two configs. "
                "Use parentheses with '|' to make an either/or slot the search chooses between, e.g. "
                "--passes [OnnxConversion, (OnnxQuantization | OnnxStaticQuantization)]. "
                "Each value must be a valid Olive pass type name and passes use their default configuration."
            ),
        )
        build_group.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "gpu", "npu"],
            help="Target device for the search. Default is cpu.",
        )
        build_group.add_argument(
            "--provider",
            type=str,
            default="CPUExecutionProvider",
            choices=[
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "QNNExecutionProvider",
                "VitisAIExecutionProvider",
                "OpenVINOExecutionProvider",
                "WebGpuExecutionProvider",
                "NvTensorRTRTXExecutionProvider",
            ],
            help="Execution provider to target during search. Default is CPUExecutionProvider.",
        )
        build_group.add_argument(
            "-t",
            "--task",
            type=str,
            default=None,
            help="HuggingFace task for the model (e.g. text-generation).",
        )
        build_group.add_argument(
            "--trust_remote_code",
            action="store_true",
            help="Trust remote code when loading a HuggingFace model.",
        )
        build_group.add_argument(
            "-o",
            "--output_path",
            type=str,
            default="search-output",
            help="Directory to save the search output. Default is 'search-output'.",
        )

        # -------------------------------------------------------------------- #
        # Dataset options (optional, used by evaluation/data-dependent passes) #
        # -------------------------------------------------------------------- #
        add_dataset_options(sub_parser, required=False, include_train=True, include_eval=True)

        add_logging_options(sub_parser, default=None)
        add_save_config_file_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=PassSearchCommand)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_search_strategy_overrides(args: Namespace) -> dict:
        overrides = {}
        if args.execution_order is not None:
            overrides["execution_order"] = args.execution_order
        if args.sampler is not None:
            overrides["sampler"] = args.sampler
        if args.max_iter is not None:
            overrides["max_iter"] = args.max_iter
        if args.max_time is not None:
            overrides["max_time"] = args.max_time
        if args.output_model_num is not None:
            overrides["output_model_num"] = args.output_model_num
        if args.stop_when_goals_met:
            overrides["stop_when_goals_met"] = True
        if args.include_pass_params is not None:
            overrides["include_pass_params"] = args.include_pass_params
        return overrides

    @staticmethod
    def _apply_search_strategy(run_config: dict, overrides: dict) -> None:
        if not overrides:
            return
        engine_config = run_config.setdefault("engine", {})
        existing = engine_config.get("search_strategy") or {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update(overrides)
        engine_config["search_strategy"] = existing
        print(f"Applying search strategy overrides: {overrides}")

    def _build_run_config(self, passes: "list[str | list[str]]", output_dir: str) -> dict[str, Any]:
        """Construct a run config dict for a single ordered group (pipeline) of passes.

        Each element of ``passes`` is a *slot*: either a single pass name (str) or a list of
        alternative pass names. An alternatives slot maps the pass to a list of configs, which
        the engine explores as an either/or choice via its search strategy.
        """
        args = self.args

        # Build passes section using default configuration for each pass.
        passes_config: dict[str, Any] = OrderedDict()

        def _unique_key(base: str) -> str:
            key, suffix = base, 1
            while key in passes_config:
                key = f"{base}_{suffix}"
                suffix += 1
            return key

        for slot in passes:
            if isinstance(slot, list):
                # Either/or slot -> map the pass name to a list of alternative configs.
                key = _unique_key("_or_".join(pass_type.lower() for pass_type in slot))
                passes_config[key] = [{"type": pass_type} for pass_type in slot]
            else:
                key = _unique_key(slot.lower())
                passes_config[key] = {"type": slot}

        # Input model
        input_model: dict[str, Any] = {"type": "HfModel", "model_path": args.model_name_or_path}
        if getattr(args, "task", None):
            input_model["task"] = args.task
        if getattr(args, "trust_remote_code", False):
            input_model["load_kwargs"] = {"trust_remote_code": True}

        # Accelerator / system
        run_config: dict[str, Any] = {
            "input_model": input_model,
            "passes": passes_config,
            "systems": {
                "local_system": {
                    "type": "LocalSystem",
                    "accelerators": [
                        {
                            "device": args.device,
                            "execution_providers": [args.provider],
                        }
                    ],
                }
            },
            "target": "local_system",
            "output_dir": output_dir,
            "no_artifacts": True,
        }

        # This is the pass-search command, so search is always on. Inject a default search
        # strategy unconditionally; without it the engine runs in no-search mode and either/or
        # slots would collapse to their first alternative (see Engine._compute_no_search_pass_configs).
        # Any --execution-order/--sampler overrides are merged on top later by _apply_search_strategy.
        run_config.setdefault("engine", {}).setdefault(
            "search_strategy", {"execution_order": "joint", "sampler": "sequential"}
        )

        # Add lm-eval based evaluator when evaluation tasks are provided.
        self._add_evaluator(run_config)

        # Add data config when a dataset is provided.
        self._add_data_config(run_config)

        if args.log_level is not None:
            run_config["log_severity_level"] = args.log_level

        return run_config

    def _add_data_config(self, run_config: dict[str, Any]) -> None:
        """Add a data config to the run config when --data_name is provided."""
        if not getattr(self.args, "data_name", None):
            return

        run_config["data_configs"] = [
            {
                "name": "default_data_config",
                "type": "HuggingfaceContainer",
                "load_dataset_config": {},
                "pre_process_data_config": {},
                "dataloader_config": {},
                "post_process_data_config": {},
            }
        ]
        update_dataset_options(self.args, run_config)

    def _add_evaluator(self, run_config: dict[str, Any]) -> None:
        """Add an lm-eval based evaluator to the run config when --eval-tasks is provided."""
        args = self.args
        if not getattr(args, "eval_tasks", None):
            return

        run_config["evaluators"] = {
            "evaluator": {
                "type": "LMEvaluator",
                "tasks": args.eval_tasks,
                "batch_size": args.eval_batch_size,
                "max_length": args.eval_max_length,
                "device": args.device,
                "limit": args.eval_limit,
            }
        }
        if args.eval_backend != "auto":
            run_config["evaluators"]["evaluator"]["model_class"] = args.eval_backend
        run_config["evaluator"] = "evaluator"
        run_config.setdefault("host", "local_system")

    @staticmethod
    def _split_top_level(segment: str) -> list[str]:
        """Split a segment on commas/whitespace while keeping parenthesized groups intact."""
        tokens: list[str] = []
        buf: list[str] = []
        depth = 0
        for ch in segment:
            if ch == "(":
                depth += 1
                buf.append(ch)
            elif ch == ")":
                depth = max(0, depth - 1)
                buf.append(ch)
            elif depth == 0 and (ch == "," or ch.isspace()):
                if buf:
                    tokens.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf))
        return tokens

    @staticmethod
    def _parse_slot(token: str) -> "Optional[str | list[str]]":
        """Parse a single slot token into a pass name or a list of alternatives.

        ``A`` -> ``"A"`` (a single pass).
        ``(A | B | C)`` -> ``["A", "B", "C"]`` (an either/or slot; the search picks one).
        Returns ``None`` for an empty token.
        """
        token = token.strip()
        if not token:
            return None
        alt_match = re.fullmatch(r"\((.*)\)", token, re.DOTALL)
        if alt_match:
            alts = [a.strip() for a in re.split(r"[|,]", alt_match.group(1)) if a.strip()]
            if not alts:
                return None
            # A single alternative degenerates to a plain single-pass slot.
            return alts if len(alts) > 1 else alts[0]
        return token

    @staticmethod
    def _parse_pass_groups(raw: list[str]) -> "list[list[str | list[str]]]":
        """Parse the --passes tokens into a list of ordered pass groups (pipelines).

        Each pipeline is a list of *slots*; a slot is either a single pass name or an either/or
        group of alternatives that the search chooses between:
          - A bracketed segment is a single pipeline: ``[A, B, C]`` -> ``["A", "B", "C"]``.
          - Each unbracketed entry is its own single-pass pipeline: ``A B`` -> ``["A"], ["B"]``.
          - A parenthesized ``(X | Y)`` slot is an either/or alternative resolved by the search:
            ``[A, (B | C)]`` -> ``["A", ["B", "C"]]`` (run A, then either B or C).
        So ``[A, B] C (D | E)`` -> ``[["A", "B"], ["C"], [["D", "E"]]]``.
        Commas and spaces separate slots; ``|`` (or commas) separate alternatives within ``(...)``.
        """
        joined = " ".join(raw)

        groups: list[list] = []
        pos = 0
        for match in re.finditer(r"\[([^\]]*)\]", joined):
            # Unbracketed tokens before this bracketed segment each form their own pipeline.
            groups.extend(
                [slot]
                for token in PassSearchCommand._split_top_level(joined[pos : match.start()])
                if (slot := PassSearchCommand._parse_slot(token)) is not None
            )
            # The bracketed segment forms a single pipeline of slots.
            slots = [
                slot
                for token in PassSearchCommand._split_top_level(match.group(1))
                if (slot := PassSearchCommand._parse_slot(token)) is not None
            ]
            if slots:
                groups.append(slots)
            pos = match.end()

        # Any trailing unbracketed tokens after the last bracketed segment.
        groups.extend(
            [slot]
            for token in PassSearchCommand._split_top_level(joined[pos:])
            if (slot := PassSearchCommand._parse_slot(token)) is not None
        )

        if not groups:
            raise ValueError("No valid pass groups parsed from --passes.")
        return groups

    @staticmethod
    def _count_search_points(run_config: dict[str, Any]) -> Optional[int]:
        """Compute how many search points the engine will evaluate for a run config.

        Remote jobs execute on the Olive-as-a-Service backend, so the engine's own
        "Search space contains N search points ..." log line (emitted by
        ``olive/engine/engine.py``) is not visible locally. Rather than re-deriving the
        combinatorics by hand, this reuses the exact machinery the engine uses:

          1. Resolve the search strategy config to determine ``include_pass_params``. This flag
             controls whether individual pass parameters participate in the search: the engine
             uses ``disable_pass_params_search = not include_pass_params`` (see
             ``Engine._compute_search_pass_configs``). When it is ``False`` the per-pass
             parameters are excluded from the space, so the count is *not* the product of the
             per-pass parameter combinations.
          2. For each pass, import the pass module and call ``Pass.get_config_params(...)`` with
             ``disable_search=not include_pass_params`` to obtain its (possibly empty) searchable
             parameters -- mirroring ``Engine._get_search_space_config``.
          3. Feed the resulting ``{pass_name: [search_params]}`` into a ``SearchStrategy`` and
             read ``SearchStrategy.max_samples`` (the product of every search space's size).

        Because it goes through the same code path, the returned count matches the effective
        search space for the run. Returns ``None`` when the size cannot be determined (e.g. an
        unknown/unimportable pass type).
        """
        passes = run_config.get("passes") or {}
        if not passes:
            return None

        # A run config without a search strategy runs in no-search mode: a single point.
        search_strategy = (run_config.get("engine") or {}).get("search_strategy")
        if search_strategy is None:
            search_strategy = run_config.get("search_strategy")
        if not search_strategy:
            return 1

        from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec
        from olive.package_config import OlivePackageConfig
        from olive.search.search_strategy import SearchStrategy, SearchStrategyConfig

        # Resolve the accelerator spec from the run config (falling back to CPU).
        accelerator_spec = DEFAULT_CPU_ACCELERATOR
        systems = run_config.get("systems") or {}
        target = systems.get(run_config.get("target")) or {}
        accelerators = target.get("accelerators") or []
        if accelerators:
            first = accelerators[0]
            providers = first.get("execution_providers") or []
            accelerator_spec = AcceleratorSpec(
                accelerator_type=first.get("device", "cpu"),
                execution_provider=providers[0] if providers else None,
            )

        # Build the strategy config first so we can honor include_pass_params (and its default).
        strategy_config = dict(search_strategy) if isinstance(search_strategy, dict) else {}
        strategy_config.setdefault("execution_order", "joint")
        strategy_config.setdefault("sampler", "sequential")
        try:
            resolved_strategy = SearchStrategyConfig(**strategy_config)
        except Exception:  # pylint: disable=broad-except
            return None

        # When include_pass_params is False, per-pass parameters are not part of the search space,
        # so we must disable search when introspecting each pass's parameters.
        disable_pass_params_search = not resolved_strategy.include_pass_params

        olive_config = OlivePackageConfig.load_default_config()

        # Mirror Engine._get_search_space_config -> {pass_name: [search_params, ...]}.
        # A pass name can map to either a single config (dict) or a list of alternative
        # configs. The engine wraps the alternatives in a Categorical, so a pass with multiple
        # configs contributes the *sum* of each alternative's search-space size.
        space_config: dict[str, list[dict[str, Any]]] = OrderedDict()
        for pass_name, pass_value in passes.items():
            pass_configs = pass_value if isinstance(pass_value, list) else [pass_value]
            params_per_config: list[dict[str, Any]] = []
            for pass_config in pass_configs:
                if not isinstance(pass_config, dict) or "type" not in pass_config:
                    return None
                try:
                    pass_cls = olive_config.import_pass_module(pass_config["type"])
                    _, _, search_params = pass_cls.get_config_params(
                        accelerator_spec, pass_config.get("config"), disable_pass_params_search
                    )
                except Exception:  # pylint: disable=broad-except
                    # Any failure to import/introspect a pass makes the count unreliable.
                    return None
                params_per_config.append(search_params)
            space_config[pass_name] = params_per_config

        # Build the strategy exactly as the engine does and read the total sample count.
        try:
            strategy = SearchStrategy(resolved_strategy)
            strategy.initialize(space_config, "search-point-count", {})
            return strategy.max_samples
        except Exception:  # pylint: disable=broad-except
            return None

    def _report_search_point_count(self, run_config: dict[str, Any], config_name: Optional[str] = None) -> None:
        """Print the number of search points that will be evaluated for a run config."""
        count = self._count_search_points(run_config)
        prefix = f"[{config_name}] " if config_name else ""
        if count is None:
            print(f"{prefix}Unable to determine the number of search points for this config.")
        else:
            print(f"{prefix}Search space contains {count} search point(s) to be evaluated for this config.")

    @staticmethod
    def _submit_remote_job(run_config: dict[str, Any], subscription_key: str) -> str:
        """Submit a run config as a remote Olive-as-a-Service job and return the operation id."""
        import requests

        response = requests.post(
            OLIVE_AAS_RUN_URL,
            headers={
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": subscription_key,
            },
            json={"run_config": run_config},
            timeout=60,
        )
        response.raise_for_status()

        # The service returns the operation id, either as a bare string or wrapped in JSON.
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        op_id = payload if isinstance(payload, str) else (payload.get("op_id") or payload.get("id"))
        if not op_id:
            raise RuntimeError(f"Unexpected response from remote job submission: {response.text!r}")
        return op_id.strip().strip('"')

    # ---------------------------------------------------------------------- #
    # Entry point                                                            #
    # ---------------------------------------------------------------------- #

    @action
    def run(self):
        from olive.workflows import run as olive_run

        if self.args.run_config is not None:
            # ---- Config-file mode ---------------------------------------- #
            from olive.common.config_utils import load_config_file

            run_config = self.args.run_config
            if not isinstance(run_config, dict):
                run_config = load_config_file(run_config)

            for arg_key, rc_key in [("log_level", "log_severity_level")]:
                if (arg_value := getattr(self.args, arg_key, None)) is not None:
                    print(f"Replacing {rc_key} in run config with {arg_value}")
                    run_config.get("engine", {}).pop(rc_key, None)
                    run_config[rc_key] = arg_value

            # Apply search strategy overrides
            overrides = self._build_search_strategy_overrides(self.args)
            self._apply_search_strategy(run_config, overrides)

            if self.args.save_config_file or self.args.dry_run:
                self._save_config_file(run_config)

            if self.args.dry_run:
                print("Dry run mode enabled. Configuration file is generated but no workflow is executed.")
                return

            if self.args.az_apim_subscription_key:
                self._report_search_point_count(run_config)
                op_id = self._submit_remote_job(run_config, self.args.az_apim_subscription_key)
                print(f"Remote job submitted. Operation id: {op_id}")
                return

            olive_run(
                run_config,
                list_required_packages=self.args.list_required_packages,
                tempdir=self.args.tempdir,
                package_config=self.args.package_config,
            )

            if self.args.list_required_packages is True:
                print("Required packages listed!")
            return

        # ---- Build mode -------------------------------------------------- #
        if not self.args.passes:
            raise ValueError(
                "At least one pass group must be specified with --passes when using build mode (--model_name_or_path)."
            )

        groups = self._parse_pass_groups(self.args.passes)
        overrides = self._build_search_strategy_overrides(self.args)

        output_root = Path(self.args.output_path)
        output_root.mkdir(parents=True, exist_ok=True)

        # Generate one config per ordered pass group, saved as config_%02d.json.
        run_configs: list[tuple[str, dict[str, Any]]] = []
        for idx, group in enumerate(groups):
            config_name = f"config_{idx:02d}"
            run_config = self._build_run_config(group, (output_root / config_name).as_posix())
            self._apply_search_strategy(run_config, overrides)

            config_path = output_root / f"{config_name}.json"
            with open(config_path, "w") as f:
                json.dump(run_config, f, indent=4)
            print(f"Config file saved at {config_path} for pass group {group}")
            run_configs.append((config_name, run_config))

        if self.args.dry_run:
            print(
                f"Dry run mode enabled. {len(run_configs)} configuration file(s) generated but no workflow is executed."
            )
            return

        if self.args.az_apim_subscription_key:
            for config_name, run_config in run_configs:
                print(f"Submitting remote job for {config_name} ...")
                self._report_search_point_count(run_config, config_name)
                op_id = self._submit_remote_job(run_config, self.args.az_apim_subscription_key)
                print(f"Remote job submitted for {config_name}. Operation id: {op_id}")
            return

        for config_name, run_config in run_configs:
            print(f"Running search workflow for {config_name} ...")
            olive_run(
                run_config,
                list_required_packages=self.args.list_required_packages,
                tempdir=self.args.tempdir,
                package_config=self.args.package_config,
            )

        if self.args.list_required_packages is True:
            print("Required packages listed!")
