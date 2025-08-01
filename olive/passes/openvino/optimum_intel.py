# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from olive.common.utils import StrEnumBase
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import CompositeModelHandler, HfModelHandler, OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config

logger = logging.getLogger(__name__)


def maybe_load_preprocessors(
    src_name_or_path: Union[str, Path], subfolder: str = "", trust_remote_code: bool = False
) -> list:
    try:
        from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer
    except Exception as e:
        raise ImportError("Unable to import transformers packages: ", e) from None

    preprocessors = []
    try:
        preprocessors.append(
            AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoFeatureExtractor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoImageProcessor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception:
        pass

    return preprocessors


def infer_task(
    task,
    model_name_or_path,
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
    library_name: Optional[str] = None,
):
    try:
        from optimum.exporters import TasksManager
    except Exception as e:
        raise ImportError("Unable to import optimum packages:", e) from None

    try:
        from requests.exceptions import ConnectionError as RequestsConnectionError
    except Exception as e:
        raise ImportError("Unable to import ConnectionError packages:", e) from None

    task = TasksManager.map_from_synonym(task)
    if task == "auto":
        if library_name == "open_clip":
            task = "zero-shot-image-classification"
        else:
            try:
                task = TasksManager._infer_task_from_model_name_or_path(  # pylint: disable=W0212
                    model_name_or_path=model_name_or_path,
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    library_name=library_name,
                )
            except KeyError as e:
                raise KeyError(
                    f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                ) from None
            except RequestsConnectionError as e:
                raise RequestsConnectionError(
                    f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                ) from None
    return task


def maybe_convert_tokenizers(library_name: str, output: Path, model=None, preprocessors=None, task=None):
    from optimum.exporters.openvino.convert import export_tokenizer

    try:
        from transformers import PreTrainedTokenizerBase, ProcessorMixin
    except Exception as e:
        raise ImportError("Unable to import transformers packages:", e) from None

    try:
        from optimum.intel.utils.import_utils import is_openvino_tokenizers_available
    except Exception as e:
        raise ImportError("openvino tokenizers unavailable :", e) from None

    if is_openvino_tokenizers_available():
        if library_name != "diffusers" and preprocessors:
            processor_chat_template = None
            tokenizer = next(filter(lambda it: isinstance(it, PreTrainedTokenizerBase), preprocessors), None)
            if len(preprocessors) > 1:
                for processor in preprocessors:
                    if isinstance(processor, ProcessorMixin) and hasattr(processor, "chat_template"):
                        processor_chat_template = processor.chat_template
            if tokenizer:
                try:
                    export_tokenizer(tokenizer, output, task=task, processor_chat_template=processor_chat_template)
                except Exception as exception:
                    logger.warning(
                        "Could not load tokenizer using specified model ID or path. OpenVINO tokenizer/detokenizer models won't be generated. Exception: %s",
                        exception,
                    )
        elif model:
            for tokenizer_name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
                tokenizer = getattr(model, tokenizer_name, None)
                if tokenizer:
                    export_tokenizer(tokenizer, output / tokenizer_name, task=task)
    else:
        logger.warning("Tokenizer won't be converted.")


class OVQuantMode(StrEnumBase):
    INT8 = "int8"
    F8E4M3 = "f8e4m3"
    F8E5M2 = "f8e5m2"
    NF4_F8E4M3 = "nf4_f8e4m3"
    NF4_F8E5M2 = "nf4_f8e5m2"
    INT4_F8E4M3 = "int4_f8e4m3"
    INT4_F8E5M2 = "int4_f8e5m2"


class OVOptimumLibrary(StrEnumBase):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    TIMM = "timm"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPEN_CLIP = "open_clip"


class OVOptimumFramework(StrEnumBase):
    PT = "pt"
    TF = "tf"


class OVWeightFormat(StrEnumBase):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    MXFP4 = "mxfp4"
    NF4 = "nf4"


class OpenVINOOptimumConversion(Pass):
    """Convert a Hugging Face PyTorch model to OpenVINO model using the Optimum export function."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "components": PassConfigParam(
                type_=list[str],
                default_value=None,
                description=(
                    "List of component models to export. E.g. ['decoder_model', 'decoder_with_past_model']. None means"
                    " export all components."
                ),
            ),
            "device": PassConfigParam(
                type_=Device,
                default_value=accelerator_spec.accelerator_type.CPU,
                description=(
                    "The device to use to do the export. Defaults to 'cpu'."
                    "This is the parameter that is directly passed to Optimum Intel export function in certain cases."
                ),
            ),
            "extra_args": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Extra arguments to pass to the `optimum.exporters.openvino.main_export` function.",
            ),
            "ov_quant_config": PassConfigParam(
                type_=dict,
                default_value=None,
                required=False,
                description=(
                    "Parameters for optimum OpenVINO quantization. Please refer to "
                    "https://huggingface.co/docs/optimum/main/intel/openvino/optimization#4-bit"
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install nncf to use OpenVINO Optimum Conversion") from None
        if not super().validate_config(config, accelerator_spec):
            return False

        # validate allowed libraries in extra_args if provided
        if (
            config.extra_args
            and config.extra_args.get("library") is not None
            and config.extra_args.get("library") not in [lib.value for lib in OVOptimumLibrary]
        ):
            logger.error(
                "Library %s is not supported. Supported libraries are %s.",
                config.extra_args.get("library"),
                ", ".join([lib.value for lib in OVOptimumLibrary]),
            )
            return False

        # validate allowed frameworks if provided
        if (
            config.extra_args
            and config.extra_args.get("framework") is not None
            and config.extra_args.get("framework") not in [framework.value for framework in OVOptimumFramework]
        ):
            logger.error(
                "Framework %s is not supported. Supported frameworks are %s.",
                config.extra_args.get("framework"),
                ", ".join([framework.value for framework in OVOptimumFramework]),
            )
            return False

        # validate quantization weight_format if provided
        if (
            config.ov_quant_config
            and config.ov_quant_config.get("weight_format") is not None
            and config.ov_quant_config.get("weight_format")
            not in [weight_format.value for weight_format in OVWeightFormat]
        ):
            logger.error(
                "Weight format %s is not supported. Supported weight formats are %s.",
                config.ov_quant_config.get("weight_format"),
                ", ".join([weight_format.value for weight_format in OVWeightFormat]),
            )
            return False

        # validate quantization quant_mode if provided
        if (
            config.ov_quant_config
            and config.ov_quant_config.get("quant_mode") is not None
            and config.ov_quant_config.get("quant_mode") not in [quant_mode.value for quant_mode in OVQuantMode]
        ):
            logger.error(
                "Quant mode %s is not supported. Supported quant modes are %s.",
                config.ov_quant_config.get("quant_mode"),
                ", ".join([quant_mode.value for quant_mode in OVQuantMode]),
            )
            return False

        # validate backup precisions if provided
        if (
            config.ov_quant_config
            and config.ov_quant_config.get("backup_precision") is not None
            and config.ov_quant_config.get("backup_precision")
            not in [backupmode.value for backupmode in nncf.BackupMode]
        ):
            logger.error(
                "Backup precision %s is not supported. Supported backup precisions are %s.",
                config.ov_quant_config.get("backup_precision"),
                ", ".join([backupmode.value for backupmode in nncf.BackupMode]),
            )
            return False

        return True

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> Union[OpenVINOModelHandler, CompositeModelHandler]:
        try:
            from optimum.exporters.openvino import main_export as export_optimum_intel
            from optimum.exporters.openvino.utils import save_preprocessors
            from optimum.intel.openvino.configuration import OVConfig, get_default_int4_config
            from optimum.intel.utils.modeling_utils import _infer_library_from_model_name_or_path
        except ImportError as e:
            raise ImportError("Please install Intel® optimum[openvino] to use OpenVINO Optimum Conversion") from e

        # import the right quantization config depending on optimum-intel version
        try:
            from optimum.intel.openvino.configuration import _DEFAULT_4BIT_WQ_CONFIG as WRAPPER_4_BIT
        except ImportError as _:
            # fallback to older version
            logger.warning("falling back to older version of optimum-intel import API.")
            from optimum.intel.openvino.configuration import _DEFAULT_4BIT_CONFIG as WRAPPER_4_BIT

        extra_args = deepcopy(config.extra_args) if config.extra_args else {}
        extra_args.update(
            {
                "device": config.device,
            }
        )

        if model.load_kwargs and "trust_remote_code" not in extra_args:
            extra_args["trust_remote_code"] = model.load_kwargs.trust_remote_code

        if extra_args.get("library") is None:
            lib_name = _infer_library_from_model_name_or_path(model.model_name_or_path)
            if lib_name == "sentence_transformers":
                logger.warning(
                    "Library is not specified. "
                    "There are multiple possible variants: `sentence_transformers`, `transformers`. "
                    "`transformers` will be selected. "
                    "If you want to load your model with the `sentence-transformers` library instead, "
                    "Please set it as sentence_transformers in extra_args dictionary under 'library' key"
                )
                lib_name = "transformers"
        else:
            lib_name = extra_args["library"]

        if config.ov_quant_config:
            if config.ov_quant_config.get("weight_format") is None and config.ov_quant_config.get("quant_mode") is None:
                ov_config = None
                if not no_compression_parameter_provided(config.ov_quant_config):
                    raise ValueError(
                        "Some compression parameters are provided, but the weight format is not specified. "
                        "Please provide it with weight_format key in ov_quant_config dictionary."
                    )
                if not no_quantization_parameter_provided(config.ov_quant_config):
                    raise ValueError(
                        "Some quantization parameters are provided, but the quant mode is not specified. "
                        "Please provide it with quant_mode key in ov_quant_config dictionary."
                    )
            elif config.ov_quant_config.get("weight_format") in {"fp16", "fp32"}:
                ov_config = OVConfig(dtype=config.ov_quant_config["weight_format"])
            else:
                if config.ov_quant_config.get("weight_format") is not None:
                    # For int4 quantization if no parameter is provided, then use the default config if exists
                    if (
                        no_compression_parameter_provided(config.ov_quant_config)
                        and config.ov_quant_config.get("weight_format") == "int4"
                    ):
                        quant_config = get_default_int4_config(model.model_name_or_path)
                    else:
                        quant_config = prep_wc_config(config.ov_quant_config, WRAPPER_4_BIT)
                    if quant_config.get("dataset", None) is not None:
                        quant_config["trust_remote_code"] = config.ov_quant_config.get("trust_remote_code", False)
                    ov_config = OVConfig(quantization_config=quant_config)
                else:
                    ov_config = None
                    if config.ov_quant_config.get("dataset", None) is None:
                        raise ValueError(
                            "Dataset is required for full quantization. "
                            "Please provide it in ov_quant_config dictionary under 'dataset' key"
                        )
                    if config.ov_quant_config.get("quant_mode") in [
                        "nf4_f8e4m3",
                        "nf4_f8e5m2",
                        "int4_f8e4m3",
                        "int4_f8e5m2",
                    ]:
                        if lib_name == "diffusers":
                            raise NotImplementedError("Mixed precision quantization isn't supported for diffusers.")
                        wc_config = prep_wc_config(config.ov_quant_config, WRAPPER_4_BIT)
                        wc_dtype, q_dtype = config.ov_quant_config["quant_mode"].split("_")
                        wc_config["dtype"] = wc_dtype

                        q_config = prep_q_config(config.ov_quant_config)
                        q_config["dtype"] = q_dtype
                        quant_config = {
                            "weight_quantization_config": wc_config,
                            "full_quantization_config": q_config,
                            "num_samples": self.args.num_samples,
                            "dataset": self.args.dataset,
                            "trust_remote_code": self.args.trust_remote_code,
                        }
                    else:
                        quant_config = prep_q_config(config.ov_quant_config)
                    ov_config = OVConfig(quantization_config=quant_config)
        else:
            ov_config = None

        # quantization config
        quant_config = ov_config.quantization_config if ov_config else None
        quantize_with_dataset = quant_config and getattr(quant_config, "dataset", None) is not None
        task = infer_task(extra_args.get("task", "auto"), model.model_name_or_path, library_name=lib_name)

        # model
        if lib_name == "diffusers" and quantize_with_dataset:
            try:
                from diffusers import DiffusionPipeline
            except ImportError:
                raise ImportError("Please install diffusers to use OpenVINO with Diffusers models.") from None

            diffusers_config = DiffusionPipeline.load_config(model.model_name_or_path)
            class_name = diffusers_config.get("_class_name", None)

            if class_name == "LatentConsistencyModelPipeline":
                from optimum.intel import OVLatentConsistencyModelPipeline

                model_cls = OVLatentConsistencyModelPipeline

            elif class_name == "StableDiffusionXLPipeline":
                from optimum.intel import OVStableDiffusionXLPipeline

                model_cls = OVStableDiffusionXLPipeline
            elif class_name == "StableDiffusionPipeline":
                from optimum.intel import OVStableDiffusionPipeline

                model_cls = OVStableDiffusionPipeline
            elif class_name == "StableDiffusion3Pipeline":
                from optimum.intel import OVStableDiffusion3Pipeline

                model_cls = OVStableDiffusion3Pipeline
            elif class_name == "FluxPipeline":
                from optimum.intel import OVFluxPipeline

                model_cls = OVFluxPipeline
            elif class_name == "SanaPipeline":
                from optimum.intel import OVSanaPipeline

                model_cls = OVSanaPipeline
            else:
                raise NotImplementedError(f"Quantization isn't supported for class {class_name}.")

            output_model = model_cls.from_pretrained(
                model.model_name_or_path, export=True, quantization_config=quant_config
            )
            output_model.save_pretrained(output_model_path)
            if not extra_args.get("disable_convert_tokenizer", False):
                maybe_convert_tokenizers(lib_name, output_model_path, model, task=task)
        elif (
            quantize_with_dataset and (task.startswith("text-generation") or "automatic-speech-recognition" in task)
        ) or (task == "image-text-to-text" and quant_config is not None):
            if task.startswith("text-generation"):
                from optimum.intel import OVModelForCausalLM

                model_cls = OVModelForCausalLM
            elif task == "image-text-to-text":
                from optimum.intel import OVModelForVisualCausalLM

                model_cls = OVModelForVisualCausalLM
            else:
                from optimum.intel import OVModelForSpeechSeq2Seq

                model_cls = OVModelForSpeechSeq2Seq

            # In this case, to apply quantization an instance of a model class is required
            output_model = model_cls.from_pretrained(
                model.model_name_or_path,
                export=True,
                quantization_config=quant_config,
                stateful=not extra_args.get("disable_stateful", False),
                trust_remote_code=extra_args.get("trust_remote_code", False),
                variant=extra_args.get("variant", None),
                cache_dir=extra_args.get("cache_dir", HUGGINGFACE_HUB_CACHE),
            )
            output_model.save_pretrained(output_model_path)

            preprocessors = maybe_load_preprocessors(
                model.model_name_or_path, trust_remote_code=extra_args.get("trust_remote_code", False)
            )
            save_preprocessors(
                preprocessors, output_model.config, output_model_path, extra_args.get("trust_remote_code", False)
            )
            if not extra_args.get("disable_convert_tokenizer", False):
                maybe_convert_tokenizers(lib_name, output_model_path, preprocessors=preprocessors, task=task)

        else:
            extra_args["ov_config"] = ov_config
            extra_args["stateful"] = not extra_args.get("disable_stateful", False)
            extra_args.pop("disable_stateful", False)
            extra_args["convert_tokenizer"] = not extra_args.get("disable_convert_tokenizer", False)
            extra_args.pop("disable_convert_tokenizer", False)
            extra_args["library_name"] = lib_name
            extra_args.pop("library", None)
            export_optimum_intel(
                model.model_name_or_path,
                output_model_path,
                **extra_args,
            )

        # check the exported components
        exported_models = [name.stem for name in Path(output_model_path).iterdir() if name.suffix == ".xml"]
        logger.debug("Exported models are: %s.", exported_models)

        # OpenVINOModelHandler requires a directory with a single xml and bin file
        # OpenVINOModelHandler does not support multiple models in a single directory
        # If tokenizers are converted, those should be in a separate directory
        # OpenVINO would usually create both a tokenizer and a detokenizer in the same folder
        # return only the folder with just the OpenVINO model, not the tokenizer and detokenizer models.
        assert exported_models is not None
        assert len(exported_models) > 0, "No OpenVINO models were exported."

        # do not include tokenizer and detokenizer models for composite model creation
        remove_list = ["openvino_tokenizer", "openvino_detokenizer"]
        components = deepcopy(exported_models)
        if len(exported_models) > 1:
            for exported_model in exported_models:
                # move all extra OpenVINO XML and bin files to their respective subfolders
                if exported_model != "openvino_model":
                    extra_model_xml = Path(output_model_path) / f"{exported_model}.xml"
                    extra_model_bin = Path(output_model_path) / f"{exported_model}.bin"
                    dest_subdir = Path(output_model_path) / exported_model
                    dest_subdir.mkdir(parents=True, exist_ok=True)
                    if extra_model_xml.exists():
                        dest_xml = Path(dest_subdir) / f"{exported_model}.xml"
                        extra_model_xml.rename(dest_xml)
                        logger.debug("Moved %s to %s.", extra_model_xml, dest_xml)
                    if extra_model_bin.exists():
                        dest_bin = Path(dest_subdir) / f"{exported_model}.bin"
                        extra_model_bin.rename(dest_bin)
                        logger.debug("Moved %s to %s.", extra_model_bin, dest_bin)
                if exported_model in remove_list:
                    components.remove(exported_model)

        assert len(components) > 0, "No OpenVINO models were exported."
        # if only one model was exported return it directly
        if len(components) == 1:
            # will always return an OpenVINO model handler with folder as the model path
            return OpenVINOModelHandler(model_path=output_model_path)

        # if there are multiple components, return a composite model
        model_components = []
        model_component_names = []
        for component_name in components:
            if component_name in remove_list:
                # skip tokenizer and detokenizer models from the composite model
                continue
            if component_name != "openvino_model":
                # Each component is in a separate subfolder
                model_components.append(OpenVINOModelHandler(model_path=Path(output_model_path) / component_name))
            else:
                # The main model is in the output_model_path
                model_components.append(OpenVINOModelHandler(model_path=output_model_path))
            model_component_names.append(component_name)

        return CompositeModelHandler(model_components, model_component_names, model_path=output_model_path)


def prep_wc_config(quant_cfg, default_cfg):
    """Prepare the weight compression config for OpenVINO."""
    is_int8 = quant_cfg.get("weight_format") == "int8"
    return {
        "bits": 8 if is_int8 else 4,
        "ratio": 1.0 if is_int8 else (quant_cfg.get("ratio") or default_cfg.get("ratio")),
        "sym": quant_cfg.get("sym", False),
        "group_size": -1 if is_int8 else quant_cfg.get("group_size"),
        "all_layers": None if is_int8 else quant_cfg.get("all_layers", False),
        "dataset": quant_cfg.get("dataset"),
        "num_samples": quant_cfg.get("num_samples"),
        "quant_method": "awq" if quant_cfg.get("awq", False) else "default",
        "sensitivity_metric": quant_cfg.get("sensitivity_metric"),
        "scale_estimation": quant_cfg.get("scale_estimation", None),
        "gptq": quant_cfg.get("gptq", None),
        "lora_correction": quant_cfg.get("lora_correction", None),
        "dtype": quant_cfg.get("weight_format"),
        "backup_precision": quant_cfg.get("backup_precision"),
    }


def prep_q_config(quant_cfg):
    """Prepare the quantization config for OpenVINO."""
    return {
        "dtype": quant_cfg.get("quant_mode"),
        "bits": 8,
        "sym": quant_cfg.get("sym", False),
        "dataset": quant_cfg.get("dataset"),
        "num_samples": quant_cfg.get("num_samples"),
        "smooth_quant_alpha": quant_cfg.get("smooth_quant_alpha"),
        "trust_remote_code": quant_cfg.get("trust_remote_code", False),
    }


def no_compression_parameter_provided(q_config):
    return all(
        it is None
        for it in (
            q_config.get("ratio", None),
            q_config.get("group_size", None),
            q_config.get("sym", None),
            q_config.get("all_layers", None),
            q_config.get("dataset", None),
            q_config.get("num_samples", None),
            q_config.get("awq", None),
            q_config.get("scale_estimation", None),
            q_config.get("gptq", None),
            q_config.get("lora_correction", None),
            q_config.get("sensitivity_metric", None),
            q_config.get("backup_precision", None),
        )
    )


def no_quantization_parameter_provided(q_config):
    return all(
        it is None
        for it in (
            q_config.get("sym", None),
            q_config.get("dataset", None),
            q_config.get("num_samples", None),
            q_config.get("smooth_quant_alpha", None),
        )
    )
