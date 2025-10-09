# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from olive.common.config_utils import validate_config
from olive.common.utils import StrEnumBase
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model.handler import CompositeModelHandler, HfModelHandler, ONNXModelHandler, OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, ParamCategory, PassConfigParam, get_user_script_data_config

logger = logging.getLogger(__name__)


class IgnoreScopeTypeEnum(StrEnumBase):
    NAMES = "names"
    TYPES = "types"
    PATTERNS = "patterns"


class OVOptimumLibrary(StrEnumBase):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    TIMM = "timm"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPEN_CLIP = "open_clip"


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
    except Exception as e:
        logger.warning("Could not load tokenizer using specified model ID or path.\n Exception: %s", e)

    try:
        preprocessors.append(
            AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception as e:
        logger.warning("Could not load processor using specified model ID or path.\n Exception: %s", e)

    try:
        preprocessors.append(
            AutoFeatureExtractor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception as e:
        logger.warning("Could not load feature extractor using specified model ID or path.\n Exception: %s", e)

    try:
        preprocessors.append(
            AutoImageProcessor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception as e:
        logger.warning("Could not load image processor using specified model ID or path.\n Exception: %s", e)

    return preprocessors


def maybe_convert_tokenizers(library_name: str, output: Path, model=None, preprocessors=None, task=None):
    try:
        from optimum.exporters.openvino.convert import export_tokenizer
    except Exception as e:
        raise ImportError("Unable to import optimum Intel® package:", e) from None

    try:
        from transformers import PreTrainedTokenizerBase
    except Exception as e:
        raise ImportError("Unable to import transformers packages:", e) from None

    try:
        from optimum.intel.utils.import_utils import is_openvino_tokenizers_available
    except Exception as e:
        raise ImportError("openvino tokenizers unavailable :", e) from None

    if is_openvino_tokenizers_available():
        if library_name != "diffusers" and preprocessors:
            tokenizer = next(filter(lambda it: isinstance(it, PreTrainedTokenizerBase), preprocessors), None)
            if tokenizer:
                try:
                    export_tokenizer(tokenizer, output, task=task)
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


class OpenVINOWeightCompression(Pass):
    """OpenVINO weight compression pass.

    Please refer to https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html for more details.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                required=False,
                description="Data config for compression.",
            ),
            "ignored_scope": PassConfigParam(
                type_=Union[str, list[str]],
                required=False,
                default_value=None,
                description=(
                    "This parameter can be used to exclude some layers "
                    "from the quantization process to preserve the model accuracy. Please refer to "
                    "https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html#tune-quantization-parameters."
                ),
            ),
            "ignored_scope_type": PassConfigParam(
                type_=IgnoreScopeTypeEnum,
                required=False,
                default_value=None,
                description="Defines the type of the ignored scope. Supported values: 'names', 'types', 'patterns'.",
            ),
            "target_device": PassConfigParam(
                type_=Device,
                required=False,
                default_value=accelerator_spec.accelerator_type,
                description=(
                    "Target device for the model. "
                    "Supported values: 'any', 'cpu', 'gpu', 'cpu_spr', 'npu'. "
                    "Default value is the same as the accelerator type of this workflow run."
                ),
            ),
            "transform_fn": PassConfigParam(
                type_=Union[Callable, str],
                category=ParamCategory.OBJECT,
                required=False,
                description="Transform function for the input data.",
            ),
            "extra_args": PassConfigParam(
                type_=dict,
                required=False,
                default_value=None,
                description="Extra arguments to pass to the `nncf.compress_weights()` function.",
            ),
            "compress_config": PassConfigParam(
                type_=dict,
                required=False,
                default_value=None,
                description=(
                    "Weight Compression configuration for OpenVINO model weight compression. Please refer to "
                    "https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html."
                ),
            ),
            "reuse_cache": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Reuse cache of previous passes to reduce storage footprint."),
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
            raise ImportError("Please install nncf to use OpenVINO Weight Compression") from None

        if not super().validate_config(config, accelerator_spec):
            return False

        # validate mode in compress_config if provided
        if (
            config.compress_config
            and config.compress_config.get("mode") is not None
            and config.compress_config.get("mode") not in [m.value for m in nncf.parameters.CompressWeightsMode]
        ):
            logger.error(
                "Mode %s is not supported. Supported CompressWeightsMode values are %s.",
                config.compress_config.get("mode"),
                ", ".join([m.value for m in nncf.parameters.CompressWeightsMode]),
            )
            return False

        # validate sensitivity metric in compress_config if provided
        if (
            config.compress_config
            and config.compress_config.get("sensitivity_metric") is not None
            and config.compress_config.get("sensitivity_metric")
            not in [m.value for m in nncf.parameters.SensitivityMetric]
        ):
            logger.error(
                "Sensitivity metric %s is not supported. Supported SensitivityMetric values are %s.",
                config.compress_config.get("sensitivity_metric"),
                ", ".join([m.value for m in nncf.parameters.SensitivityMetric]),
            )
            return False

        # validate backup mode in compress_config if provided
        if (
            config.compress_config
            and config.compress_config.get("backup_mode") is not None
            and config.compress_config.get("backup_mode") not in [m.value for m in nncf.parameters.BackupMode]
        ):
            logger.error(
                "Backup mode %s is not supported. Supported BackupMode values are %s.",
                config.compress_config.get("backup_mode"),
                ", ".join([m.value for m in nncf.parameters.BackupMode]),
            )
            return False

        # validate compression format in compress_config if provided
        if (
            config.compress_config
            and config.compress_config.get("compression_format") is not None
            and config.compress_config.get("compression_format")
            not in [m.name for m in nncf.parameters.CompressionFormat]
        ):
            logger.error(
                "Compression format %s is not supported. Supported CompressionFormat values are %s.",
                config.compress_config.get("compression_format"),
                ", ".join([m.name for m in nncf.parameters.CompressionFormat]),
            )
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

        # convert all enum strings to actual enum values for compress_config
        # this is for mode, sensitivity_metrics, backup_mode, compression_format
        if config.compress_config:
            if config.compress_config.get("mode"):
                config.compress_config["mode"] = nncf.parameters.CompressWeightsMode(config.compress_config.get("mode"))
            if config.compress_config.get("sensitivity_metric"):
                config.compress_config["sensitivity_metric"] = nncf.parameters.SensitivityMetric(
                    config.compress_config.get("sensitivity_metric")
                )
            if config.compress_config.get("backup_mode"):
                config.compress_config["backup_mode"] = nncf.parameters.BackupMode(
                    config.compress_config.get("backup_mode")
                )
            if config.compress_config.get("compression_format"):
                config.compress_config["compression_format"] = nncf.parameters.CompressionFormat(
                    config.compress_config.get("compression_format")
                )

        return True

    @staticmethod
    def _create_calibration_dataset(common_dataloader):
        """Create an nncf.Dataset instance from a common dataloader."""
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        def transform_fn(data_item):
            data, _ = data_item
            return data

        return nncf.Dataset(common_dataloader, transform_fn)

    def _get_nncf_dataset(self, config, tokenizer: Optional = None):
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        data_loader = None
        if config.data_config:
            data_config = validate_config(config.data_config, DataConfig)
            data_loader = data_config.to_data_container().create_dataloader()

        # if a data_config is not specified, return None
        if data_loader is None:
            return None

        def transform_fn(data_item):
            data, _ = data_item
            return data

        transform_func = (
            self._user_module_loader.load_object(config.transform_fn) if config.transform_fn else transform_fn
        )

        # use extra args to load tokenizer and pass via partial
        if config.extra_args and tokenizer is not None:
            transform_func = partial(transform_func, tokenizer=tokenizer)

        return nncf.Dataset(data_loader, transform_func)

    @staticmethod
    def _get_extra_params(config):
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        extra_params = {}
        if config.extra_args and config.extra_args.get("model_type") is not None:
            extra_params["model_type"] = (
                nncf.ModelType.TRANSFORMER if config.extra_args.get("model_type") == "TRANSFORMER" else None
            )
        if config.extra_args and config.extra_args.get("preset") is not None:
            extra_params["preset"] = (
                nncf.QuantizationPreset.PERFORMANCE
                if config.extra_args.get("preset") == "PERFORMANCE"
                else nncf.QuantizationPreset.MIXED
            )
        # target device is not needed for weight compression with NNCF
        if config.ignored_scope:
            kwargs = {config.ignored_scope_type: config.ignored_scope}
            extra_params["ignored_scope"] = nncf.IgnoredScope(**kwargs)

        return extra_params

    @staticmethod
    def _get_advanced_compression_params(config):
        advanced_params = {}
        if config.extra_args and config.extra_args.get("advanced_compression_parameters") is not None:
            advanced_params = deepcopy(config.extra_args.get("advanced_compression_parameters"))
        if advanced_params and advanced_params.get("backend_params") is not None:
            backend_params = advanced_params.get("backend_params")
            if backend_params.get("external_dir") is not None:
                # pop external_dir from advanced_params. Will be added at runtime
                advanced_params["backend_params"].pop("external_dir")
        return advanced_params

    def _run_for_config(
        self,
        model: Union[HfModelHandler, ONNXModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[OpenVINOModelHandler, ONNXModelHandler, CompositeModelHandler]:
        if not isinstance(model, (HfModelHandler, ONNXModelHandler)):
            raise TypeError("OpenVINOWeightCompression pass can only be applied to Hugging Face or ONNX models")

        if config.reuse_cache:
            model_name_path = Path(model.model_path)
            weight_name_path = None
            if isinstance(model, OpenVINOModelHandler):
                model_name = model.model_config["model_name"]
                model_name_path = Path(model.model_path) / (f"{model_name}.xml")
                weight_name_path = Path(model.model_path) / (f"{model_name}.bin")
                output_model_path = model.model_path
            elif isinstance(model, ONNXModelHandler):
                output_model_path = str(
                    Path(model.model_path).with_name(Path(model.model_path).stem + "_compressed.onnx")
                )

        # initialize output_model to None
        output_model = None

        if isinstance(model, HfModelHandler):
            output_model = self._run_hf_pass(model, config, output_model_path)
        elif isinstance(model, ONNXModelHandler):
            output_model = self._run_onnx_pass(model, config, output_model_path)

        if config.reuse_cache:
            if os.path.exists(model_name_path):
                os.remove(model_name_path)
            if weight_name_path is not None and os.path.exists(weight_name_path):
                os.remove(weight_name_path)

        return output_model

    def _run_hf_pass(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[OpenVINOModelHandler, CompositeModelHandler]:
        try:
            import nncf
            from nncf.onnx.quantization.backend_parameters import BackendParameters
            from optimum.exporters.openvino import main_export as export_optimum_intel
            from optimum.intel.utils.modeling_utils import _infer_library_from_model_name_or_path
        except ImportError:
            raise ImportError(
                "Please install Intel® optimum[openvino] to use NNCF for weight compression on HF models"
            ) from None

        # local copy of extra_args
        extra_args = deepcopy(config.extra_args) if config.extra_args else {}

        # local copy of compress_config
        compress_config = deepcopy(config.compress_config) if config.compress_config else {}

        # set the library name for the HF Model
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
            extra_args["library"] = lib_name
        else:
            lib_name = extra_args["library"]

        # infer task
        task = infer_task(extra_args.get("task", "auto"), model.model_name_or_path, library_name=lib_name)

        # model
        if lib_name == "diffusers":
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
                raise NotImplementedError(f"{class_name} isn't supported.")

            output_model = model_cls.from_pretrained(
                model.model_name_or_path, export=True, load_in_8bit=False, compile=False
            )
            if not extra_args.get("disable_convert_tokenizer", False):
                maybe_convert_tokenizers(lib_name, output_model_path, model, task=task)
        elif (task.startswith("text-generation") or "automatic-speech-recognition" in task) or (
            task == "image-text-to-text"
        ):
            if task.startswith("text-generation"):
                from optimum.intel import OVModelForCausalLM

                model_cls = OVModelForCausalLM
            elif task == "image-text-to-text":
                from optimum.intel import OVModelForVisualCausalLM

                model_cls = OVModelForVisualCausalLM
            else:
                from optimum.intel import OVModelForSpeechSeq2Seq

                model_cls = OVModelForSpeechSeq2Seq

            output_model = model_cls.from_pretrained(
                model.model_name_or_path,
                export=True,
                load_in_8bit=False,
                compile=False,
                stateful=not extra_args.get("disable_stateful", False),
                trust_remote_code=extra_args.get("trust_remote_code", False),
                variant=extra_args.get("variant", None),
                cache_dir=extra_args.get("cache_dir", HUGGINGFACE_HUB_CACHE),
            )

            preprocessors = maybe_load_preprocessors(
                model.model_name_or_path, trust_remote_code=extra_args.get("trust_remote_code", False)
            )
            if not extra_args.get("disable_convert_tokenizer", False):
                maybe_convert_tokenizers(lib_name, output_model_path, preprocessors=preprocessors, task=task)

        else:
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

        # redirect to ONNXModelHandler if extra_args requests ONNX processing
        # this is also only for CausalLM models
        from optimum.intel import OVModelForCausalLM

        if config.extra_args and config.extra_args.get("use_onnx") and isinstance(output_model, OVModelForCausalLM):
            try:
                from optimum.onnxruntime import ORTModelForCausalLM
            except ImportError:
                raise ImportError("Please install optimum[onnxruntime] to use ONNX models.") from None
            output_model = ORTModelForCausalLM.from_pretrained(model.model_name_or_path, export=True)

            # if pad_token_id is not set, set it to eos_token_id to avoid warnings during generation
            if output_model.config.pad_token_id == -1:
                output_model.config.pad_token_id = output_model.config.eos_token_id
                logger.warning(
                    "pad_token_id is not set. Setting pad_token_id to eos_token_id: %s to avoid warnings during generation.",
                    output_model.config.eos_token_id,
                )
            if output_model.generation_config.pad_token_id == -1:
                output_model.generation_config.pad_token_id = output_model.config.eos_token_id
                logger.warning(
                    "generation_config.pad_token_id is not set. Setting generation_config.pad_token_id to eos_token_id: %s to avoid warnings during generation.",
                    output_model.config.eos_token_id,
                )

            output_model.save_pretrained(output_model_path)
            omp = Path(output_model_path) / "model.onnx"
            omh = ONNXModelHandler(model_path=omp)
            return self._run_onnx_pass(omh, config, output_model_path)

        # initialize tokenizer to None
        tokenizer = None
        if config.extra_args and config.extra_args.get("tokenizer"):
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "Install transformers to use NNCF for weight compression with tokenizers for Huggingface models"
                ) from None
            tokenizer = AutoTokenizer.from_pretrained(model.model_name_or_path)

        # get the weight compression dataset
        compression_dataset = self._get_nncf_dataset(config, tokenizer)

        # get the extra params
        extra_params = self._get_extra_params(config)

        # append extra params to compress config
        compress_config.update(extra_params)

        # get nncf.AdvancedCompressionParameters if any
        advanced_params = None
        adv_par = self._get_advanced_compression_params(config)
        if config.extra_args and config.extra_args.get("advanced_compression_parameters") and adv_par is not None:
            acp = config.extra_args.get("advanced_compression_parameters")
            bp = acp.get("backend_params")
            if bp is not None and bp.get("external_dir") is not None:
                adv_par["backend_params"][BackendParameters.EXTERNAL_DATA_DIR] = output_model_path

        if adv_par is not None:
            advanced_params = nncf.AdvancedCompressionParameters(**adv_par)

        # perform weight compression
        output_model.model = nncf.compress_weights(
            output_model.model, dataset=compression_dataset, advanced_parameters=advanced_params, **compress_config
        )

        # save to output_model_path
        output_model.save_pretrained(output_model_path)

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

    def _run_onnx_pass(
        self,
        model: ONNXModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            import nncf
            import onnx
            from nncf.onnx.quantization.backend_parameters import BackendParameters
        except ImportError:
            raise ImportError(
                "Please install Intel® NNCF and ONNX to use nncf.compress_weights() on ONNX models"
            ) from None

        # load model
        loaded_model = onnx.load(model.model_path, load_external_data=False)

        # convert model to target opset version if necessary
        target_opset = 21 if config.extra_args is None else config.extra_args.get("target_opset", 21)
        if loaded_model.opset_import[0].version != target_opset:
            loaded_model = onnx.version_converter.convert_version(loaded_model, target_opset)

        # local copy of compress_config
        compress_config = deepcopy(config.compress_config) if config.compress_config else {}

        # get the weight compression dataset
        compression_dataset = self._get_nncf_dataset(config)

        # get the extra params
        extra_params = self._get_extra_params(config)

        # append extra params to compress config
        compress_config.update(extra_params)

        # get nncf.AdvancedCompressionParameters if any
        advanced_params = None
        adv_par = self._get_advanced_compression_params(config)
        if config.extra_args and config.extra_args.get("advanced_compression_parameters") and adv_par is not None:
            acp = config.extra_args.get("advanced_compression_parameters")
            bp = acp.get("backend_params")
            if bp is not None and bp.get("external_dir") is not None:
                adv_par["backend_params"][BackendParameters.EXTERNAL_DATA_DIR] = output_model_path

        if adv_par is not None:
            advanced_params = nncf.AdvancedCompressionParameters(**adv_par)

        # perform weight compression
        output_model = nncf.compress_weights(
            loaded_model, dataset=compression_dataset, advanced_parameters=advanced_params, **compress_config
        )

        # save to output_model_path
        model_name = Path(model.model_path).name.replace(".onnx", "_compressed.onnx")
        model_dir = Path(output_model_path)

        if Path(output_model_path).is_dir():
            output_model_path = Path(output_model_path) / model_name
        onnx.save(output_model, output_model_path, save_as_external_data=True)

        # generate the genai_config.json file for GenAI ONNX models
        create_genai_config(model_name, model_dir, config)

        return ONNXModelHandler(model_path=output_model_path)


def create_genai_config(model_name: str, output_path: str, config: type[BasePassConfig]) -> None:
    """Generate the genai_config.json from the model config files.

    This is only for Generative AI models for which the config.json and generation_config.json files exist
    Arguments:
    @param model_name: name of model ONNX file that is generated
    @param output_path: path to the output directory where the genai_config.json file will be created
    @return: None
    """
    ip_conf_pth = Path(output_path) / "config.json"

    # do not create genai_config.json if config.json does not exist
    if not ip_conf_pth.exists():
        return

    ip_gen_pth = Path(output_path) / "generation_config.json"

    # do not create genai_config.json if generation_config.json does not exist
    if not ip_gen_pth.exists():
        return

    # Step 1: Create your data structure
    genai_config = {
        "model": {
            "bos_token_id": -1,
            "context_length": -1,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "graph_optimization_level": "ORT_DISABLE_ALL",
                    "provider_options": [
                        {"OpenVINO": {"device_type": config.target_device.upper(), "enable_causallm": "True"}}
                    ],
                },
                "filename": "openvino_model.onnx",
                "head_size": -1,
                "hidden_size": -1,
                "inputs": {},
                "outputs": {},
                "num_attention_heads": -1,
                "num_hidden_layers": -1,
                "num_key_value_heads": -1,
            },
            "eos_token_id": -1,
            "type": "",
            "vocab_size": -1,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": -1,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }

    import json

    with open(ip_conf_pth) as f:
        src_config = json.load(f)

    with open(ip_gen_pth) as f:
        src_gen_config = json.load(f)

    try:
        import onnx
    except ImportError:
        raise ImportError(
            "Please install onnx to create genai_config.json for ONNX OpenVINO IR Encapsulated model"
        ) from None

    model_path = Path(output_path) / model_name
    model = onnx.load(model_path)

    # Get input and output tensor names
    inputs = [inp.name for inp in model.graph.input]
    outputs = [out.name for out in model.graph.output]

    genai_config["model"]["bos_token_id"] = src_config.get("bos_token_id", -1)
    genai_config["model"]["context_length"] = src_config.get("max_position_embeddings", -1)
    genai_config["model"]["decoder"]["filename"] = model_name
    num_attention_heads = src_config.get("num_attention_heads", -1)
    hidden_size = src_config.get("hidden_size", -1)
    if (
        isinstance(num_attention_heads, int)
        and isinstance(hidden_size, int)
        and num_attention_heads > 0
        and hidden_size >= 0
    ):
        genai_config["model"]["decoder"]["head_size"] = hidden_size // num_attention_heads
    else:
        if not isinstance(num_attention_heads, int):
            logger.warning("num_attention_heads is not an int: %s found in src_config", num_attention_heads)
        elif num_attention_heads <= 0:
            logger.warning("Invalid num_attention_heads (<= 0) %s found in src_config", num_attention_heads)
        if not isinstance(hidden_size, int):
            logger.warning("hidden_size is not an int: %s found in src_config", hidden_size)
        elif hidden_size < 0:
            logger.warning("Invalid hidden_size (< 0) %s found in src_config", hidden_size)
        logger.warning("Setting genai_config['model']['decoder']['head_size'] to -1")
        genai_config["model"]["decoder"]["head_size"] = -1
    genai_config["model"]["decoder"]["hidden_size"] = src_config.get("hidden_size", -1)

    for name in inputs:
        if name != "beam_idx":
            genai_config["model"]["decoder"]["inputs"].update({name: name})

    for name in outputs:
        genai_config["model"]["decoder"]["outputs"].update({name: name})

    genai_config["model"]["decoder"]["num_attention_heads"] = src_config.get("num_attention_heads", -1)
    genai_config["model"]["decoder"]["num_hidden_layers"] = src_config.get("num_hidden_layers", -1)
    genai_config["model"]["decoder"]["num_key_value_heads"] = src_config.get("num_key_value_heads", -1)

    eos_token_id = src_gen_config.get("eos_token_id", -1)
    genai_config["model"]["eos_token_id"] = eos_token_id
    pad_token_id = src_gen_config.get("pad_token_id", None)
    if pad_token_id is not None:
        genai_config["model"]["pad_token_id"] = pad_token_id
    elif eos_token_id != -1:
        genai_config["model"]["pad_token_id"] = (
            eos_token_id[0] if isinstance(eos_token_id, list) and len(eos_token_id) > 0 else eos_token_id
        )
    else:
        genai_config["model"]["pad_token_id"] = -1
    genai_config["model"]["type"] = src_config.get("model_type", "")
    genai_config["model"]["vocab_size"] = src_config.get("vocab_size", -1)

    genai_config["search"]["max_length"] = src_config.get("max_position_embeddings", -1)

    # Step 2: Write to JSON file
    output_genai_config = Path(output_path) / "genai_config.json"
    with open(output_genai_config, "w") as f:
        json.dump(genai_config, f, indent=4)
