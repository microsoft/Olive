# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

from olive.engine.footprint import Footprint
from olive.packaging.packaging_config import PackagingConfig, PackagingType

logger = logging.getLogger(__name__)


def generate_output_artifacts(
    packaging_config: PackagingConfig, foot_print: Footprint, pf_footprint: Footprint, output_dir: Path
):
    if packaging_config.type == PackagingType.Zipfile:
        _generate_zipfile_output(packaging_config, foot_print, pf_footprint, output_dir)


def _generate_zipfile_output(
    packaging_config: PackagingConfig, footprint: Footprint, pf_footprint: Footprint, output_dir: Path
) -> None:
    logger.info("Packaging Zipfile output artifacts")
    cur_path = Path(__file__).parent
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        _package_sample_code(cur_path, tempdir)
        _package_candidate_models(tempdir, footprint, pf_footprint)
        shutil.make_archive(packaging_config.name, "zip", tempdir)
        shutil.move(f"{packaging_config.name}.zip", output_dir / f"{packaging_config.name}.zip")


def _package_sample_code(cur_path, tempdir):
    shutil.copytree(cur_path / "sample_code", tempdir / "SampleCode")
    

def _package_candidate_models(tempdir, footprint: Footprint, pf_footprint: Footprint) -> None:
    candidate_models_dir = tempdir / "CandidateModels"
    candidate_models_dir.mkdir()
    model_rank = 1
    for model_id, node in pf_footprint.nodes.items():
        model_dir = candidate_models_dir / f"BestCandidateModel_{model_rank}"
        model_dir.mkdir()
        model_rank += 1
        # Copy model file
        model_path = pf_footprint.get_model_path(model_id)
        model_type = pf_footprint.get_model_type(model_id)

        if model_type == "ONNXModel":
            shutil.copy2(model_path, model_dir / "model.onnx")
        elif model_type == "OpenVINOModel":
            shutil.copytree(model_path, model_dir / "model")
        else:
            raise ValueError(f"Unsupported model type: {model_type} for packaging")

        # Copy inference config
        inference_config_path = str(model_dir / "inference_config.json")
        with open(inference_config_path, "w") as f:
            json.dump(pf_footprint.get_model_inference_config(model_id), f)

        # Copy Passes configurations
        configuration_path = str(model_dir / "configurations.json")
        with open(configuration_path, "w") as f:
            json.dump(OrderedDict(reversed(footprint.trace_back_run_history(model_id).items())), f)
            
        # Copy metrics
        # TODO: Add target info to metrics file
        metric_path = str(model_dir / "metrics.json")
        with open(metric_path, "w") as f:
            json.dump(node.metrics.value, f)
