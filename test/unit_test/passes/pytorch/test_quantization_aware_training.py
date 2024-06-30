# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import create_dummy_dataloader, get_pytorch_model

from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import FullPassConfig, create_pass_from_dict
from olive.passes.pytorch.quantization_aware_training import QuantizationAwareTraining


# TODO(shaahji): Remove this once QuantizationAwareTraining pass supports DataConfig
def _create_dummy_dataloader(data_dir, **kwargs):
    kwargs.pop("batch_size", None)
    return create_dummy_dataloader(data_dir, max_samples=1, batch_size=1, **kwargs)


def test_quantization_aware_training_pass_default(tmp_path):
    # setup
    input_model = get_pytorch_model()
    config = {
        "train_dataloader_func": _create_dummy_dataloader,
        "checkpoint_path": str(tmp_path / "checkpoint"),
    }

    p = create_pass_from_dict(QuantizationAwareTraining, config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


def test_optional_ep(tmp_path):
    accl = AcceleratorSpec("cpu", None)
    script_path = tmp_path / "user_script.py"
    with script_path.open("w"):
        pass
    p = create_pass_from_dict(
        QuantizationAwareTraining,
        {"train_dataloader_func": "create_dataloader", "user_script": str(script_path)},
        accelerator_spec=accl,
    )
    qat_json = p.to_json()
    pass_config = FullPassConfig.from_json(qat_json)
    sp = pass_config.create_pass()
    assert sp.accelerator_spec.execution_provider is None
