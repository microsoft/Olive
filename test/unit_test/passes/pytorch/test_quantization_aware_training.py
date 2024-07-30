# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_pytorch_model

from torch.utils.data import DataLoader

from olive.data.component.dataset import DummyDataset
from olive.data.registry import Registry
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import FullPassConfig, create_pass_from_dict
from olive.passes.pytorch.quantization_aware_training import QuantizationAwareTraining


@Registry.register_dataloader()
def _dummy_qat_dataloader(dataset, batch_size=1, max_samples=1, **kwargs):
    kwargs.pop("batch_size", None)
    return DataLoader(DummyDataset([(batch_size or 1, 1)], max_samples=max_samples), batch_size=None)


def test_quantization_aware_training_pass_default(tmp_path):
    # setup
    input_model = get_pytorch_model()
    config = {
        "train_data_config": {
            "name": "train_data_config",
            "type": "DummyDataContainer",
            "load_dataset_config": {"type": "simple_dataset"},
            "pre_process_data_config": {"type": "skip_pre_process"},
            "post_process_data_config": {"type": "skip_post_process"},
            "dataloader_config": {"type": "_dummy_qat_dataloader"},
        },
        "checkpoint_path": str(tmp_path / "checkpoint"),
    }

    p = create_pass_from_dict(QuantizationAwareTraining, config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)


def test_optional_ep(tmp_path):
    accl = AcceleratorSpec("cpu", None)
    config = {
        "train_data_config": {
            "name": "train_data_config",
            "type": "DummyDataContainer",
            "load_dataset_config": {"type": "simple_dataset"},
            "pre_process_data_config": {"type": "skip_pre_process"},
            "post_process_data_config": {"type": "skip_post_process"},
            "dataloader_config": {"type": "_dummy_qat_dataloader"},
        },
    }
    p = create_pass_from_dict(QuantizationAwareTraining, config, accelerator_spec=accl)
    qat_json = p.to_json()
    pass_config = FullPassConfig.from_json(qat_json)
    sp = pass_config.create_pass()
    assert sp.accelerator_spec.execution_provider is None
