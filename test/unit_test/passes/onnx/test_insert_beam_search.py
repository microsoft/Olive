import tempfile
from pathlib import Path
from test.unit_test.utils import get_onnx_model

from transformers import AutoConfig

from olive.model import CompositeOnnxModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.insert_beam_search import InsertBeamSearch
from olive.systems.local import LocalSystem


def test_insert_beam_search_pass():
    # setup
    local_system = LocalSystem()
    input_models = []
    input_models.append(get_onnx_model())
    input_models.append(get_onnx_model())
    composite_model = CompositeOnnxModel(
        input_models,
        ["encoder_decoder_init", "decoder"],
        model_attributes=AutoConfig.from_pretrained("openai/whisper-base.en").to_dict(),
    )

    p = create_pass_from_dict(InsertBeamSearch, {}, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, composite_model, None, output_folder)
