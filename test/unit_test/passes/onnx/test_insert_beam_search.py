import tempfile
from pathlib import Path
from test.unit_test.utils import get_onnx_model

from olive.model import CompositeOnnxModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.insert_beam_search import InsertBeamSearchPass
from olive.systems.local import LocalSystem


def test_ort_mixed_precision_pass():
    # setup
    local_system = LocalSystem()
    input_models = []
    input_models.append(get_onnx_model())
    input_models.append(get_onnx_model())
    composite_model = CompositeOnnxModel(input_models)

    p = create_pass_from_dict(InsertBeamSearchPass, {}, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, composite_model, output_folder)
