import tempfile
from pathlib import Path

from olive.model import PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.append_pre_post_processing_ops import AppendPrePostProcessingOps
from olive.passes.onnx.conversion import OnnxConversion
from olive.systems.local import LocalSystem


def test_pre_post_processing_op():
    # setup
    local_system = LocalSystem()
    p = create_pass_from_dict(
        AppendPrePostProcessingOps,
        {"tool_command": "superresolution", "tool_command_args": {"output_format": "png"}},
        disable_search=True,
    )
    with tempfile.TemporaryDirectory() as tempdir:
        input_model = get_superresolution_model(tempdir, local_system)
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, input_model, output_folder)


def get_superresolution_model(tempdir, local_system):
    import torch.nn as nn
    import torch.nn.init as init

    # Super Resolution model definition in PyTorch
    import torch.utils.model_zoo as model_zoo

    class SuperResolutionNet(nn.Module):
        def __init__(self, upscale_factor, inplace=False):
            super(SuperResolutionNet, self).__init__()

            self.relu = nn.ReLU(inplace=inplace)
            self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            self._initialize_weights()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x

        def _initialize_weights(self):
            init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
            init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
            init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
            init.orthogonal_(self.conv4.weight)

    def load_pytorch_model(model_path: str) -> nn.Module:
        # Create the super-resolution model by using the above model definition.
        torch_model = SuperResolutionNet(upscale_factor=3)

        # Load pretrained model weights
        model_url = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"

        # Initialize model with the pretrained weights
        torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=lambda storage, loc: storage))

        # set the model to inference mode
        torch_model.eval()

        return torch_model

    pytorch_model = PyTorchModel(
        model_loader=load_pytorch_model,
        io_config={
            "input_names": ["input"],
            "input_shapes": [[1, 1, 224, 224]],
            "input_types": ["float32"],
            "output_names": ["output"],
        },
    )
    onnx_conversion_pass = create_pass_from_dict(OnnxConversion, {"target_opset": 15}, disable_search=True)
    onnx_model = local_system.run_pass(onnx_conversion_pass, pytorch_model, str(Path(tempdir) / "onnx"))

    return onnx_model
