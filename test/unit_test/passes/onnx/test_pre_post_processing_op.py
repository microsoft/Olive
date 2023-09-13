from olive.model import ONNXModel, PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.append_pre_post_processing_ops import AppendPrePostProcessingOps
from olive.passes.onnx.conversion import OnnxConversion


def test_pre_post_processing_op(tmp_path):
    # setup
    p = create_pass_from_dict(
        AppendPrePostProcessingOps,
        {"tool_command": "superresolution", "tool_command_args": {"output_format": "png"}},
        disable_search=True,
    )

    pytorch_model = get_superresolution_model()
    input_model = convert_superresolution_model(pytorch_model, tmp_path)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


def test_pre_post_pipeline(tmp_path):
    config = {
        "pre": [
            {"ConvertImageToBGR": {}},
            {
                "Resize": {
                    "resize_to": [
                        {"type": "__model_input__", "input_index": 0, "dim_index": -2},
                        {"type": "__model_input__", "input_index": 0, "dim_index": -1},
                    ]
                }
            },
            {
                "CenterCrop": {
                    "height": {"type": "__model_input__", "input_index": 0, "dim_index": -2},
                    "width": {"type": "__model_input__", "input_index": 0, "dim_index": -1},
                }
            },
            {"PixelsToYCbCr": {"layout": "BGR"}},
            {"ImageBytesToFloat": {}},
            {"Unsqueeze": {"axes": [0, 1]}},
        ],
        "post": [
            {"Squeeze": {"axes": [0, 1]}},
            {"FloatToImageBytes": {"name": "Y1_uint8"}},
            {
                "Resize": {
                    "params": {
                        "resize_to": [
                            {"type": "__model_output__", "output_index": 0, "dim_index": -2},
                            {"type": "__model_output__", "output_index": 0, "dim_index": -1},
                        ],
                        "layout": "HW",
                    },
                    "io_map": [["PixelsToYCbCr", 1, 0]],
                }
            },
            {"FloatToImageBytes": {"multiplier": 1.0, "name": "Cb1_uint8"}},
            {
                "Resize": {
                    "params": {
                        "resize_to": [
                            {"type": "__model_output__", "output_index": 0, "dim_index": -2},
                            {"type": "__model_output__", "output_index": 0, "dim_index": -1},
                        ],
                        "layout": "HW",
                    },
                    "io_map": [["PixelsToYCbCr", 2, 0]],
                }
            },
            {"FloatToImageBytes": {"multiplier": 1.0, "name": "Cr1_uint8"}},
            {
                "YCbCrToPixels": {
                    "params": {
                        "layout": "BGR",
                    },
                    "io_map": [
                        ["Y1_uint8", 0, 0],
                        ["Cb1_uint8", 0, 1],
                        ["Cr1_uint8", 0, 2],
                    ],
                }
            },
            {"ConvertBGRToImage": {"image_format": "png"}},
        ],
        "tool_command_args": [
            {
                "name": "image",
                "data_type": "uint8",
                "shape": ["num_bytes"],
            }
        ],
        "target_opset": 16,
    }

    p = create_pass_from_dict(
        AppendPrePostProcessingOps,
        config,
        disable_search=True,
    )
    assert p is not None

    pytorch_model = get_superresolution_model()
    input_model = convert_superresolution_model(pytorch_model, tmp_path)
    input_model_graph = input_model.get_graph()
    assert input_model_graph.node[0].op_type == "Conv"
    output_folder = str(tmp_path / "onnx_pre_post")

    # execute
    model = p.run(input_model, None, output_folder)
    assert model is not None
    assert isinstance(model, ONNXModel)
    graph = model.get_graph()

    # assert the first node is ConvertImageToBGR
    assert graph.node[0].op_type == "DecodeImage"
    assert graph.node[0].domain == "com.microsoft.extensions"


def get_superresolution_model():
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

    return pytorch_model


def convert_superresolution_model(pytorch_model, tmp_path):
    onnx_conversion_pass = create_pass_from_dict(OnnxConversion, {"target_opset": 15}, disable_search=True)
    onnx_model = onnx_conversion_pass.run(pytorch_model, None, str(tmp_path / "onnx"))

    return onnx_model
