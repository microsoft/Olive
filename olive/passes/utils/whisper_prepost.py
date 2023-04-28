# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import io

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime
import torch
from onnx import numpy_helper
from onnxruntime_extensions import PyOrtFunction, util
from onnxruntime_extensions.cvt import HFTokenizerConverter

# the flags for pre-processing
USE_ONNX_STFT = True
USE_AUDIO_DECODER = True


if not USE_AUDIO_DECODER:
    try:
        import librosa
    except ImportError:
        raise ImportError("Please pip3 install librosa without ort-extensions audio codec support.")


# hard-coded audio hyperparameters
# copied from https://github.com/openai/whisper/blob/main/whisper/audio.py#L12
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH


class CustomOpStftNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(g, self, n_fft, hop_length, window):
        t_n_fft = g.op("Constant", value_t=torch.tensor(n_fft, dtype=torch.int64))
        t_hop_length = g.op("Constant", value_t=torch.tensor(hop_length, dtype=torch.int64))
        t_frame_size = g.op("Constant", value_t=torch.tensor(n_fft, dtype=torch.int64))
        return g.op("ai.onnx.contrib::StftNorm", self, t_n_fft, t_hop_length, window, t_frame_size)

    @staticmethod
    def forward(ctx, audio, n_fft, hop_length, window):
        win_length = window.shape[0]
        stft = torch.stft(
            audio,
            n_fft,
            hop_length,
            win_length,
            window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return stft.abs() ** 2


class WhisperPrePipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window = torch.hann_window(N_FFT)
        self.mel_filters = torch.from_numpy(util.mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))

    def forward(self, audio_pcm: torch.Tensor):
        stft_norm = CustomOpStftNorm.apply(audio_pcm, N_FFT, HOP_LENGTH, self.window)
        magnitudes = stft_norm[:, :, :-1]
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        spec_min = log_spec.max() - 8.0
        log_spec = torch.maximum(log_spec, spec_min)
        spec_shape = log_spec.shape
        padding_spec = torch.ones(
            spec_shape[0], spec_shape[1], (N_SAMPLES // HOP_LENGTH - spec_shape[2]), dtype=torch.float
        )
        padding_spec *= spec_min
        log_spec = torch.cat((log_spec, padding_spec), dim=2)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


def _to_onnx_stft(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert custom-op STFT-Norm to ONNX STFT"""
    node_idx = 0
    new_stft_nodes = []
    stft_norm_node = None
    for node in onnx_model.graph.node:
        if node.op_type == "StftNorm":
            stft_norm_node = node
            break
        node_idx += 1

    if stft_norm_node is None:
        raise RuntimeError("Cannot find STFTNorm node in the graph")

    make_node = onnx.helper.make_node
    replaced_nodes = [
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_14_output_0"],
            name="const_14",
            value=numpy_helper.from_array(np.array([0, N_FFT // 2, 0, N_FFT // 2], dtype="int64"), name="const_14"),
        ),
        make_node(
            "Pad", inputs=[stft_norm_node.input[0], "const_14_output_0"], outputs=["pad_1_output_0"], mode="reflect"
        ),
        make_node(
            "STFT",
            inputs=["pad_1_output_0", stft_norm_node.input[2], stft_norm_node.input[3], stft_norm_node.input[4]],
            outputs=["stft_output_0"],
            name="stft",
            domain="",
            onesided=1,
        ),
        make_node(
            "Transpose",
            inputs=["stft_output_0"],
            outputs=["transpose_1_output_0"],
            name="transpose_1",
            perm=[0, 2, 1, 3],
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_17_output_0"],
            name="const_17",
            value=numpy_helper.from_array(np.array([2], dtype="int64"), name=""),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_18_output_0"],
            name="const_18",
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name=""),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_19_output_0"],
            name="const_19",
            value=numpy_helper.from_array(np.array([-1], dtype="int64"), name=""),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_20_output_0"],
            name="const_20",
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name=""),
        ),
        make_node(
            "Slice",
            inputs=[
                "transpose_1_output_0",
                "const_18_output_0",
                "const_19_output_0",
                "const_17_output_0",
                "const_20_output_0",
            ],
            outputs=["slice_1_output_0"],
            name="slice_1",
        ),
        make_node("Constant", inputs=[], outputs=["const0_output_0"], name="const0", value_int=0),
        make_node("Constant", inputs=[], outputs=["const1_output_0"], name="const1", value_int=1),
        make_node(
            "Gather",
            inputs=["slice_1_output_0", "const0_output_0"],
            outputs=["gather_4_output_0"],
            name="gather_4",
            axis=3,
        ),
        make_node(
            "Gather",
            inputs=["slice_1_output_0", "const1_output_0"],
            outputs=["gather_5_output_0"],
            name="gather_5",
            axis=3,
        ),
        make_node("Mul", inputs=["gather_4_output_0", "gather_4_output_0"], outputs=["mul_output_0"], name="mul0"),
        make_node("Mul", inputs=["gather_5_output_0", "gather_5_output_0"], outputs=["mul_1_output_0"], name="mul1"),
        make_node("Add", inputs=["mul_output_0", "mul_1_output_0"], outputs=[stft_norm_node.output[0]], name="add0"),
    ]
    new_stft_nodes.extend(onnx_model.graph.node[:node_idx])
    new_stft_nodes.extend(replaced_nodes)
    node_idx += 1
    new_stft_nodes.extend(onnx_model.graph.node[node_idx:])
    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(new_stft_nodes)
    return onnx_model


def _load_test_data(filepath: str) -> npt.NDArray[np.uint8]:
    if USE_AUDIO_DECODER:
        with open(filepath, "rb") as strm:
            audio_blob = np.asarray(list(strm.read()), dtype=np.uint8)
    else:
        audio_blob, _ = librosa.load(filepath)
    audio_blob = np.expand_dims(audio_blob, axis=0)  # add a batch_size
    return audio_blob


def _preprocessing(audio_data: npt.NDArray[np.uint8]) -> onnx.ModelProto:
    if USE_AUDIO_DECODER:
        decoder = PyOrtFunction.from_customop("AudioDecoder")

        # This is required in newer versions of ORT as per the error.
        # Extensions probably need to be updated.
        decoder.ort_session = onnxruntime.InferenceSession(
            decoder.onnx_model.SerializeToString(),
            decoder.get_ort_session_options(),
            providers=["CPUExecutionProvider"],
        )
        audio_pcm = torch.from_numpy(decoder(audio_data))
    else:
        audio_pcm = torch.from_numpy(audio_data)

    whisper_processing = WhisperPrePipeline()
    model_args = (audio_pcm,)

    with io.BytesIO() as strm:
        torch.onnx.export(
            whisper_processing,
            model_args,
            strm,
            input_names=["audio_pcm"],
            output_names=["log_mel"],
            do_constant_folding=True,
            export_params=True,
            opset_version=17,
            dynamic_axes={
                "audio_pcm": {1: "sample_len"},
            },
        )
        model = onnx.load_from_string(strm.getvalue())

    if USE_ONNX_STFT:
        model = _to_onnx_stft(model)

    if USE_AUDIO_DECODER:
        model = onnx.compose.merge_models(decoder.onnx_model, model, io_map=[("floatPCM", "audio_pcm")])

    return model


def _postprocessing(name: str) -> onnx.ModelProto:
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(name)
    fn_decoder = PyOrtFunction.from_customop(
        "BpeDecoder", cvt=HFTokenizerConverter(processor.tokenizer).bpe_decoder, skip_special_tokens=True
    )

    return fn_decoder.onnx_model


def _merge_models(
    pre_model: onnx.ModelProto, core_model: onnx.ModelProto, post_model: onnx.ModelProto
) -> onnx.ModelProto:
    pre_core_model = onnx.compose.merge_models(pre_model, core_model, io_map=[("log_mel", "input_features")])
    all_models = onnx.compose.merge_models(pre_core_model, post_model, io_map=[("sequences", "ids")])
    bpe_decoder_node = all_models.graph.node.pop(-1)
    bpe_decoder_node.input.pop(0)
    bpe_decoder_node.input.extend(["generated_ids"])
    all_models.graph.node.extend(
        [onnx.helper.make_node("Cast", ["sequences"], ["generated_ids"], to=onnx.TensorProto.INT64), bpe_decoder_node]
    )
    return all_models


def add_pre_post_processing_to_model(
    model: onnx.ModelProto, output_filepath: str, model_name: str, testdata_filepath: str
) -> onnx.ModelProto:
    audio_blob = _load_test_data(testdata_filepath)
    pre_model = _preprocessing(audio_blob)
    post_model = _postprocessing(model_name)
    final_model = _merge_models(pre_model, model, post_model)
    onnx.checker.check_model(final_model)

    # For some reason, the model with external data doesn't work!!
    onnx.save_model(
        final_model,
        output_filepath,
        # save_as_external_data=True,
        # all_tensors_to_one_file=True,
        convert_attribute=True,
    )

    return final_model
