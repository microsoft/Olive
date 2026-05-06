# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Example speech evaluation script for Olive.

This script demonstrates how to use the CUSTOM metric type with a speech model
to compute WER. It is referenced from the Olive config via:
    "user_script": "speech_eval_script.py",
    "evaluate_func": "evaluate_speech_wer"

The evaluate_func receives the model handler, device, and execution providers,
and is responsible for running inference and computing the metric.

IMPORTANT: You must implement the _transcribe() function for your specific model.
See the function docstring for guidance.
"""

import numpy as np


def evaluate_speech_wer(model, device, execution_providers):
    """Evaluate speech model with WER metric.

    This function is called by Olive's CUSTOM metric evaluator.
    It loads an ASR dataset, runs model inference, and computes WER.

    Args:
        model: OliveModelHandler (e.g., ONNXModelHandler)
        device: Device enum value
        execution_providers: List of execution providers

    Returns:
        dict with metric names as keys and float values.

    """
    import jiwer
    from datasets import Audio, load_dataset

    # Load dataset
    dataset = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        split="test.clean",
        streaming=False,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # Use a subset for quick evaluation
    dataset = dataset.select(range(min(100, len(dataset))))

    # Prepare model session
    session = model.prepare_session(device=device, execution_providers=execution_providers)

    predictions = []
    references = []

    for sample in dataset:
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        reference_text = sample["text"]

        # Run model inference
        # NOTE: Adapt this section to your specific model's input/output format
        # For onnxruntime-genai models (e.g., Nemotron), use the genai API:
        #   import onnxruntime_genai as og
        #   predicted_text = transcribe_with_genai(model_path, audio_array)
        #
        # For standard ONNX models:
        #   input_feed = preprocess_audio(audio_array)
        #   result = session.run(None, input_feed)
        #   predicted_text = decode_tokens(result)

        predicted_text = _transcribe(session, model, audio_array)
        predictions.append(predicted_text)
        references.append(reference_text.lower())

    # Compute WER
    wer = jiwer.wer(references, predictions)

    return {"wer": wer}


def _transcribe(session, model, audio_array):
    """Transcribe audio using model inference.

    Replace with your model's specific inference logic.
    For the Nemotron streaming model, see the onnxruntime-genai streaming API.
    """
    # Example for a standard encoder-decoder ASR ONNX model:
    # input_feed = {"audio_signal": audio_array.reshape(1, -1, 1)}
    # result = model.run_session(session, input_feed)
    # token_ids = result[0].argmax(axis=-1)
    # text = tokenizer.decode(token_ids)
    # return text
    raise NotImplementedError(
        "Implement _transcribe() for your specific model. See comments in evaluate_speech_wer() for guidance."
    )
