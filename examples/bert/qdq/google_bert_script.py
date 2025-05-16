from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm.auto import tqdm
from transformers import (
    AutoModelForNextSentencePrediction,
    AutoTokenizer,
)

from olive.common.utils import format_data
from olive.data.registry import Registry

if TYPE_CHECKING:
    from olive.hardware.accelerator import Device
    from olive.model import ONNXModelHandler


def load_bert_nsp_model(model_name: str) -> torch.nn.Module:
    return AutoModelForNextSentencePrediction.from_pretrained(model_name).eval()


@Registry.register_post_process()
def bert_scl_post_process(outputs):
    """Post-processing for Sequence Classification task."""
    match outputs:
        case torch.Tensor():
            return outputs.argmax(dim=-1)
        case OrderedDict() | dict() if "logits" in outputs:
            return outputs["logits"].argmax(dim=-1)
        case OrderedDict() | dict() if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
        case _:
            raise ValueError(f"Unsupported output type: {type(outputs)}")


@Registry.register_dataset()
def dataset_to_nsp_dataset(
    data_name: str,
    subset: str,
    split: str,
    input_cols: list[str],
    label_col: str,
    max_samples: int | None,
):
    from wikitext import create_nsp_dataset

    return create_nsp_dataset(
        dataset=data_name,
        subset=subset,
        split=split,
        sent_cols=input_cols,
        label_col=label_col,
        max_samples=max_samples,
    )


def eval_squad(
    model: ONNXModelHandler,
    device: Device,
    execution_providers: str,
    dataset_config: dict[str, str],
    model_name: str,
    max_samples: int | None = None,
) -> dict[str, float | int]:
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    sample_queue, result_queue = Queue(maxsize=500), Queue(maxsize=10)

    dataset = load_dataset(
        path=dataset_config["data_name"],
        split=dataset_config["split"],
    )
    if max_samples is not None:
        dataset = dataset.take(min(max_samples, len(dataset)))

    def data_thread_func():
        io_config = model.io_config
        input_ids_index = io_config["input_names"].index("input_ids")
        input_ids_shape = io_config["input_shapes"][input_ids_index]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for sample in tqdm(dataset, position=0, desc="Loading Data"):
            encoded_input = tokenizer(
                sample["question"],
                sample["context"],
                padding="max_length",
                max_length=input_ids_shape[1],
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            inputs = format_data(
                {
                    "input_ids": encoded_input.input_ids,
                    "attention_mask": encoded_input.attention_mask,
                },
                io_config,
            )
            sample_queue.put((inputs, encoded_input.offset_mapping, sample))

        # Sentinel value to indicate end of data
        sample_queue.put((None, None, None))

    def inference_thread_func():
        sess = model.prepare_session(
            device=device,
            execution_providers=execution_providers,
        )
        with tqdm(total=len(dataset), position=1, desc="Inferencing") as pbar:
            while True:
                inputs, offset_mapping, sample = sample_queue.get()
                if inputs is None:
                    result_queue.put((None, None, None))
                    break  # Exit if sentinel value is received

                pred = model.run_session(session=sess, inputs=inputs)
                result_queue.put((pred, offset_mapping, sample))
                pbar.update(1)

    def post_process_thread_func():
        predictions, references = [], []
        with tqdm(total=len(dataset), position=2, desc="Post Processing") as pbar:
            while True:
                pred, offset_mapping, sample = result_queue.get()
                if pred is None:
                    break  # Exit if sentinel value is received

                start_index, end_index = pred[0].argmax(-1), pred[1].argmax(-1)
                answer_start, answer_end = (
                    offset_mapping[:, start_index, 0].squeeze(),
                    offset_mapping[:, end_index, 1].squeeze(),
                )
                predictions.append(
                    {
                        "id": sample["id"],
                        "prediction_text": sample["context"][answer_start:answer_end],
                    }
                )
                references.append(
                    {
                        "id": sample["id"],
                        "answers": {
                            "answer_start": sample["answers"]["answer_start"],
                            "text": sample["answers"]["text"],
                        },
                    }
                )
                pbar.update(1)

        return predictions, references

    with ThreadPoolExecutor(max_workers=3) as executor:
        data_future = executor.submit(data_thread_func)
        inference_future = executor.submit(inference_thread_func)
        post_process_future = executor.submit(post_process_thread_func)

        data_future.result()
        inference_future.result()
        predictions, references = post_process_future.result()

    results = load_metric("squad").compute(
        predictions=predictions,
        references=references,
    )

    return (
        {"f1": results["f1"], "exact_match": results["exact_match"]}
        if results
        else {"f1": float("nan"), "exact_match": float("nan")}
    )
