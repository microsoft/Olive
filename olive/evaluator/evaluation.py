# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import time
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from olive.common.user_module_loader import UserModuleLoader
from olive.constants import Framework
from olive.evaluator.accuracy import AUC, AccuracyScore, F1Score, Precision, Recall
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric
from olive.model import OliveModel
from olive.systems.common import Device


def evaluate_accuracy(model: OliveModel, metric: Metric, device: Device = Device.CPU) -> Dict[str, Any]:
    """
    Evaluate model accuracy according to config, return accuracy metrics
    """
    dataloader, post_func, _ = get_user_config(metric.user_config)

    preds = []
    targets = []
    inference_settings = metric.user_config.inference_settings
    model_inference_settings = inference_settings.get(model.framework.lower()) if inference_settings else None
    sess = model.prepare_session(inference_settings=model_inference_settings, device=device)

    if model.framework == Framework.ONNX:
        preds, targets = evaluate_accuracy_onnx(sess=sess, dataloader=dataloader, post_func=post_func)
    elif model.framework == Framework.PYTORCH:
        preds, targets = evaluate_accuracy_pytorch(sess=sess, dataloader=dataloader, post_func=post_func, device=device)

    elif model.framework == Framework.SNPE:
        preds, targets = evaluate_accuracy_snpe(sess=sess, dataloader=dataloader, post_func=post_func)
    elif model.framework == Framework.OPENVINO:
        preds, targets = evaluate_accuracy_openvino(sess=sess, dataloader=dataloader, post_func=post_func)
    else:
        raise Exception(
            f"Current model framework {model.framework} doesn't support metric {metric.type}.{metric.sub_type}."
        )

    metric_config = metric.metric_config
    if metric.sub_type == AccuracySubType.ACCURACY_SCORE:
        metric_res = AccuracyScore(metric_config).evaluate(preds, targets)
    elif metric.sub_type == AccuracySubType.F1_SCORE:
        metric_res = F1Score(metric_config).evaluate(preds, targets)
    elif metric.sub_type == AccuracySubType.PRECISION:
        metric_res = Precision(metric_config).evaluate(preds, targets)
    elif metric.sub_type == AccuracySubType.RECALL:
        metric_res = Recall(metric_config).evaluate(preds, targets)
    elif metric.sub_type == AccuracySubType.AUC:
        metric_res = AUC(metric_config).evaluate(preds, targets)
    else:
        raise TypeError(f"{metric.sub_type} is not a accuracy metric supported")

    return metric_res


def evaluate_latency(model: OliveModel, metric: Metric, device: Device = Device.CPU) -> Dict[str, Any]:
    """
    Evaluate model latency according to config, return latency metrics
    """
    dataloader, _, _ = get_user_config(metric.user_config)
    warmup_num = metric.metric_config.warmup_num
    repeat_test_num = metric.metric_config.repeat_test_num
    sleep_num = metric.metric_config.sleep_num

    latencies = []
    inference_settings = metric.user_config.inference_settings
    model_inference_settings = inference_settings.get(model.framework.lower()) if inference_settings else None
    sess = model.prepare_session(inference_settings=model_inference_settings, device=device)

    if model.framework == Framework.ONNX:
        latencies = evaluate_latency_onnx(
            sess=sess,
            dataloader=dataloader,
            user_config=metric.user_config,
            device=device,
            warmup_num=warmup_num,
            repeat_test_num=repeat_test_num,
            sleep_num=sleep_num,
        )
    elif model.framework == Framework.PYTORCH:
        latencies = evaluate_latency_pytorch(
            sess=sess, dataloader=dataloader, device=device, warmup_num=warmup_num, repeat_test_num=repeat_test_num
        )

    elif model.framework == Framework.SNPE:
        latencies = evaluate_latency_snpe(
            sess=sess,
            dataloader=dataloader,
            warmup_num=warmup_num,
            repeat_test_num=repeat_test_num,
            sleep_num=sleep_num,
        )
    elif model.framework == Framework.OPENVINO:
        latencies = evaluate_latency_openvino(sess=sess, dataloader=dataloader)
    else:
        raise Exception(
            f"Current model framework {model.framework} doesn't support metric {metric.type}.{metric.subtype}."
        )

    latency_metrics = {
        LatencySubType.AVG: round(sum(latencies) / len(latencies) * 1000, 5),
        LatencySubType.MAX: round(max(latencies) * 1000, 5),
        LatencySubType.MIN: round(min(latencies) * 1000, 5),
        LatencySubType.P50: round(np.percentile(latencies, 50) * 1000, 5),
        LatencySubType.P75: round(np.percentile(latencies, 75) * 1000, 5),
        LatencySubType.P90: round(np.percentile(latencies, 90) * 1000, 5),
        LatencySubType.P95: round(np.percentile(latencies, 95) * 1000, 5),
        LatencySubType.P99: round(np.percentile(latencies, 99) * 1000, 5),
        LatencySubType.P999: round(np.percentile(latencies, 99.9) * 1000, 5),
    }
    return latency_metrics[metric.sub_type]


def evaluate_custom_metric(model: OliveModel, metric: Metric, device: Device = Device.CPU):
    _, _, eval_func = get_user_config(metric.user_config)

    if not eval_func:
        raise Exception("Please specify 'evaluate_func' for custom metrics")
    return eval_func(model, metric.user_config.data_dir, metric.user_config.batch_size, device)


def evaluate_accuracy_onnx(sess, dataloader, post_func):
    preds = []
    targets = []
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    for input_data, labels in dataloader:
        if isinstance(input_data, dict):
            input_dict = {k: input_data[k].tolist() for k in input_data.keys() if k in input_names}
        else:
            input_data = input_data.tolist()
            input_dict = dict(zip(input_names, [input_data]))
        res = sess.run(input_feed=input_dict, output_names=None)
        if len(output_names) == 1:
            result = torch.Tensor(res[0])
        else:
            result = torch.Tensor(res)
        if post_func:
            outputs = post_func(result)
        else:
            outputs = result
        preds.extend(outputs.tolist())
        targets.extend(labels.data.tolist())
    return preds, targets


def evaluate_accuracy_pytorch(sess, dataloader, post_func, device):
    preds = []
    targets = []
    device = device_string_to_torch_device(device)
    if device:
        sess.to(device)
    for input_data, labels in dataloader:
        if isinstance(input_data, dict):
            if device:
                input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
            result = sess(**input_data)
        else:
            if device:
                input_data = input_data.to(device)
            result = sess(input_data)
        if post_func:
            outputs = post_func(result)
        else:
            outputs = result
        # use the list.extend instead of list.append to avoid the different sub-array has different size when
        # batch size is greater than 2 so that the residue array has different size with the batch size,
        # which will result the exception like:
        #  ValueError: expected sequence of length 128 at dim 1 (got 3)
        preds.extend(outputs.tolist())
        targets.extend(labels.data.tolist())
    sess.to(torch.device(Device.CPU))
    return preds, targets


def evaluate_accuracy_snpe(sess, dataloader, post_func):
    preds = []
    targets = []
    for data_dir, input_list, labels in dataloader:
        result = sess(input_list, data_dir)
        if post_func:
            outputs = post_func(result)
        else:
            raise ValueError("Post processing function is required for SNPE model")
        preds.extend(outputs.tolist())
        targets.extend(labels.tolist())
    return preds, targets


def evaluate_accuracy_openvino(sess, dataloader, post_func):
    preds = []
    targets = []
    for input_data, labels in dataloader:
        result = sess.infer_new_request({0: input_data})
        if post_func:
            outputs = post_func(result)
        else:
            outputs = result
        if not isinstance(labels, list):
            labels = [labels]
        preds.extend(outputs)
        targets.extend(labels)
    return preds, targets


def evaluate_latency_onnx(sess, dataloader, user_config, device, warmup_num, repeat_test_num, sleep_num):
    latencies = []
    input_names = [i.name for i in sess.get_inputs()]
    input_data, _ = next(iter(dataloader))

    if isinstance(input_data, dict):
        input_dict = {
            k: np.ascontiguousarray(input_data[k].cpu().numpy()) for k in input_data.keys() if k in input_names
        }
    else:
        input_data = np.ascontiguousarray(input_data.cpu().numpy())
        input_dict = dict(zip(input_names, [input_data]))

    if user_config.io_bind:
        io_bind_op = sess.io_binding()
        io_bind_device = "cuda" if device == "gpu" else "cpu"
        for k, v in input_dict.items():
            io_bind_op.bind_cpu_input(k, v)
        for item in sess.get_outputs():
            io_bind_op.bind_output(item.name, io_bind_device)

    for _ in range(warmup_num):
        if user_config.io_bind:
            sess.run_with_iobinding(io_bind_op)
        else:
            sess.run(input_feed=input_dict, output_names=None)

    for _ in range(repeat_test_num):
        if user_config.io_bind:
            t = time.perf_counter()
            sess.run_with_iobinding(io_bind_op)
            latencies.append(time.perf_counter() - t)
        else:
            t = time.perf_counter()
            sess.run(input_feed=input_dict, output_names=None)
            latencies.append(time.perf_counter() - t)
    return latencies


def evaluate_latency_pytorch(sess, dataloader, device, warmup_num, repeat_test_num):
    latencies = []
    input_data, _ = next(iter(dataloader))
    device = device_string_to_torch_device(device)
    if device:
        sess.to(device)
    if isinstance(input_data, dict):
        if device:
            input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
        for _ in range(warmup_num):
            sess(**input_data)
        for _ in range(repeat_test_num):
            t = time.perf_counter()
            sess(**input_data)
            latencies.append(time.perf_counter() - t)
    else:
        if device:
            input_data = input_data.to(device)
        for _ in range(warmup_num):
            sess(input_data)
        for _ in range(repeat_test_num):
            t = time.perf_counter()
            sess(input_data)
            latencies.append(time.perf_counter() - t)
    sess.to(torch.device(Device.CPU))
    return latencies


def evaluate_latency_snpe(sess, dataloader, warmup_num, repeat_test_num, sleep_num):
    data_dir, input_data, _ = next(iter(dataloader))
    total_runs = warmup_num + repeat_test_num
    results = sess(input_data, data_dir, runs=total_runs, sleep=sleep_num)
    latencies = results["latencies"]["total_inference_time"][warmup_num:]  # fmt: skip
    return latencies


def evaluate_latency_openvino(sess, dataloader):
    latencies = []
    for input_data, labels in dataloader:
        t = time.perf_counter()
        sess(input_data)
        latencies.append(time.perf_counter() - t)
    return latencies


def get_user_config(config: dict):
    user_module = None
    user_module = UserModuleLoader(config.user_script, config.script_dir)

    dataloader = None
    dataloader_func = getattr(config, "dataloader_func", None)
    dataloader = user_module.call_object(dataloader_func, config.data_dir, config.batch_size)
    if not dataloader:
        dataloader = DummyDataloader(config.input_names, config.input_shapes, config.input_types)

    post_func = getattr(config, "post_processing_func", None)
    post_func = user_module.load_object(post_func)

    eval_func = getattr(config, "evaluate_func", None)
    eval_func = user_module.load_object(eval_func)

    return dataloader, post_func, eval_func


def device_string_to_torch_device(device: Device):
    device = torch.device("cuda") if device == Device.GPU else torch.device(device)
    return device


class DummyDataloader(Dataset):
    def __init__(self, input_names, input_shapes, input_types):
        self.input_names = input_names
        self.input_shapes = input_shapes
        self.input_types = input_types

    def __len__(self):
        return 100

    def __getitem__(self, index):
        str_to_type = {"float32": torch.float32, "float16": torch.float16, "int32": torch.int32, "int64": torch.int64}
        input_types = []
        if self.input_types:
            for input_type in self.input_types:
                input_types.append(str_to_type[input_type])
        else:
            for _ in range(len(self.input_names)):
                input_types.append(torch.float32)
        if len(self.input_names) == 1:
            dummy_inputs = torch.ones(self.input_shapes[0], dtype=input_types[0])
        else:
            dummy_inputs = {}
            for input_name, input_shape, input_type in zip(self.input_names, self.input_shapes, input_types):
                dummy_inputs.update({input_name: torch.ones(input_shape, dtype=input_type)})
        label = 0
        return dummy_inputs, label
