import argparse
import onnx
from onnx import ModelProto
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error,
    _add_pre_post_qdq_pair,
    modify_model_output_intermediate_tensors)
from olive.workflows import run as olive_run
import numpy
from pathlib import Path
import json
import networkx
from collections import defaultdict, deque
from typing import Dict, Sequence, Optional
from olive.data.registry import Registry
from transformers import AutoTokenizer
from onnxruntime.quantization.quantize import CalibrationDataReader

text = "How do I locate my card?"

class DataReader(CalibrationDataReader):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        encoded_input = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors='np')
        model_input = {
            "input_ids": encoded_input.input_ids.astype(numpy.int64),
            "attention_mask": encoded_input.attention_mask.astype(numpy.int64),
            "token_type_ids": encoded_input.token_type_ids.astype(numpy.int64)
        }
        self.data = [model_input]
        self.id = 0

    def get_next(self):
        if self.id >= len(self.data): return None
        self.id += 1
        return self.data[self.id - 1]
    
    def rewind(self):
        self.id = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--float_model", type=str, default="original_model.onnx", help="Path to original floating point model"
    )
    parser.add_argument("--qdq_config", type=str, default="qdq_config.json", help="Path to qdq config")
    parser.add_argument("--error", type=float, default=10, help="Error to exclude")
    args = parser.parse_args()
    return args


def _generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"


def bfs_nodes(model: ModelProto):
    G = networkx.DiGraph()
    inputs = set([inp.name for inp in model.graph.input])
    outputs = set([out.name for out in model.graph.output])
    for node in model.graph.node:
        for input_name in node.input:
            if "Constant_" in input_name: continue
            if input_name in inputs or "_output_" in input_name or input_name in outputs:
                G.add_edge(input_name, node.name)
        for output_name in node.output:
            if "Constant_" in output_name: continue
            if output_name in outputs or "_output_" in output_name:
                G.add_edge(node.name, output_name)
    
    levels = []
    visited = set()
    queue = deque([(inp.name, 0) for inp in model.graph.input])
    input_count = defaultdict(int)

    for node in G.nodes:
        input_count[node] = G.in_degree(node)

    while queue:
        node, level = queue.popleft()
        if node not in visited:
            visited.add(node)
            if len(levels) <= level:
                levels.append([])
            levels[level].append(node)
            for neighbor in G.successors(node):
                input_count[neighbor] -= 1
                if input_count[neighbor] == 0:
                    queue.append((neighbor, level + 1))
    return levels


def augment_collect(model_path: str, input_data_reader, augment_model_path: str = None) -> Dict[str, numpy.ndarray]:
    print(f"augment_collect {model_path}")
    input_data_reader.rewind()
    augment_model_path = _generate_aug_model_path(model_path) if augment_model_path is None else augment_model_path
    modify_model_output_intermediate_tensors(model_path, augment_model_path)
    return collect_activations(augment_model_path, input_data_reader)


def create_activation_matching(
    qdq_activations: Dict[str, Sequence[numpy.ndarray]],
    float_activations: Optional[Dict[str, Sequence[numpy.ndarray]]] = None,
) -> Dict[str, Dict[str, Sequence[numpy.ndarray]]]:
    """Comparing activation values to help debugging accuracy loss due to quantization.

    This functions takes saved activations from the QDQ model and (optionally) the
    float point model, and provides a data structure for comparing:
        * from the qdq model, activation values before and after QDQ operation
        * across both models, activations from the orignal model vs the corresponding
          activations in the QDQ model

    Arg:
        qdq_activations: Output of `collect_activations`. This must be from a quantized
            model with QDQ format.
        float_activations: Output of `collect_activations`. This must be from the float
            point model.

    Returns:
        Dict for comparing pre and post quantized activation tensors. E.g.
        ```
        qdq_cmp = cmp_qdq_input_output(qdq_activations)
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])


        qdq_cmp = cmp_qdq_input_output(qdq_activations, float_activations)
        print(qdq_cmp['activation1']['float'][0])
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])
        ```
    """

    qdq_cmp: Dict[str, Dict[str, Sequence[numpy.ndarray]]] = {}
    for tensor_name, tensors in qdq_activations.items():
        pre_name = tensor_name
        pre_qdq_tensors = qdq_activations.get(pre_name)
        post_qdq_tensors = tensors
        _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)

    if not float_activations:
        return qdq_cmp

    for act_name, act_values in qdq_cmp.items():
        float_acts = float_activations.get(act_name)
        if float_acts is not None:
            act_values["float"] = float_acts

    return qdq_cmp


def compare_get(qdq_activations, float_activations, error: float, level_nodes: list[list[str]]):
    print("Comparing activations of float model vs qdq model......")
    act_matching = create_activation_matching(qdq_activations, float_activations)
    act_error = compute_activation_error(act_matching)
    return None

def main():
    # Process input parameters and setup model input data reader
    args = get_args()
    float_model_path = args.float_model

    with Path(args.qdq_config).open() as fin:
        olive_config = json.load(fin)
    qdq_model_path = Path(olive_config["output_dir"]) / "output_model" / "model.onnx"

    level_nodes = bfs_nodes(onnx.load(float_model_path))
    data_reader = DataReader()
    float_activations = augment_collect(float_model_path, data_reader)

    while True:
        olive_run(olive_config)
        qdq_activations = augment_collect(qdq_model_path, data_reader)
        error_node = compare_get(qdq_activations, float_activations, args.error, level_nodes)
        if error_node is None:
            print("No error node found")
            break
        if error_node in olive_config["passes"]["OnnxQuantization"]["nodes_to_exclude"]:
            print(f"{error_node} is already excluded")
            break
        print(f"Error node: {error_node}")
        olive_config["passes"]["OnnxQuantization"]["nodes_to_exclude"].append(error_node)

    json.dump(olive_config, (Path(args.qdq_config) / ".final.json").open("w"))


if __name__ == "__main__":
    main()