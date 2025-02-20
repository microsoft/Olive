import argparse
import onnx
from onnx import ModelProto
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_signal_to_quantization_noice_ratio,
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
    parser.add_argument("--qdq_config", type=str, default="qdq_config.json", help="Path to qdq config")
    parser.add_argument("--error", type=float, default=20, help="Error to exclude")
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
    inits = set([init.name for init in model.graph.initializer])

    for node in model.graph.node:
        for input_name in node.input:
            if "Constant_" in input_name: continue
            if input_name in inits: continue
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
            if node in outputs: outputs.remove(node)
            for neighbor in G.successors(node):
                input_count[neighbor] -= 1
                if input_count[neighbor] == 0:
                    queue.append((neighbor, level + 1))
    assert not outputs
    return levels


def augment_collect(model_path: str, input_data_reader, augment_model_path: str = None) -> Dict[str, numpy.ndarray]:
    print(f"augment_collect {model_path}")
    input_data_reader.rewind()
    augment_model_path = _generate_aug_model_path(model_path) if augment_model_path is None else augment_model_path
    modify_model_output_intermediate_tensors(model_path, augment_model_path)
    return collect_activations(augment_model_path, input_data_reader)


def compare_get(qdq_activations, float_activations, error: float, level_nodes: list[list[str]]):
    print("Comparing activations of float model vs qdq model......")
    results = []
    for nodes in level_nodes:
        for node in nodes:
            qdq_tensor = qdq_activations.get(node)
            float_tensor = float_activations.get(node)
            if qdq_tensor is None or float_tensor is None:
                continue
            ratio = compute_signal_to_quantization_noice_ratio(float_tensor, qdq_tensor)
            if ratio < error:
                print(f"Node {node} has error {ratio}")
                index = node.find("_output_")
                results.append(node[:index])
        if results:
            return results
    return results

def main():
    # Process input parameters and setup model input data reader
    args = get_args()

    with Path(args.qdq_config).open() as fin:
        olive_config = json.load(fin)
    qdq_model_path = olive_config["output_dir"] + "/output_model/model.onnx"
    float_model_path = olive_config["input_model"]["model_path"]
    model = onnx.load(float_model_path)
    level_nodes = bfs_nodes(model)
    data_reader = DataReader()
    float_activations = augment_collect(float_model_path, data_reader)

    while True:
        olive_run(olive_config)
        qdq_activations = augment_collect(qdq_model_path, data_reader)
        error_node = compare_get(qdq_activations, float_activations, args.error, level_nodes)
        if not error_node:
            print("No error node found")
            break
        if set(error_node) & set(olive_config["passes"]["OnnxQuantization"]["nodes_to_exclude"]):
            print(f"{error_node} is already excluded")
            break
        print(f"Error node: {error_node}")
        olive_config["passes"]["OnnxQuantization"]["nodes_to_exclude"].extend(error_node)

    json.dump(olive_config, Path(args.qdq_config +  ".final.json").open("w"))


if __name__ == "__main__":
    main()