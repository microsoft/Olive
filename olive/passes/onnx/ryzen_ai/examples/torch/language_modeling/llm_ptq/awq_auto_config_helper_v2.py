#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import re
import json
import torch
import argparse
from operator import add, mul
from typing import Tuple, Dict, List, Any

is_rotation_mode = False


def find_nearest_module_name(node: torch.fx.node.Node) -> str:
    module_info: str = ""
    if 'nn_module_stack' in node.meta.keys():
        name_info = [value for name, value in node.meta['nn_module_stack'].items()][-1][0]
        module_info = name_info.replace("L['self'].", "")
        module_info = re.sub(r'\[(\d+)\]\.', r'.\1.', module_info)
        module_info = re.sub(r'\[(\d+)\]', r'.\1', module_info)
    return module_info


class EasyGraph():

    def __init__(self, model: torch.nn.Module, sample_input: torch.Tensor) -> None:
        self.name_2_module = {}
        for name, module in model.named_modules():
            self.name_2_module[name] = module
        self.pair_list: Dict[str, List[str]] = {}
        self.rotation_pair_list: List[List[str]] = []

        def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Any:
            model_graph = "model_graph"
            if not os.path.exists(model_graph):
                os.makedirs(model_graph)
            graph_str = gm.print_readable()
            with open(os.path.join(model_graph, self.model.__class__.__name__ + ".txt"), 'w') as f:
                f.write(graph_str)
            self.gm = gm
            # ---------------------------------------------------------------------
            # make name convert by param memory address
            param_dict: Dict[int, str] = {}
            self.parameters_convert: Dict[str, str] = {}

            for name, module in model.named_modules():
                for param_name, param in module.named_parameters():
                    param_dict[id(param)] = name

            for name, module in self.gm.named_modules():
                for param_name, param in module.named_parameters():
                    self.parameters_convert[name.lower()] = param_dict[id(param)]
            # ---------------------------------------------------------------------
            self.find_nn_linear()
            return gm.forward

        torch._dynamo.reset()
        self.model = model
        torch.compile(model, backend=custom_backend)(sample_input, use_cache=False)

    def _is_weight_node(self, node: torch.fx.node.Node) -> bool:
        return "source_fn_stack" not in node.meta.keys() and node.op == 'get_attr'

    def _check_has_non_linear_sub_node(self, node: torch.fx.node.Node) -> bool:
        node_stack = [node]
        linear_node_list = [
            add,
            mul,
            "view",
            "getitem",
            "expand",
            "reshape",
            "contiguous",
            "transpose",
            "view",
            "permute",
            "to",
            "chunk",
            "sub",
            "int",
            "float",
            "pow",  # ?
            "cat",
            "matmul",
            "bmm",
            "truediv",
            "mean",
        ]
        node_has_const_parameters = [
            torch.nn.modules.linear.Linear, torch.nn.modules.normalization.LayerNorm, torch.nn.modules.sparse.Embedding
        ]
        while len(node_stack) > 0:
            tmp_node = node_stack.pop()
            if tmp_node.op == "output":
                continue
            node_type = [x for x in tmp_node.meta['source_fn_stack']][-1][-1]
            if any(keyword is node_type for keyword in node_has_const_parameters):
                continue
            if any(str(keyword) in str(node_type) for keyword in linear_node_list):
                node_stack.extend([k for k, v in tmp_node.users.items()])
            else:
                print("non-linear:", node_type, type(node_type))
                return False
        return True

    def _check_node(self, node: torch.fx.node.Node) -> bool:
        # 1.all sub branch are linear
        # 2.no has same sub ??
        for k, v in node.users.items():
            self._check_has_non_linear_sub_node(k)

        if all([self._check_has_non_linear_sub_node(k) for k, v in node.users.items()]):
            return True
        return False

    def get_module_name_by_node(self, node: torch.fx.node.Node) -> Any:
        if "source_fn_stack" in node.meta.keys():
            module_info = [x for x in node.meta['source_fn_stack']]
            node_name = module_info[-1][0]
            if node_name in self.parameters_convert.keys():
                return self.parameters_convert[node_name]
            elif (node_name.lower() + '.weight') in self.parameters_convert.keys():
                return self.parameters_convert[node_name + '.weight'][:-7]
            elif (node_name.lower() + '_weight') in self.parameters_convert.keys():
                return self.parameters_convert[node_name + '.weight'][:-7]
            else:
                return find_nearest_module_name(node)
        else:
            return find_nearest_module_name(node)

    def find_nn_linear(self) -> None:
        for node in self.gm.graph.nodes:
            if 'source_fn_stack' in node.meta.keys():
                module_info = [x for x in node.meta['source_fn_stack']]
                if module_info[-1][-1] is torch.nn.Linear:
                    args_name_list = node.args
                    module_name = self.get_module_name_by_node(node)
                    self.find_merge_pair(args_name_list[0], [module_name])

    def _add_pair_list(self, node: torch.fx.node.Node, prefix_model: List[str]) -> None:
        global is_rotation_mode
        prefix_model = [*prefix_model]
        module_name = self.get_module_name_by_node(node)

        if is_rotation_mode:
            if len(prefix_model) != 2:
                prefix_model.append(module_name)
                if len(node.args) == 0:
                    # const parameters
                    parent_node = [k for k in node.users.keys()][0]
                    for node_args in parent_node.args:
                        if isinstance(node_args, torch.fx.node.Node):
                            if node_args is not node:
                                return self.find_merge_pair(node_args, prefix_model)
                else:
                    if node.meta["source_fn_stack"][-1][-1] is not torch.nn.Linear:
                        if isinstance(node.args[0], torch.fx.node.Node):
                            return self.find_merge_pair(node.args[0], prefix_model)
            else:
                prefix_model.append(module_name)
                self.rotation_pair_list.append(prefix_model)
        else:
            if not self._check_node(node):
                return
            if module_name not in self.pair_list.keys():
                self.pair_list[module_name] = prefix_model
            else:
                self.pair_list[module_name].extend(prefix_model)

    def find_merge_pair(self, node: torch.fx.node.Node, prefix_model: List[str]) -> None:
        node_has_const_parameters = [
            torch.nn.modules.linear.Linear, torch.nn.modules.normalization.LayerNorm, torch.nn.modules.sparse.Embedding
        ]
        node_has_double_input_tensor = [mul, add]
        node_has_double_input_tensor_only_left = [torch.matmul, torch.bmm]
        node_has_one_input_tensor = [
            "getitem", "expand", "reshape", "contiguous", "transpose", "view", "permute", "to", "chunk"
        ]

        if not isinstance(node, torch.fx.node.Node):
            print("node may is not torch.fx.node.Node")
            return

        if node.op == "placeholder":
            return

        if self._is_weight_node(node):
            return self._add_pair_list(node, prefix_model)

        node_type = [x for x in node.meta["source_fn_stack"]][-1][-1]
        node_type_str = str(node_type).replace("torch.nn.modules", "").replace("torch", "")
        if any(keyword is node_type for keyword in node_has_const_parameters):
            return self._add_pair_list(node, prefix_model)
        elif any(keyword is node_type for keyword in node_has_double_input_tensor):
            # double input tensor
            self.find_merge_pair(node.args[0], prefix_model)
            self.find_merge_pair(node.args[1], prefix_model)
        elif any(keyword is node_type for keyword in node_has_double_input_tensor_only_left):
            # double input tensor
            self.find_merge_pair(node.args[1], prefix_model)
        elif any(keyword in node_type_str for keyword in node_has_one_input_tensor):
            # single input tensor
            self.find_merge_pair(node.args[0], prefix_model)
        else:
            print("except:", node.name, node_type)

    def dump_json(self) -> None:
        scaling_layers = []
        for k, v in self.pair_list.items():
            scaling_layers.append({"prev_op": k, "layers": v})

        json_str = json.dumps({"scaling_layers": scaling_layers})
        model_config_result = "model_config_result"
        if not os.path.exists(model_config_result):
            os.makedirs(model_config_result)
        with open(os.path.join(model_config_result, args.model_dir.replace('/', '_') + "_v2.json"), 'w') as f:
            f.write(json_str)


def format_to_json(data: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    data_dict: Dict[Any, Any] = {}
    result = []
    for k in data:
        if (k[2], k[1]) not in data_dict.keys():
            data_dict[(k[2], k[1])] = [k[0]]
        else:
            data_dict[(k[2], k[1])].append(k[0])

    data_dict2: Dict[Any, Any] = {}

    for k, v in data_dict.items():
        v.sort()
        v = tuple(v)
        if (v, k[1]) not in data_dict2.keys():
            data_dict2[(v, k[1])] = [k[0]]
        else:
            data_dict2[(v, k[1])].append(k[0])

    for k, v in data_dict2.items():
        result.append({"prev_modules": v, "norm_module": [k[1]], "next_modules": k[0]})

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        default="facebook/opt-125m",
                        help="Specify where the HuggingFace model is. This example support Llama, OPT models")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig

    def get_model(ckpt_path: str,
                  data_type: str = 'auto',
                  device: str = "cuda",
                  multi_gpu: bool = False) -> torch.nn.Module:
        config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                                 attn_implementation="eager").half().cuda()
        model.eval()
        assert (isinstance(model, torch.nn.Module))
        return model

    def main(args: argparse.Namespace) -> None:
        model = get_model(args.model_dir)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        input_data = torch.randint(0, 100, [1, 512], dtype=torch.int64).to(device)
        eg = EasyGraph(model, input_data)
        if is_rotation_mode:
            data_dict = format_to_json(eg.rotation_pair_list)  # type: ignore
            with open('tmp_rotations.json', 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=4)
        eg.dump_json()

    main(args)
