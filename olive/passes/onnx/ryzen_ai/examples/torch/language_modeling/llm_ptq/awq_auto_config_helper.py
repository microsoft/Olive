#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import json
import re
import os
import torch.nn as nn
import argparse
from typing import Tuple, Dict
from transformers import AutoModelForCausalLM, AutoConfig  # type: ignore

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ModelTypeList:
    def __init__(self, model_list):
        self.model_list = model_list

class EasyGraph():
    def __init__(self, model, sample_input) -> None:

        self.name_2_module = {}
        for name, module in model.named_modules():
            self.name_2_module[name] = module

        def custom_backend(gm: torch.fx.GraphModule, example_inputs):
            model_graph = "model_graph"
            if not os.path.exists(model_graph):
                os.makedirs(model_graph)
            graph_str = gm.print_readable()
            with open(os.path.join(model_graph, self.model.__class__.__name__ + ".txt"), 'w') as f:
                f.write(graph_str)

            self._get_connect_info(gm)
            return gm.forward

        self._connect_info: Dict = dict()
        torch._dynamo.reset()
        self.model = model
        torch.compile(model, backend=custom_backend)(sample_input, use_cache=False)

    def _get_nn_module_name(self, meta):
        module_info = ''
        if 'nn_module_stack' in meta.keys():
            name_list = [name for name in meta['nn_module_stack'].keys()]
            module_info = meta['nn_module_stack'][name_list[-1]][0].replace("L['self'].", "")
            module_info = re.sub(r'\[(\d+)\]\.', r'.\1.', module_info)
            module_info = re.sub(r'\[(\d+)\]', r'.\1', module_info)
        return module_info

    def _get_connect_info(self, gm):
        def get_next_module(node_name, res):
            can_skip_class_list = {
                torch.nn.modules.activation.ReLU
            }

            can_skip_node_list = [
                "torch.nn.modules.",
                "view",
                "transpose",
                "getitem",
                "expand",
                "reshape",
                "matmul",
                "contiguous",
                "function mul",
                "ReLU",
                "permute",
                "method bmm",
                "scaled_dot",
                "chunk",
                "getitem",
                "torch.nn.modules.activation.SiLU"
            ]

            for sub_node in self._connect_info[node_name]['children_nodes']:
                # not safe!!! need be refactor
                if self._connect_info[sub_node.name]["belong_module"] in self._connect_info[node_name]["belong_module"] or self._connect_info[sub_node.name]["class"] in can_skip_class_list:
                    # 1.for case custom nn.module, has inner functional node and nn.module node
                    # 2.some nn.module skip,this is not safe
                    if not all([name not in str(self._connect_info[sub_node.name]['class']) for name in can_skip_node_list]):
                        get_next_module(sub_node.name, res)
                else:
                    res.append((sub_node.name, self._connect_info[sub_node.name]["belong_module"]))

        for node in gm.graph.nodes:
            module_name = self._get_nn_module_name(node.meta)
            node_type = ''
            if 'source_fn_stack' in node.meta.keys():
                assert(len(node.meta['source_fn_stack']) == 1)
                node_type = node.meta['source_fn_stack'][-1][-1]
            self._connect_info[node.name] = {"node": node, "children_nodes": [node for node in node.users.keys()], "op": node.op, "belong_module": module_name, "class": node_type}

        for node_name, _ in self._connect_info.items():
            res = []
            get_next_module(node_name, res)
            self._connect_info[node_name]["children_module"] = list(set(res))

    def find_pair(self, pair_list):
        def _check_type(test_module, target):
            if isinstance(target, str):
                return test_module.__class__.__name__ == target
            if isinstance(target, ModelTypeList):
                return not all([test_module.__class__.__name__ != model_class_name for model_class_name in target.model_list])
            return isinstance(test_module, target)

        def _get_module_by_graph_node_name(node_name):
            if node_name in self._connect_info.keys():
                torch_module_name = self._connect_info[node_name]["belong_module"]
                if torch_module_name in self.name_2_module:
                    return self.name_2_module[torch_module_name], torch_module_name
                elif ('model.' + torch_module_name) in self.name_2_module:
                    return self.name_2_module['model.' + torch_module_name], 'model.' + torch_module_name
                else:
                    print(f"torch_module_name `{torch_module_name}` not find in module!")
            else:
                print(f"node name `{node_name}` not have match torch module name!")
            return None, None

        def _find_match_nodes_name(node_name, pair_list, prefix=[], all_result=[]):
            module, torch_module_name = _get_module_by_graph_node_name(node_name)
            if module is not None and _check_type(module, pair_list[0]):
                if len(pair_list) != 1:
                    for sub_node_name, _ in self._connect_info[node_name]['children_module']:
                        _find_match_nodes_name(sub_node_name, pair_list[1:], [*prefix, torch_module_name], all_result)
                else:
                    all_result.append([*prefix, torch_module_name])

        def _find(pair_list):
            for node_name in self._connect_info:
                res = []
                _find_match_nodes_name(node_name, pair_list, [], res)
                pattern_result.extend(res)

        pattern_result = []
        _find(pair_list)
        return pattern_result

    def get_nearest_parent_module(self, module_name_list):
        if len(module_name_list) == 1:
            return module_name_list[0]
        shortest_str = min(module_name_list, key=len)
        for i, char in enumerate(shortest_str):
            for other_str in module_name_list:
                if other_str[i] != char:
                    if shortest_str[i - 1] == ".":
                        return shortest_str[:i - 1]
                    return shortest_str[:i]
        if shortest_str[-1] == ".":
            shortest_str = shortest_str[:-1]
        return shortest_str

    def combine_into_json(self, ln_fcs, fc_fc):
        res = [*ln_fcs, *fc_fc]
        res = set([tuple(lst) for lst in res])
        res_not_filter = []
        prefix_count_dict = {}
        pair_dict = {}
        for item in res:
            m1 = re.match(r"(.*\.\D+)\.\d+\.(.*)", item[0])
            m2 = re.match(r"(.*\.\D+)\.\d+\.(.*)", item[1])
            if m1 is not None and m2 is not None and len(m1.groups()) == len(m2.groups()) == 2:
                if m1.group(1) not in prefix_count_dict.keys():
                    prefix_count_dict[m1.group(1)] = 1
                else:
                    prefix_count_dict[m1.group(1)] = prefix_count_dict[m1.group(1)] + 1

                if m1.group(2) not in pair_dict.keys():
                    pair_dict[m1.group(2)] = [m2.group(2)]
                else:
                    pair_dict[m1.group(2)].append(m2.group(2))
            else:
                res_not_filter.append(item)
        for k in pair_dict:
            pair_dict[k] = list(set(pair_dict[k]))

        print("not match list:", res_not_filter)
        config_dict = {}

        config_dict["model_decoder_layers"] = "" if prefix_count_dict == {} else max(prefix_count_dict, key=prefix_count_dict.get)
        config_dict["scaling_layers"] = []
        for k, v in pair_dict.items():

            config_dict["scaling_layers"].append({
                "prev_op": k,
                "layers": v,
                "inp": v[0],
                "module2inspect": self.get_nearest_parent_module(v)
            })
        json_str = json.dumps(config_dict)
        return json_str

def get_model(ckpt_path: str, data_type: str = 'auto', device: str = "cuda", multi_gpu: bool = False) -> Tuple[nn.Module, torch.dtype]:
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager").half().cuda()
    model.eval()
    return model

def main(args: argparse.Namespace) -> None:
    model = get_model(args.model_dir)
    input_data = torch.randint(0, 100, [1, 512], dtype=torch.int64).to(device)
    eg = EasyGraph(model, input_data)
    gelu_fcs = eg.find_pair([ModelTypeList(['NewGELUActivation']), torch.nn.Linear])
    ln_fcs = eg.find_pair([ModelTypeList(['LayerNorm', 'Qwen2RMSNorm', 'LlamaRMSNorm', 'MixtralRMSNorm', 'MistralRMSNorm', 'RMSNorm', 'Phi3RMSNorm']), torch.nn.Linear])
    fc_fc = eg.find_pair([torch.nn.Linear, torch.nn.Linear])
    fc_fc.extend(gelu_fcs)
    json_str = eg.combine_into_json(ln_fcs, fc_fc)
    model_config_result = "model_config_result"
    if not os.path.exists(model_config_result):
        os.makedirs(model_config_result)
    with open(os.path.join(model_config_result, args.model_dir.replace('/', '_') + ".json"), 'w') as f:
        f.write(json_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", help="Specify where the HuggingFace model is. This example support Llama, OPT models", required=True)
    args = parser.parse_args()
    main(args)
