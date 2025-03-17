##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import onnx
import argparse
import copy
import numpy

to_bf16 = {"to": onnx.TensorProto.BFLOAT16}
to_fp32 = {"to": onnx.TensorProto.FLOAT}


class Node:
    def __init__(self, node) -> None:
        self.node_inputs = node.input
        self.node_outputs = node.output
        self.name = node.name
        return self

    def cast_node(self, input, output, node_name, dtype):
        cast = onnx.helper.make_node(
            "Cast", inputs=[input], outputs=[output], name=node_name, **dtype
        )
        return cast

    def update_vi(self, input, output):
        cast_out = copy.deepcopy(vi_map[input])
        cast_out.name = output
        if "fp32" in output:
            cast_out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        elif "bf16" in output:
            cast_out.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
        vi_map[cast_out.name] = cast_out


class SimplifiedLayerNormalization(Node):
    def __init__(self, model, changes, custom_ops) -> None:
        new_nodes = []
        use_custom_op = custom_ops
        for node in model.graph.node:
            if not use_custom_op:
                if node.op_type == "SimplifiedLayerNormalization":
                    curr = super().__init__(node)
                    recast_node = super().cast_node(
                        input=curr.node_outputs[0],
                        output=curr.node_outputs[0] + "_to_bf16_",
                        node_name=node.name + "_cast_bf16_",
                        dtype=to_bf16,
                    )

                    super().update_vi(
                        curr.node_outputs[0], curr.node_outputs[0] + "_to_bf16_"
                    )

                    changes[curr.node_outputs[0]] = curr.node_outputs[0] + "_to_bf16_"
                    new_nodes.append(recast_node)
                    new_nodes.append(node)
                else:
                    # No change
                    new_nodes.append(node)
            else:
                if node.op_type == "SimplifiedLayerNormalization":
                    node.op_type = "AMDSimplifiedLayerNormalization"
                    node.domain = "com.amd"

                    oname = node.output[0]
                    vi = vi_map[oname]
                    vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                    vi_map[oname] = vi

                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)


class MatMulNBits(Node):
    def __init__(self, model, changes) -> None:
        new_nodes = []
        mm_cast_idx = 0
        for node in model.graph.node:
            if node.op_type == "MatMulNBits":
                curr = super().__init__(node)

                cast_in_name = self.check_updates(changes, curr)

                cast_node = curr.cast_node(
                    input=curr.node_inputs[0],
                    output=curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx),
                    node_name=node.name + "_cast_fp32_",
                    dtype=to_fp32,
                )

                super().update_vi(
                    curr.node_inputs[0],
                    curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx),
                )

                recast_node = curr.cast_node(
                    input=cast_in_name,
                    output=curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx),
                    node_name=node.name + "_cast_bf16",
                    dtype=to_bf16,
                )

                if "lm_head" not in node.name:
                    super().update_vi(
                        cast_in_name,
                        curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx),
                    )

                if "lm_head" in node.name:
                    # print("- Node LM Head: ", node.op_type, node.name)
                    # lm_head_cast = curr.recast_node(curr.node_outputs[0], ["logits"], node, to_fp32)
                    lm_head_cast = onnx.helper.make_node(
                        "Cast",
                        inputs=[curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx)],
                        outputs=["logits"],
                        name=node.name + "_cast_logits",
                        **to_fp32
                    )
                    node.output[0] = curr.node_outputs[0] + "_bf16_"
                    new_nodes.append(lm_head_cast)

                node.input[0] = curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx)

                # adding new inputs to MatMulNBits
                # bias
                if not any("bias" in inputs for inputs in node.input):
                    # next_node_is_add=False
                    # for other_node in model.graph.node:
                    #     if other_node.op_type=="Add" and node.output[0] == other_node.input[0]:
                    #         next_node_is_add=True
                    # if not next_node_is_add:
                    node.input.append("")
                # packed_const
                node.input.append("")

                node.domain = "com.amd"

                self.update_changes(changes, curr, mm_cast_idx)

                mm_cast_idx += 1

                # empty_array = numpy.array([], dtype=numpy.uint8)

                # # Create the tensor
                # zero_point = onnx.helper.make_tensor(
                #     name='zero_point',
                #     data_type=onnx.TensorProto.UINT8,
                #     dims=empty_array.shape,
                #     vals=empty_array.tobytes()
                # )

                # node.input.append(zero_point.name)
                # i_map[zero_point.name] = zero_point

                new_nodes.append(cast_node)
                new_nodes.append(recast_node)
                new_nodes.append(node)
            else:
                # No change
                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

    def check_updates(self, changes, curr):
        if curr.node_inputs[0] in changes.keys():
            curr.node_inputs[0] = changes[curr.node_inputs[0]]
        if "lm_head" in curr.name:
            cast_in_name = curr.node_outputs[0] + "_bf16_"
        else:
            cast_in_name = curr.node_outputs[0]
        return cast_in_name

    def update_changes(self, changes, curr, mm_cast_idx):
        changes[curr.node_inputs[0]] = (
            curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx)
        )
        changes[curr.node_outputs[0]] = (
            curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx)
        )


class RotaryEmbedding(Node):
    def __init__(self, model, changes) -> None:
        new_nodes = []
        mm_cast_idx = 0
        for node in model.graph.node:
            if node.op_type == "RotaryEmbedding":
                curr = super().__init__(node)

                self.check_changes(changes, curr)

                cast_node = curr.cast_node(
                    input=curr.node_inputs[0],
                    output=curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx),
                    node_name=node.name + "_cast_fp32_",
                    dtype=to_fp32,
                )

                super().update_vi(
                    curr.node_inputs[0],
                    curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx),
                )

                cast_in_name = curr.node_outputs[0]

                recast_node = curr.cast_node(
                    input=cast_in_name,
                    output=curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx),
                    node_name=node.name + "_cast_bf16_",
                    dtype=to_bf16,
                )

                super().update_vi(
                    cast_in_name, curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx)
                )

                node.input[0] = curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx)
                self.update_changes(changes, curr, mm_cast_idx)

                mm_cast_idx += 1

                new_nodes.append(cast_node)
                new_nodes.append(recast_node)
                new_nodes.append(node)
            else:
                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

    def check_changes(self, changes, curr):
        if curr.node_inputs[0] in changes.keys():
            curr.node_inputs[0] = changes[curr.node_inputs[0]]

    def update_changes(self, changes, curr, mm_cast_idx):
        changes[curr.node_inputs[0]] = (
            curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx)
        )
        changes[curr.node_outputs[0]] = (
            curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx)
        )


class MultiHeadAttention(Node):
    def __init__(self, model, changes) -> None:
        new_nodes = []
        for node in model.graph.node:
            if node.op_type == "MultiHeadAttention":
                curr = super().__init__(node)

                node_in_arr = self.check_changes(changes, curr)

                q_cast = curr.cast_node(
                    input=node_in_arr[0],
                    output=curr.node_inputs[0] + "_to_fp32_",
                    node_name=node.name + "_q_cast_fp32_",
                    dtype=to_fp32,
                )

                super().update_vi(node_in_arr[0], curr.node_inputs[0] + "_to_fp32_")

                k_cast = curr.cast_node(
                    input=node_in_arr[1],
                    output=curr.node_inputs[1] + "_to_fp32_",
                    node_name=node.name + "_k_cast_fp32_",
                    dtype=to_fp32,
                )

                super().update_vi(node_in_arr[1], curr.node_inputs[1] + "_to_fp32_")

                v_cast = curr.cast_node(
                    input=node_in_arr[2],
                    output=curr.node_inputs[2] + "_to_fp32_",
                    node_name=node.name + "_v_cast_fp32_",
                    dtype=to_fp32,
                )

                super().update_vi(node_in_arr[2], curr.node_inputs[2] + "_to_fp32_")

                recast_node = curr.cast_node(
                    output=curr.node_outputs[0],
                    input=curr.node_outputs[0] + "_to_bf16_",
                    node_name=node.name + "_cast_bf16_",
                    dtype=to_bf16,
                )

                kvcache_inputs = [
                    curr.node_inputs[0],
                    curr.node_inputs[1],
                    curr.node_inputs[2],
                ]
                for iname in kvcache_inputs:
                    vi = vi_map[iname]
                    vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                    vi_map[iname] = vi
                for oname in curr.node_outputs:
                    if "present" not in oname:
                        vi = vi_map[oname]
                        vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                        vi_map[oname] = vi

                node.input[0] = curr.node_inputs[0] + "_to_fp32_"
                node.input[1] = curr.node_inputs[1] + "_to_fp32_"
                node.input[2] = curr.node_inputs[2] + "_to_fp32_"

                node.output[0] = curr.node_outputs[0] + "_to_bf16_"

                # Add new nodes to list
                new_nodes.append(q_cast)
                new_nodes.append(k_cast)
                new_nodes.append(v_cast)
                new_nodes.append(recast_node)
                new_nodes.append(node)
            else:
                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

    def check_changes(self, changes, curr):
        node_in_arr = []
        if curr.node_inputs[0] in changes.keys():
            node_in_0 = changes[curr.node_inputs[0]]
        else:
            node_in_0 = curr.node_inputs[0]
        if curr.node_inputs[1] in changes.keys():
            node_in_1 = changes[curr.node_inputs[1]]
        else:
            node_in_1 = curr.node_inputs[1]
        if curr.node_inputs[2] in changes.keys():
            node_in_2 = changes[curr.node_inputs[2]]
        else:
            node_in_2 = curr.node_inputs[2]
        node_in_arr.append(node_in_0)
        node_in_arr.append(node_in_1)
        node_in_arr.append(node_in_2)
        return node_in_arr


class SkipSimplifiedLayerNormalization(Node):
    def __init__(self, model, changes, custom_ops) -> None:
        use_custom_op_sslrn = custom_ops
        new_nodes = []
        for node in model.graph.node:
            if node.op_type == "SkipSimplifiedLayerNormalization":
                curr = super().__init__(node)
                if use_custom_op_sslrn:
                    node.op_type = "AMDSkipSimplifiedLayerNormalization"
                    node.domain = "com.amd"
                # Use previously changed inputs
                if curr.node_inputs[1] in changes.keys():
                    node_in = changes[curr.node_inputs[1]]
                else:
                    node_in = curr.node_inputs[1]

                if not use_custom_op_sslrn:
                    # Cast Nodes
                    cast_node = curr.cast_node(
                        input=node_in,
                        output=curr.node_inputs[1] + "_to_fp32_",
                        node_name=node.name + "_cast_fp32_",
                        dtype=to_fp32,
                    )

                    super().update_vi(node_in, curr.node_inputs[1] + "_to_fp32_")

                    recast_node = onnx.helper.make_node(
                        "Cast",
                        inputs=[curr.node_outputs[0] + "_to_bf16_"],
                        outputs=[curr.node_outputs[0]],
                        name=node.name + "_cast_bf16_",
                        **to_bf16
                    )

                    oname = node.output[0]
                    vi = vi_map[oname]
                    vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                    vi_map[oname] = vi

                    node.input[1] = curr.node_inputs[1] + "_to_fp32_"
                    node.output[0] = curr.node_outputs[0] + "_to_bf16_"
                    changes[curr.node_outputs[0]] = curr.node_outputs[0] + "_to_bf16_"

                    # Add new nodes to list
                    new_nodes.append(cast_node)
                    new_nodes.append(recast_node)
                else:
                    node.input[1] = node_in
                    if node.input[0] == "/model/embed_tokens/Gather/output_0":
                        # Cast Nodes
                        cast_node = curr.cast_node(
                            input=node.input[0],
                            output=curr.node_inputs[0] + "_to_bf16_",
                            node_name=node.name + "_gather_cast_",
                            dtype=to_bf16,
                        )

                        super().update_vi(
                            node.input[0], curr.node_inputs[0] + "_to_bf16_"
                        )

                        # node.input[0] = curr.node_inputs[0] + "_to_bf16_"
                        # Add new nodes to list
                        # new_nodes.append(cast_node)
                    # Update output data-types
                    oname = node.output[0]
                    vi = vi_map[oname]
                    vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                    vi_map[oname] = vi
                    if len(node.output) > 1:
                        oname = node.output[3]
                        vi = vi_map[oname]
                        vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                        vi_map[oname] = vi
                new_nodes.append(node)
            else:
                # No change
                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)


class Sigmoid(Node):
    def __init__(self, model, changes) -> None:
        new_nodes = []
        mm_cast_idx = 0
        for node in model.graph.node:
            if node.op_type == "Sigmoid":
                curr = super().__init__(node)
                node_in = self.check_changes(changes, curr)

                cast_node = curr.cast_node(
                    input=node_in,
                    output=curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx),
                    node_name=node.name + "_cast_fp32_" + str(mm_cast_idx),
                    dtype=to_fp32,
                )

                super().update_vi(
                    node_in, curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx)
                )

                recast_node = curr.cast_node(
                    input=curr.node_outputs[0],
                    output=curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx),
                    node_name=node.name + "_cast_bf16_" + str(mm_cast_idx),
                    dtype=to_bf16,
                )

                super().update_vi(
                    curr.node_outputs[0],
                    curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx),
                )

                node.input[0] = curr.node_inputs[0] + "_to_fp32_" + str(mm_cast_idx)
                changes[curr.node_outputs[0]] = (
                    curr.node_outputs[0] + "_to_bf16_" + str(mm_cast_idx)
                )

                mm_cast_idx += 1

                # Add new nodes to list
                new_nodes.append(cast_node)
                new_nodes.append(recast_node)
                new_nodes.append(node)
            else:
                # No change
                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

    def check_changes(self, changes, curr):
        if curr.node_inputs[0] in changes.keys():
            node_in = changes[curr.node_inputs[0]]
        else:
            node_in = curr.node_inputs[0]
        return node_in


class Mul(Node):
    def __init__(self, model, changes) -> None:
        new_nodes = []
        # mm_cast_idx = 0
        for node in model.graph.node:
            if node.op_type == "Mul" and "attn_mask_reformat" not in node.name:
                curr = super().__init__(node)

                node_in_arr = self.check_changes(changes, curr)

                # Cast Nodes
                cast_node_1 = curr.cast_node(
                    input=node_in_arr[0],
                    output=curr.node_inputs[0] + "_to_fp32_",
                    node_name=node.name + "_cast_fp32_x",
                    dtype=to_fp32,
                )

                super().update_vi(node_in_arr[0], curr.node_inputs[0] + "_to_fp32_")

                cast_node_2 = curr.cast_node(
                    input=node_in_arr[1],
                    output=curr.node_inputs[1] + "_to_fp32_",
                    node_name=node.name + "_cast_fp32_y",
                    dtype=to_fp32,
                )

                super().update_vi(node_in_arr[1], curr.node_inputs[1] + "_to_fp32_")

                recast_node = curr.cast_node(
                    output=curr.node_outputs[0] + "_to_bf16_",
                    input=curr.node_outputs[0],
                    node_name=node.name + "_cast_bf16_",
                    dtype=to_bf16,
                )

                super().update_vi(
                    curr.node_outputs[0], curr.node_outputs[0] + "_to_bf16_"
                )

                changes[curr.node_outputs[0]] = curr.node_outputs[0] + "_to_bf16_"

                for nodes in model.graph.node:
                    if nodes.op_type == "Cast":
                        if nodes.input[0] == curr.node_outputs[0]:
                            # print("updating input for :", nodes.input[0])
                            nodes.input[0] = changes[curr.node_outputs[0]]

                node.input[0] = curr.node_inputs[0] + "_to_fp32_"
                node.input[1] = curr.node_inputs[1] + "_to_fp32_"
                node.output[0] = curr.node_outputs[0]

                # Add new nodes to list
                new_nodes.append(cast_node_1)
                new_nodes.append(cast_node_2)
                new_nodes.append(recast_node)
                new_nodes.append(node)
            else:
                # No change
                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

    def check_changes(self, changes, curr):
        node_in_arr = []
        if curr.node_inputs[0] in changes.keys():
            node_in_1 = changes[curr.node_inputs[0]]
        else:
            node_in_1 = curr.node_inputs[0]
        node_in_arr.append(node_in_1)

        if curr.node_inputs[1] in changes.keys():
            node_in_2 = changes[curr.node_inputs[1]]
        else:
            node_in_2 = curr.node_inputs[1]
        node_in_arr.append(node_in_2)

        if curr.node_outputs[0] in changes.keys():
            print("Mul output should change!")
        return node_in_arr


class GroupQueryAttention(Node):
    def __init__(self, model, changes, custom_ops) -> None:
        use_custom_op_gqa = custom_ops
        split_dims = onnx.helper.make_tensor(
            "split_dims", onnx.TensorProto.INT64, [3], [4096, 1024, 1024]
        )
        new_nodes = []
        for node in model.graph.node:
            if node.op_type == "GroupQueryAttention":
                if use_custom_op_gqa:
                    node.op_type = "AMDGroupQueryAttention"
                    node.domain = "com.amd"
                curr = super().__init__(node)
                # Use previously changed inputs
                node_in_1 = self.check_changes(changes, curr)

                if use_custom_op_gqa:
                    node.input[0] = node_in_1
                else:
                    # Cast Nodes
                    cast_node_1 = curr.cast_node(
                        input=[node_in_1],
                        output=[curr.node_inputs[0] + "_to_fp32_"],
                        node_name=node.name + "_cast_fp32_",
                        dtype=to_fp32,
                    )

                    super().update_vi(node_in_1, curr.node_inputs[0] + "_to_fp32_")

                    kinput_cast = curr.cast_node(
                        input=curr.node_inputs[3],
                        output=curr.node_inputs[3] + "_to_fp32_",
                        node_name=node.name + "k_cast_fp32_",
                        dtype=to_fp32,
                    )

                    vinput_cast = curr.cast_node(
                        input=curr.node_inputs[4],
                        output=curr.node_inputs[4] + "_to_fp32_",
                        node_name=node.name + "v_cast_fp32_",
                        dtype=to_fp32,
                    )

                    recast_node = curr.cast_node(
                        output=curr.node_outputs[0],
                        inputs=curr.node_outputs[0] + "_to_bf16_",
                        node_name=node.name + "_cast_bf16_",
                        dtype=to_bf16,
                    )

                    koutput_cast = curr.cast_node(
                        "Cast",
                        outputs=curr.node_outputs[1],
                        inputs=curr.node_outputs[1] + "_to_bf16_",
                        node_name=node.name + "k_cast_bf16_",
                        dtype=to_bf16,
                    )

                    voutput_cast = curr.cast_node(
                        output=curr.node_outputs[2],
                        inputs=curr.node_outputs[2] + "_to_bf16_",
                        node_name=node.name + "v_cast_bf16_",
                        dtype=to_bf16,
                    )

                    # Update dtype
                    kvcache_inputs = [curr.node_inputs[3], curr.node_inputs[4]]
                    for iname in kvcache_inputs:
                        vi = vi_map[iname]
                        vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                        vi_map[iname] = vi
                    for oname in curr.node_outputs:
                        vi = vi_map[oname]
                        vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                        vi_map[oname] = vi

                    changes[curr.node_outputs[0]] = curr.node_outputs[0] + "_to_bf16_"

                    node.input[0] = curr.node_inputs[0] + "_to_fp32_"
                    node.input[3] = curr.node_inputs[3] + "_to_fp32_"
                    node.input[4] = curr.node_inputs[4] + "_to_fp32_"

                    node.output[0] = curr.node_outputs[0] + "_to_bf16_"
                    node.output[1] = curr.node_outputs[1] + "_to_bf16_"
                    node.output[2] = curr.node_outputs[2] + "_to_bf16_"

                    # mm_cast_idx += 1

                    # Add new nodes to list
                    new_nodes.append(cast_node_1)
                    new_nodes.append(kinput_cast)
                    new_nodes.append(vinput_cast)
                    new_nodes.append(recast_node)
                    new_nodes.append(koutput_cast)
                    new_nodes.append(voutput_cast)
                new_nodes.append(node)
            else:
                # No change
                new_nodes.append(node)
        # model.graph.initializer.append(split_dims)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

        if use_custom_op_gqa:
            self.update_vi(model)

    def check_changes(self, changes, curr):
        if curr.node_inputs[0] in changes.keys():
            node_in_1 = changes[curr.node_inputs[0]]
        else:
            node_in_1 = curr.node_inputs[0]
        return node_in_1

    def update_vi(self, model):
        for node in model.graph.node:
            if node.op_type == "AMDGroupQueryAttention":
                inames = [node.input[3], node.input[4]]
                onames = [node.output[0], node.output[1], node.output[2]]
                for iname in inames:
                    vi = vi_map[iname]
                    vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                    vi_map[iname] = vi
                for oname in onames:
                    vi = vi_map[oname]
                    vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
                    vi_map[oname] = vi
        # Update opset
        opsets = model.opset_import
        opsets.extend([onnx.helper.make_opsetid("com.amd", 1)])


class Add(Node):
    def __init__(self, model, changes) -> None:
        new_nodes = []
        for node in model.graph.node:
            if node.op_type == "Add":
                curr = super().__init__(node)
                node_in = self.check_changes(changes, curr)

                cast_node = curr.cast_node(
                    input=node_in,
                    output=curr.node_inputs[0] + "_to_fp32_",
                    node_name=node.name + "_cast_fp32_",
                    dtype=to_fp32,
                )

                super().update_vi(node_in, curr.node_inputs[0] + "_to_fp32_")

                recast_node = curr.cast_node(
                    input=curr.node_outputs[0],
                    output=curr.node_outputs[0] + "_to_bf16_",
                    node_name=node.name + "_cast_bf16_",
                    dtype=to_bf16,
                )

                super().update_vi(
                    curr.node_outputs[0], curr.node_outputs[0] + "_to_bf16_"
                )

                changes[curr.node_outputs[0]] = curr.node_outputs[0] + "_to_bf16_"
                node.input[0] = curr.node_inputs[0] + "_to_fp32_"

                new_nodes.append(cast_node)
                new_nodes.append(recast_node)
                new_nodes.append(node)
            else:
                new_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

    def check_changes(self, changes, curr):
        if curr.node_inputs[0] in changes.keys():
            node_in = changes[curr.node_inputs[0]]
        else:
            node_in = curr.node_inputs[0]
        return node_in

def get_parent_node(graph, target_node):
    for node in graph.node:
        if target_node.input[0] in node.output:
            return node
    return None

def get_child_nodes(graph, target_node):
    child_nodes = []
    for node in graph.node:
        if any(output in node.input for output in target_node.output):
            child_nodes.append(node)
    return child_nodes

def is_input_in_initializer(graph, input_name):
    initializers = {init.name for init in graph.initializer}
    return input_name in initializers

def is_first_node(graph, target_node,op_type):
    # Check if the node is a Gather node
    if target_node.op_type != op_type:
        return False

    # Check if the node is the first node in the model
    first_node = graph.node[0]
    if target_node == first_node:
        return True, 
    else:
        return False
def runner(model, custom_ops):
   
    for node in model.graph.node:
        if node.op_type == "MatMulNBits":
            mat_child_nodes=get_child_nodes(model.graph,node)
            
            for ch in mat_child_nodes:
                if ch.op_type=="Add": 
                    add_child_nodes=get_child_nodes(model.graph,ch)
                    if len(add_child_nodes)==1 and add_child_nodes[0].op_type=="GroupQueryAttention" and is_input_in_initializer( model.graph ,ch.input[1]):
                        bias_name=ch.input[1]
                        node.input.extend(["", bias_name])


           
    new_nodes = []
    add_map = {}
    for node in model.graph.node:
        if node.op_type == "Add" :
            add_child_nodes=get_child_nodes(model.graph,node)
                    
            if len(add_child_nodes)==1 and add_child_nodes[0].op_type=="GroupQueryAttention" and is_input_in_initializer( model.graph ,node.input[1]):
                    add_map[node.output[0]] = node.input[0]
        else:
            new_nodes.append(node)

    for node in new_nodes:
        if node.op_type == "GroupQueryAttention" and len(add_map) != 0:
            node.input[0] = add_map[node.input[0]]

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)

    changes = {}
    SimplifiedLayerNormalization(model, changes, custom_ops)
    MatMulNBits(model, changes)
    MultiHeadAttention(model, changes)
    Add(model, changes)
    Sigmoid(model, changes)
    Mul(model, changes)
    RotaryEmbedding(model, changes)
    GroupQueryAttention(model, changes, custom_ops)
    SkipSimplifiedLayerNormalization(model, changes, custom_ops)

    for node in model.graph.node:
        if is_first_node(model.graph,node,"Gather"):
        # if node.name == "/model/embed_tokens/Gather":  #--- TODO : Make it generic
            iname = node.input[0]
            tensor = i_map[iname]
            float_data = numpy.frombuffer(tensor.raw_data, dtype=numpy.float32)
            bf16_data = (float_data.view(numpy.uint32) >> 16).astype(numpy.uint16)

            tensor.raw_data = bf16_data.tobytes()
            tensor.data_type = onnx.TensorProto.BFLOAT16

            i_map[iname] = tensor
            output_name = node.output[0]
            vi = vi_map[output_name] #--- TODO : Make it generic
            vi.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
            vi.type.tensor_type
            vi_map[iname] = vi

        # removing g_idx
        if node.op_type == "MatMulNBits" and len(node.input) == 7:
            node.input[4] = node.input[5]
            node.input[5] = ""
            node.input.pop()

        if (
            node.op_type == "AMDSkipSimplifiedLayerNormalization"
            and "final" not in node.name
        ):
            node.output.pop(1)
            node.output.pop(1)


vi_map = {}
i_map = {}


def cast_main(model, output, custom_ops, fuse, pack):
    # load the model
    model = onnx.load(model)
    if fuse or pack:
        location = "cast_model.data"
    else:
        location = "model.data"

    # Create and update value info map
    for value_info in model.graph.value_info:
        vi_map[value_info.name] = value_info
    for value_info in model.graph.input:
        vi_map[value_info.name] = value_info
    for value_info in model.graph.output:
        vi_map[value_info.name] = value_info

    for init in model.graph.initializer:
        i_map[init.name] = init

    runner(model, custom_ops)

    value_info_list = list(vi_map.values())
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(value_info_list)

    # imap_list = list(i_map.values())
    # model.graph.ClearField("initializer")
    # model.graph.initializer.extend(imap_list)
    onnx.save_model(
        model,
        output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=1024,
        convert_attribute=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Input model", type=str, required=True)
    parser.add_argument("--output", help="Output path", type=str, required=True)
    parser.add_argument("--custom_ops", action="store_true")
    args = parser.parse_args()

    # Required from generate.py
    fuse = True
    pack = False

    cast_main(args.model, args.output, args.custom_ops, fuse, pack)
