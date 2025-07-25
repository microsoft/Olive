import logging

import numpy as np
import onnx
import onnx_ir as ir
import onnxscript
from onnx import helper, numpy_helper
from onnxscript.rewriter import pattern

logger = logging.getLogger(__name__)

"""
ALL transform function starts with transform_ and called from GraphSurgeries pass.
ASSUMPTIONS:
- quantized values are either uint8 or uint16
Hardcoded:
- Transpose perm=[0,3,2,1]
- initializers to 4D
  - 1D [K] -> [1x1x1xK]
  - 2D [CxK] -> [KxCx1x1]
  - 3D insert 1 at 3rd dimension
- Reshape-ReduceSum to Slice-ReduceSum-Concat axes
- Expand 3D to 4D. Insert 1 at 0th dimension
- output Squeeze if Squeeze node exists axes = [0, 3]
- Flatten to reshape with axes [1, 1, 1, -1]
- Unsqueeze
    - From 3D axes=[1]
    - From 2D axes=[0, -1]
- Squeeze
    - Update existing squeeze: [0, 3]
    - New squeeze: [1, 2]
- Gather
    - indices = [2]
- ReduceSum axes = [2] or keep [-1]
"""


def _get_tensor_shape_map(graph_value_info):
    tensor_name_dim_map = {}
    for value_info in graph_value_info:
        tensor_type = value_info.type.tensor_type
        shape = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append(0)
                    break
        tensor_name_dim_map[value_info.name] = shape
    return tensor_name_dim_map


def _get_initializer_by_name(model, init_name):
    for init in model.graph.initializer:
        if init.name == init_name:
            return numpy_helper.to_array(init)
    return None


def _calculate_clip_range(node, model):
    x_scale = _get_initializer_by_name(model, node.input[1])
    x_zero_point = _get_initializer_by_name(model, node.input[2])
    assert x_scale, f"{node.name} should have x_scale value"
    int_max = np.int32(65535 if x_zero_point.dtype == np.uint16 else 255)
    if x_zero_point is None:
        logger.info("x_zero_point is None!")
        x_zero_point = np.array(0, dtype=np.int32)
    else:
        x_zero_point = x_zero_point.astype(np.int32)
    clip_min = ((0 - x_zero_point) * x_scale).astype(np.float32)
    clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
    return clip_min, clip_max


def transform_matmul_to_transpose_conv_transpose(model):
    cnt = 0
    graph = model.graph
    tensor_name_dim_map = _get_tensor_shape_map(graph.value_info)
    initializer_dim_map = {init.name: len(init.dims) for init in graph.initializer}
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            need_transform = False
            for node_input in node.input:
                # Input is either in initializer or value_info
                if (node_input in initializer_dim_map and initializer_dim_map[node_input] != 4) or (
                    node_input in tensor_name_dim_map and len(tensor_name_dim_map[node_input]) != 4
                ):
                    need_transform = True
                    break
            if need_transform:
                nodes_to_remove.append(node)
            else:
                logger.info("Skipped MatMul/Gemm node %s as both inputs are 4D", node.name)

    initializers_to_remove = []
    initializers_to_add = []

    for node in nodes_to_remove:
        matmul_node_name = node.name
        nodes_to_add = []

        # Check the first input to Gemm/MatMul.
        # If C (channel) dimesion is 1, needs to add Transposes around Conv.
        # Otherwise, no need to add Transposes for now.
        def check_to_apply_transpose(conv_node):
            node_input = conv_node.input[0]
            if node_input in tensor_name_dim_map:
                shape = tensor_name_dim_map[node_input]
                if (
                    len(shape) == 2 and shape[0] != 1 and shape[1] != 1
                ):  # The 2D tesnor (MxN, width and height) will later adding leading dimensions to 4D with 1x1xMxN
                    return True
                if len(shape) == 3 and shape[0] == 1:  # C == 1
                    return True
                if len(shape) == 4 and shape[1] == 1:  # C == 1
                    return True
                if (
                    len(shape) == 4 and shape[2] != 1 and shape[3] != 1
                ):  # In some cases, C != 1 but it still needs the transpose
                    return True
            # Otherwise, we considered the input tensor has been "transposed" to perform 1x1 Conv
            return False

        need_to_apply_transpose = check_to_apply_transpose(node)
        graph.node.remove(node)

        # Add Transpose if needed
        if need_to_apply_transpose:
            transpose_before_node = helper.make_node(
                "Transpose",
                inputs=[node.input[0]],
                outputs=[matmul_node_name + "_transpose_before_output"],
                name=matmul_node_name + "_transpose_before",
                perm=[0, 3, 2, 1],
            )
            nodes_to_add.append(transpose_before_node)
            conv_inputs = [matmul_node_name + "_transpose_before_output", node.input[1]]
        else:
            conv_inputs = [node.input[0], node.input[1]]

        if len(node.input) == 3:  # Gemm has optional third input
            conv_inputs.append(node.input[2])

        # Add Conv
        conv_node = helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[matmul_node_name + "_transpose_output"] if need_to_apply_transpose else node.output,
            name=matmul_node_name + "_conv",
        )
        nodes_to_add.append(conv_node)

        # Update Conv's weight to 4D if needed
        def update_initializers(graph, name, initializers_to_remove, initializers_to_add, trans_b=False):
            for initializer in graph.initializer:
                if initializer.name != name:
                    continue
                # 2D [CxK] -> [KxCx1x1] if TransB != 1
                # 2D [CxK] -> [CxKx1x1] if TransB == 1
                c, k = initializer.dims[0], initializer.dims[1]
                init_arr = numpy_helper.to_array(initializer)
                if not trans_b:
                    init_arr = init_arr.T
                    shape = (k, c, 1, 1)
                else:
                    shape = (c, k, 1, 1)
                reshaped_arr = np.reshape(init_arr, shape)
                new_initializer = numpy_helper.from_array(reshaped_arr, initializer.name)
                initializers_to_remove.append(initializer)
                initializers_to_add.append(new_initializer)

        # Check input B of Gemm is transposed or not
        trans_b = False
        if node.op_type == "Gemm":
            for attr in node.attribute:
                if attr.name == "transB" and attr.i == 1:
                    trans_b = True
                    break

        update_initializers(graph, node.input[1], initializers_to_remove, initializers_to_add, trans_b)

        if need_to_apply_transpose:
            transpose_after_node = helper.make_node(
                "Transpose",
                inputs=[matmul_node_name + "_transpose_output"],
                outputs=node.output,
                name=matmul_node_name + "_transpose_after",
                perm=[0, 3, 2, 1],
            )
            nodes_to_add.append(transpose_after_node)

        graph.node.extend(nodes_to_add)
        cnt += 1

    [graph.initializer.remove(init) for init in initializers_to_remove]
    [graph.initializer.append(init) for init in initializers_to_add]

    logger.info("Replaced %d MatMul nodes with Transpose-Conv-Transpose nodes", cnt)


def transform_remove_intermediary_squeeze_and_unsqueeze(model):
    graph = model.graph

    # Get all input names
    input_names = {graph_input.name for graph_input in graph.input}

    # Track nodes to remove
    nodes_to_remove = []

    # Create a mapping of node output names to their consumers
    output_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)

    # Find all Unsqueeze or Squeeze nodes not directly connected to inputs
    for node in graph.node:
        if (node.op_type in ("Unsqueeze", "Squeeze")) and node.input[0] not in input_names:
            # Get the input (source) of this Unsqueeze
            unsqueeze_input = node.input[0]

            # Get the output of this Unsqueeze
            unsqueeze_output = node.output[0]

            # Get all consumers of this Unsqueeze's output
            consumers = output_to_consumers.get(unsqueeze_output, [])

            # Rewire the graph to bypass this Unsqueeze
            for consumer in consumers:
                for i, input_name in enumerate(consumer.input):
                    if input_name == unsqueeze_output:
                        consumer.input[i] = unsqueeze_input

            # Mark this Unsqueeze for removal
            nodes_to_remove.append(node)

    # Remove the Unsqueeze or Squeeze nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    logger.info("Removed %d intermediary Unsqueeze or Squeeze operations", len(nodes_to_remove))


def transform_qdq_to_clip(model):
    cnt = 0
    qualin_name_node_map, deqlin_name_node_map = (
        {},
        {},
    )  # quantize_output_name -> quantize_node, dequantize_input_name -> dequantize_node
    graph = model.graph
    clip_range = {}  # deq_input_name -> (clip_min, clip_max)
    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # check no output or if multiple connected nodes
            if not node.output[0] or len(node.output) > 1:
                continue
            qualin_name_node_map[node.output[0]] = node
        if node.op_type == "DequantizeLinear":
            if not node.input[0]:
                continue
            assert len(node.input) == 3
            deqlin_name_node_map[node.input[0]] = node
            x_scale = _get_initializer_by_name(model, node.input[1])
            x_zero_point = _get_initializer_by_name(model, node.input[2])
            assert x_scale, f"{node.name} should have x_scale value"
            int_max = np.int32(65535 if x_zero_point.dtype == np.uint16 else 255)
            if x_zero_point is None:
                logger.info("x_zero_point is None!")
                x_zero_point = np.array(0, dtype=np.int32)
            else:
                x_zero_point = x_zero_point.astype(np.int32)
            clip_min = ((0 - x_zero_point) * x_scale).astype(np.float32)
            clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
            clip_range[node.input[0]] = (clip_min, clip_max)
    subgraph_output_input_map = {}  # deqlin output -> qualin input
    clip_nodes_to_add = []
    for q_output, qualin_node in qualin_name_node_map.items():
        if q_output in deqlin_name_node_map:
            deqlin_node = deqlin_name_node_map[q_output]
            graph.node.remove(qualin_node)
            graph.node.remove(deqlin_node)
            subgraph_output_input_map[deqlin_node.output[0]] = qualin_node.input[0]
            clip_min, clip_max = clip_range[q_output]
            clip_min_init = numpy_helper.from_array(np.array([clip_min]), deqlin_node.name + "_clip_min")
            clip_max_init = numpy_helper.from_array(np.array([clip_max]), deqlin_node.name + "_clip_max")
            model.graph.initializer.extend([clip_min_init, clip_max_init])
            clip_node = helper.make_node(
                "Clip",
                inputs=[
                    qualin_node.input[0],
                    deqlin_node.name + "_clip_min",
                    deqlin_node.name + "_clip_max",
                ],  # data, axes
                outputs=[deqlin_node.output[0]],
                name=deqlin_node.name + "_clip",
            )
            clip_nodes_to_add.append(clip_node)
            cnt += 1

    for clip_node in clip_nodes_to_add:
        graph.node.append(clip_node)
    logger.info("Replaced %d QuantizeLinear and DequantizeLinear pairs with Clip", cnt)


def transform_remove_qdq(model, keep_clip_after_inputs=False):
    q_output_to_q_node_map, dq_input_to_dq_node_map = {}, {}  # q_output_name -> q_node, dq_input_name -> dq_node
    graph = model.graph

    # Collect all the candidate Q and DQ nodes
    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # check no output or if multiple connected nodes
            if not node.output[0] or len(node.output) > 1:
                continue
            q_output_to_q_node_map[node.output[0]] = node
        elif node.op_type == "DequantizeLinear":
            if not node.input[0]:
                continue
            dq_input_to_dq_node_map[node.input[0]] = node

    qdq_node_pair_output_to_input_map = {}  # qd output -> q input
    qdq_node_pair_input_to_output_map = {}  # q input -> qd output
    clip_nodes_to_add = []
    cnt = 0

    # Find the Q and DQ node pairs and remove them.
    # There are following scenarios:
    # 1) Node --> Q --> DQ --> Node'
    # 2) graph input --> Q --> DQ --> Node
    # 3) Node --> Q --> DQ --> graph output
    # 4) Node --> Q --> DQ ... --> Q --> DQ --> Node'
    for q_output, q_node in q_output_to_q_node_map.items():
        if q_output in dq_input_to_dq_node_map:
            dq_node = dq_input_to_dq_node_map[q_output]

            if keep_clip_after_inputs and q_node.input[0] in [graph_input.name for graph_input in model.graph.input]:
                # Calculate clip range if we want to keep the clip after inputs
                clip_min, clip_max = _calculate_clip_range(dq_node, model)
                clip_min_init = numpy_helper.from_array(np.array([clip_min]), dq_node.name + "_clip_min")
                clip_max_init = numpy_helper.from_array(np.array([clip_max]), dq_node.name + "_clip_max")
                model.graph.initializer.extend([clip_min_init, clip_max_init])
                clip_node = helper.make_node(
                    "Clip",
                    inputs=[q_node.input[0], dq_node.name + "_clip_min", dq_node.name + "_clip_max"],  # data, axes
                    outputs=[dq_node.output[0]],
                    name=dq_node.name + "_clip",
                )
                clip_nodes_to_add.append(clip_node)
            else:
                qdq_node_pair_output_to_input_map[dq_node.output[0]] = q_node.input[0]
                qdq_node_pair_input_to_output_map[q_node.input[0]] = dq_node.output[0]

            graph.node.remove(q_node)
            graph.node.remove(dq_node)
            cnt += 1

    for clip_node in clip_nodes_to_add:
        graph.node.append(clip_node)

    # Make sure the predecessor and successor to the Q and DQ node pair are connected.
    # e.g. Node --> Q --> DQ --> Node'  =>  Node --> Node'
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            # This node's ith input is from a DQ node.
            #
            # Please note that the following while loop is for handling the #4 case mentioned above where
            # there could be multiple (ususally no more than 2) consecutive Q and DQ node pairs which are connected
            # e.g. Node --> Q --> DQ --> Q --> DQ --> Node'
            if input_name in qdq_node_pair_output_to_input_map:
                qdq_node_pair_output = input_name
                num_connected_qdq_node_pair = 0
                while True:
                    qdq_node_pair_input = qdq_node_pair_output_to_input_map[qdq_node_pair_output]
                    if qdq_node_pair_input not in qdq_node_pair_output_to_input_map:
                        node.input[i] = qdq_node_pair_input
                        break
                    qdq_node_pair_output = qdq_node_pair_input

                    num_connected_qdq_node_pair += 1
                    if num_connected_qdq_node_pair > 5:
                        logger.error(
                            "Number of connected QDQ node pair is %d which is not normal.", num_connected_qdq_node_pair
                        )
                        raise ValueError("Number of connected QDQ node pair is not normal.")

        for i, output_name in enumerate(node.output):
            # This node's ith output is consumed by a Q node.
            if output_name in qdq_node_pair_input_to_output_map:
                graph_output_candidate = qdq_node_pair_input_to_output_map[output_name]
                if graph_output_candidate in [output.name for output in graph.output]:
                    node.output[i] = graph_output_candidate

    logger.info("Removed %d QuantizeLinear and DequantizeLinear pairs", cnt)


def transform_remove_deqlin(model):
    def dequantize_initializer(deq_initializers, node_input, graph):
        x = None
        scale = None
        zero_point = None
        for init in deq_initializers:
            if init.name == node_input[0]:
                x = numpy_helper.to_array(init).astype(np.int32)
            if init.name == node_input[1]:
                scale = numpy_helper.to_array(init).astype(np.float32)
            if init.name == node_input[2]:
                zero_point = numpy_helper.to_array(init).astype(np.int32)
        if x is None or scale is None or zero_point is None:
            raise ValueError("Missing required initializers for dequantization")
        return ((x - zero_point) * scale).astype(
            np.float32
        )  # Might have type issues: X, zero_point uint16, scale float32

    cnt = 0
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    deqlin_output_initializer_mapping = {}
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and len(node.input) > 0 and node.input[0] in initializer_names:
            deq_initializers = [init for init in graph.initializer if init.name in node.input]
            dequantized_arr = dequantize_initializer(deq_initializers, node.input, graph).astype(np.float32)
            dequantized_init = numpy_helper.from_array(dequantized_arr, name=node.input[0])
            # remove all initializers in deq_initializers
            for init in deq_initializers:
                graph.initializer.remove(init)
            graph.initializer.append(dequantized_init)
            deqlin_output_initializer_mapping[node.output[0]] = node.input[0]
            nodes_to_remove.append(node)
        # Replace the node input after DequantizeLinear with initializer
        for i, node_input in enumerate(node.input):
            if node_input in deqlin_output_initializer_mapping:
                node.input[i] = deqlin_output_initializer_mapping[node_input]

    # Remove the nodes
    for node in nodes_to_remove:
        graph.node.remove(node)
        cnt += 1
    logger.info("Removed %d DequantizeLinear nodes", cnt)


def transform_non4d_model_inputs(model):
    cnt = 0
    graph = model.graph
    unsqueeze_to_add = []
    for graph_input in graph.input:
        dims = len(graph_input.type.tensor_type.shape.dim)
        if dims in (2, 3):
            unsqueeze_arr = np.array([0, -1], dtype=np.int64) if dims == 2 else np.array([1], dtype=np.int64)

            # Check if there's already an Unsqueeze node for this input
            has_unsqueeze = False
            for node in graph.node:
                if len(node.input) > 0 and node.input[0] == graph_input.name and node.op_type == "Unsqueeze":
                    has_unsqueeze = True
                    existing_unsqueeze_axes = None
                    for init in graph.initializer:
                        if init.name == node.input[1]:
                            existing_unsqueeze_axes = numpy_helper.to_array(init)
                            break
                    if existing_unsqueeze_axes is not None:
                        if len(existing_unsqueeze_axes) + dims == 4:
                            continue
                        new_unsqueeze_axes_name = node.name + "unsqueeze_axes_transformed"
                        unsqueeze_axes = numpy_helper.from_array(unsqueeze_arr, new_unsqueeze_axes_name)
                        node.input[1] = new_unsqueeze_axes_name
                        graph.initializer.append(unsqueeze_axes)
                    cnt += 1
                    break

            # If no Unsqueeze exists and this input is used by other nodes, create one
            if not has_unsqueeze:
                input_is_used = False
                for node in graph.node:
                    if graph_input.name in node.input:
                        input_is_used = True
                        break

                if input_is_used:
                    new_unsqueeze_axes_name = graph_input.name + "_unsqueeze_axes"
                    unsqueezed_input_name = graph_input.name + "_unsqueeze_input"
                    unsqueeze_axes = numpy_helper.from_array(unsqueeze_arr, new_unsqueeze_axes_name)
                    graph.initializer.append(unsqueeze_axes)
                    unsqueeze_node = helper.make_node(
                        "Unsqueeze",
                        inputs=[graph_input.name, new_unsqueeze_axes_name],
                        outputs=[unsqueezed_input_name],
                        name=graph_input.name + "_unsqueeze",
                    )
                    unsqueeze_to_add.append(unsqueeze_node)

                    # Update all nodes that use this input
                    for node in graph.node:
                        for i, node_input_name in enumerate(node.input):
                            if node_input_name == graph_input.name:
                                node.input[i] = unsqueezed_input_name
                    cnt += 1

    # Add all new unsqueeze nodes
    for node in unsqueeze_to_add:
        graph.node.append(node)
    logger.info("Added/Updated %d Unsqueeze node", cnt)


def transform_non4d_model_outputs(model):
    def is_squeeze_clip_output_pattern(squeeze_node, model):
        squeeze_output = squeeze_node.output[0]
        graph_output_names = [output.name for output in model.graph.output]
        for node in graph.node:
            if node.op_type == "Clip" and node.input[0] == squeeze_output and node.output[0] in graph_output_names:
                return True
        return False

    cnt = 0
    graph = model.graph
    for graph_output in graph.output:
        update_existing_squeeze = False
        output_dim = len(graph_output.type.tensor_type.shape.dim)
        if 1 < output_dim < 4:
            for node in graph.node:
                if node.op_type == "Squeeze" and (
                    node.output[0] == graph_output.name or is_squeeze_clip_output_pattern(node, model)
                ):
                    update_existing_squeeze = True
                    squeeze_axes_name = graph_output.name + "_squeeze_axes"
                    if output_dim == 2:
                        squeeze_axes_arr = np.array([0, 3], dtype=np.int64)
                    squeeze_axes = numpy_helper.from_array(squeeze_axes_arr, squeeze_axes_name)
                    node.input[1] = squeeze_axes_name
                    graph.initializer.append(squeeze_axes)
            if not update_existing_squeeze:
                squeeze_input_name = graph_output.name + "_squeeze_input"
                squeeze_axes_name = graph_output.name + "_squeeze_axes"
                if output_dim == 2:
                    squeeze_axes_arr = np.array([0, 1], dtype=np.int64)
                elif output_dim == 3:
                    squeeze_axes_arr = np.array([2], dtype=np.int64)
                squeeze_axes = numpy_helper.from_array(squeeze_axes_arr, squeeze_axes_name)
                graph.initializer.append(squeeze_axes)
                squeeze_node = helper.make_node(
                    "Squeeze",
                    inputs=[squeeze_input_name, squeeze_axes_name],
                    outputs=[graph_output.name],
                    name=graph_output.name + "_squeeze",
                )
                # Change output of previous node to squeeze_input_name
                for node in graph.node:
                    for i, node_output_name in enumerate(node.output):
                        if node_output_name == graph_output.name:
                            node.output[i] = squeeze_input_name
                    # Handle intermediate nodes that have graph_output as input
                    for i, node_input_name in enumerate(node.input):
                        if node_input_name == graph_output.name:
                            node.input[i] = squeeze_input_name
                graph.node.append(squeeze_node)
                cnt += 1
    logger.info("Added %d Squeeze nodes for graph output", cnt)


def transform_standalone_reducesum(model):
    graph = model.graph
    for node in graph.node:
        # A single ReduceSum, not transformed with reshape_reducesum_to_slice_reducesum_concat
        if node.op_type == "ReduceSum" and "transformed" not in node.name:
            # set keepdims = 1
            for attr in node.attribute:
                if attr.name == "keepdims":
                    attr.i = 1
            axes_initializer = node.input[1]  # axes
            new_reducesum_axes = None
            for initializer in graph.initializer:
                if initializer.name == axes_initializer:
                    current_axes = numpy_helper.to_array(initializer)
                    # Only set axes to [2] if current_axes is not [-1]
                    if not (current_axes.size == 1 and current_axes[0] == -1):
                        reducesum_axes_name = node.name + "_axes"
                        new_reducesum_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), reducesum_axes_name)
                        node.input[1] = reducesum_axes_name
            if new_reducesum_axes is not None:
                graph.initializer.append(new_reducesum_axes)


def transform_gather(model):
    cnt = 0
    initializer_to_remove = None
    for node in model.graph.node:
        if node.op_type == "Gather":
            cnt += 1
            indices_initializer_name = node.input[1]
            existing_indices = None
            for initializer in model.graph.initializer:
                if initializer.name == indices_initializer_name:
                    existing_indices = numpy_helper.to_array(initializer)
                    initializer_to_remove = initializer
                    break
            if existing_indices is not None:
                if existing_indices.ndim == 0:
                    indices_array = np.array([existing_indices.item()], dtype=existing_indices.dtype)
                else:
                    indices_array = existing_indices
            else:
                indices_array = np.array([0], dtype=np.int64)
            indices_initializer = numpy_helper.from_array(indices_array, name=indices_initializer_name)
            for attr in node.attribute:
                if attr.name == "axis" and attr.i == 1:
                    attr.i = 2
            # Remove the old initializer if it exists
            if initializer_to_remove is not None:
                model.graph.initializer.remove(initializer_to_remove)

            # Add the new initializer
            model.graph.initializer.append(indices_initializer)
    logger.info("Updated %d Gather axis", cnt)


def transform_gatherelements(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "GatherElements":
            cnt += 1
            indices_initializer_name = node.input[1]
            existing_indices = None
            for initializer in model.graph.initializer:
                if initializer.name == indices_initializer_name:
                    existing_indices = numpy_helper.to_array(initializer)
                    initializer_to_remove = initializer
                    break
            if existing_indices is not None:
                if existing_indices.ndim == 0:
                    indices_array = np.array([existing_indices.item()], dtype=existing_indices.dtype)
                elif existing_indices.ndim == 3:
                    # Handle 3D indices, reshape to 4D by adding dimension at front
                    logger.info(
                        "Reshaping 3D indices %s from %s to 4D", initializer_to_remove.name, existing_indices.shape
                    )
                    indices_array = np.expand_dims(existing_indices, axis=0)  # Eg.[12,64,64]->[1,12,64,64]
                else:
                    indices_array = existing_indices

                indices_initializer = numpy_helper.from_array(indices_array, name=indices_initializer_name)
                # Remove the old initializer if it exists
                if initializer_to_remove is not None:
                    model.graph.initializer.remove(initializer_to_remove)

                # Add the new initializer
                model.graph.initializer.append(indices_initializer)
    logger.info("Updated %d GatherElements indices", cnt)


def transform_non4d_initializers(model):
    need_to_expand_4d_init_names = []
    skip_init_names = []
    unary_dim_at_front_init_names = []
    for node in model.graph.node:
        if node.op_type in ["Div", "Sub", "Mul"]:
            need_to_expand_4d_init_names = list(node.input)
        if node.op_type in ["MatMul"]:
            skip_init_names = list(node.input)
        if node.op_type in ["Gemm"]:
            unary_dim_at_front_init_names = list(node.input)

    initializers_to_add = []
    initializer_to_remove = []
    for initializer in model.graph.initializer:
        if len(initializer.dims) == 1 and initializer.name in need_to_expand_4d_init_names:
            # 1D: [K] -> [1x1x1xK]
            initializer.dims.insert(0, 1)
            initializer.dims.insert(0, 1)
            initializer.dims.insert(0, 1)
        elif len(initializer.dims) == 2 and initializer.name in unary_dim_at_front_init_names:
            # 2D: [K, C] -> [1, 1, K, C]
            new_dims = [1, 1, *list(initializer.dims)]
            initializer.dims[:] = new_dims
        elif len(initializer.dims) == 2 and initializer.name not in skip_init_names:
            # 2D [CxK] -> [KxCx1x1]
            c, k = initializer.dims[0], initializer.dims[1]
            init_arr = numpy_helper.to_array(initializer)
            transposed_arr = init_arr.T
            reshaped_arr = np.reshape(transposed_arr, (k, c, 1, 1))
            new_initializer = numpy_helper.from_array(reshaped_arr, initializer.name)
            initializer_to_remove.append(initializer)
            initializers_to_add.append(new_initializer)
            initializer.dims[0], initializer.dims[1] = k, c
            initializer.dims.insert(2, 1)
            initializer.dims.insert(3, 1)
            # new_dims = [1] + list(initializer.dims)
        elif len(initializer.dims) == 3:
            # initializer.dims.insert(0, 1)
            new_dims = [1, *list(initializer.dims)]
            initializer.dims[:] = new_dims

    [model.graph.initializer.remove(init) for init in initializer_to_remove]
    [model.graph.initializer.append(init) for init in initializers_to_add]


def transform_remove_all_tensor_value_shapes(model):
    for value_info in model.graph.value_info:
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField("shape"):
            tensor_type.ClearField("shape")


def transform_non4d_reshape(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Reshape":
            # Check if axes
            for init in model.graph.initializer:
                if init.name == node.input[1] and init.dims[0] == 3:
                    # eg [-1, 512, 768] or [0,0,-1]
                    old_shape = numpy_helper.to_array(init)
                    init.dims[0] = 4
                    # insert at the first non-zero index
                    idx = next(i for i, v in enumerate(old_shape) if v != 0)
                    new_shape = np.insert(old_shape, idx, 1).astype(np.int64)
                    new_init = numpy_helper.from_array(new_shape, name=init.name)
                    model.graph.initializer.remove(init)
                    model.graph.initializer.append(new_init)
                    cnt += 1
    logger.info("Updated %d non4D Reshape nodes", cnt)


def transform_non4d_expand(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Expand":
            for init in model.graph.initializer:
                if init.name == node.input[1] and init.dims[0] == 3:
                    old_shape = numpy_helper.to_array(init)
                    init.dims[0] = 4
                    new_shape = np.insert(old_shape, 0, 1).astype(np.int64)
                    new_init = numpy_helper.from_array(new_shape, name=init.name)
                    model.graph.initializer.remove(init)
                    model.graph.initializer.append(new_init)
                    cnt += 1
    logger.info("Updated %d non4D Expand nodes", cnt)


def transform_non4d_transpose(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm" and len(attr.ints) == 2:
                    old_perm = list(attr.ints)
                    new_perm = [0, 1, old_perm[0] + 2, old_perm[1] + 2]
                    attr.ints[:] = new_perm
                    cnt += 1
                elif attr.name == "perm" and len(attr.ints) == 3:
                    # [0, 2, 1] -> [0, 3, 2, 1]
                    old_perm = list(attr.ints)
                    new_perm = [0, old_perm[0] + 1, old_perm[1] + 1, old_perm[2] + 1]
                    attr.ints[:] = new_perm
                    cnt += 1

    logger.info("Updated %d non4D Transpose nodes", cnt)


def transform_non4d_slice(model):
    cnt = 0
    for node in model.graph.node:
        # Skip transformed nodes
        if node.op_type == "Slice" and len(node.input) > 3 and "transformed_" not in node.input[3]:
            new_init_to_add = None
            for init in model.graph.initializer:
                if init.name == node.input[3] and init.dims[0] == 1:
                    new_init_name = node.name + "_axes"
                    new_init_to_add = numpy_helper.from_array(np.array([-1], dtype=np.int64), name=new_init_name)
                    node.input[3] = new_init_name
                    cnt += 1
            if new_init_to_add is not None:
                model.graph.initializer.append(new_init_to_add)
    logger.info("Updated %d non4D Slice axes", cnt)


def transform_non4d_lpnorm(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "LpNormalization":
            for attr in node.attribute:
                if attr.name == "axis":
                    attr.i = -1
            cnt += 1
    logger.info("Updated %d non4D LpNormalization axes", cnt)


def transform_flatten(model):
    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == "Flatten":
            reshape_axes = numpy_helper.from_array(
                np.array([1, 1, 1, -1], dtype=np.int64), name=node.name + "_reshape_axes"
            )
            reshape_node = helper.make_node(
                "Reshape", inputs=[node.input[0], reshape_axes.name], outputs=node.output, name=node.name + "_reshape"
            )
            nodes_to_remove.append(node)
            model.graph.initializer.append(reshape_axes)
            model.graph.node.append(reshape_node)
    # Remove flatten node(s)
    for node in nodes_to_remove:
        model.graph.node.remove(node)


def reshape_reducesum_pattern(op, x, shape, axes):
    reshape_output = op.Reshape(x, shape)
    return op.ReduceSum(reshape_output, axes)


def slice_reducesum_concat(op, x, shape, axes):
    slice_0_starts = op.initializer(ir.tensor([0], dtype=ir.DataType.INT64, name=x.name + "_slice_0_starts"))
    slice_0_ends = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_0_ends"))
    slice_1_starts = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_1_starts"))
    slice_1_ends = op.initializer(ir.tensor([8], dtype=ir.DataType.INT64, name=x.name + "_slice_1_ends"))
    slice_reduce_axes = op.initializer(ir.tensor([3], dtype=ir.DataType.INT64, name=x.name + "_transformed_axes"))
    slice_output0 = op.Slice(x, slice_0_starts, slice_0_ends, slice_reduce_axes)
    slice_output1 = op.Slice(x, slice_1_starts, slice_1_ends, slice_reduce_axes)
    reducesum_output0 = op.ReduceSum(slice_output0, slice_reduce_axes)
    reducesum_output1 = op.ReduceSum(slice_output1, slice_reduce_axes)
    return op.Concat(reducesum_output0, reducesum_output1, axis=3)


def transform_reshape_reducesum(model):
    reshape_reducesum_rule = pattern.RewriteRule(reshape_reducesum_pattern, slice_reducesum_concat, verbose=10)
    return onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[reshape_reducesum_rule],
    )


def reshape_clip_reducesum_pattern(op, x, shape, clip_min, clip_max, axes):
    reshape_output = op.Reshape(x, shape)
    clip_output = op.Clip(reshape_output, clip_min, clip_max)
    return op.ReduceSum(clip_output, axes)


def slice_clip_reducesum_concat(op, x, shape, clip_min, clip_max, axes):
    slice_0_starts = op.initializer(ir.tensor([0], dtype=ir.DataType.INT64, name=x.name + "_slice_0_starts"))
    slice_0_ends = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_0_ends"))
    slice_1_starts = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_1_starts"))
    slice_1_ends = op.initializer(ir.tensor([8], dtype=ir.DataType.INT64, name=x.name + "_slice_1_ends"))
    slice_reduce_axes = op.initializer(ir.tensor([3], dtype=ir.DataType.INT64, name=x.name + "_transformed_axes"))
    slice_output0 = op.Slice(x, slice_0_starts, slice_0_ends, slice_reduce_axes)
    slice_output1 = op.Slice(x, slice_1_starts, slice_1_ends, slice_reduce_axes)
    clip_output0 = op.Clip(slice_output0, clip_min, clip_max)
    clip_output1 = op.Clip(slice_output1, clip_min, clip_max)
    reducesum_output0 = op.ReduceSum(clip_output0, slice_reduce_axes)
    reducesum_output1 = op.ReduceSum(clip_output1, slice_reduce_axes)
    return op.Concat(reducesum_output0, reducesum_output1, axis=3)


def transform_reshape_clip_reducesum(model):
    reshape_clip_reducesum_rule = pattern.RewriteRule(
        reshape_clip_reducesum_pattern, slice_clip_reducesum_concat, verbose=10
    )
    return onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[reshape_clip_reducesum_rule],
    )


def reducemax_pattern(op, data, axes, keepdims):
    return op.Reducemax(data, axes, keepdims)


def reducemax_reshape(op, data, axes, keepdims):
    new_axes = op.initializer(ir.tensor([3], dtype=ir.DataType.INT64, name=data.name + "_reducemax_axes"))
    new_keepdims = op.initializer(ir.value(1, dtype=ir.DataType.INT64, name=data.name + "_reducemax_keepdim"))
    reducemax_output = op.Reducemax(data, new_axes, new_keepdims)
    reshape_shape = op.initializer(
        ir.tensor([1, 1, 1, 3600], dtype=ir.DataType.INT64, name=data.name + "_reshape_shape")
    )
    return op.Reshape(reducemax_output, reshape_shape)


def transform_reducemax(model):
    reducemax_rule = pattern.RewriteRule(reducemax_pattern, reducemax_reshape, verbose=10)
    return onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[reducemax_rule],
    )


def transform_add_intermediate_tensors_to_outputs(model, intermediate_tensor_to_add=None):
    # Get existing output names
    existing_outputs = {output.name for output in model.graph.output}

    # Collect all intermediate tensor names from node outputs
    if intermediate_tensor_to_add is None:
        for node in model.graph.node:
            for output in node.output:
                if output and output not in existing_outputs:
                    intermediate_tensor_to_add.add(output)

    # Create ValueInfoProto for each intermediate tensor
    for tensor_name in intermediate_tensor_to_add:
        # Create a new output with default type FLOAT
        output_info = onnx.helper.make_tensor_value_info(
            tensor_name,
            onnx.TensorProto.FLOAT,  # Default to FLOAT type
            None,  # Shape will be inferred if possible
        )
        model.graph.output.append(output_info)
